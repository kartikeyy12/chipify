"""
Chip-style synthesis for Chipify: note events in, authentic 8-bit audio out.

Channels (mirroring the NES 2A03, one voice each)
-------------------------------------------------
pulse 1   lead      band-limited pulse wave, duty 12.5 / 25 / 50 %, 4-bit stepped envelope
pulse 2   harmony   same engine, its own duty and envelope
triangle  bass      32-step, 4-bit triangle (no volume control on real hardware)
noise     drums     15-bit LFSR noise (long and short mode), 4-bit decaying envelope

Authenticity details
--------------------
* Pitch comes from ``NoteEvent.freq_hz``, i.e. already snapped to real chip registers.
* Volumes are integers 0..15 and envelopes update at 60 Hz, so they step audibly.
* Channels are combined with the NES's non-linear mixer (pulse and triangle/noise
  groups saturate differently) rather than a plain sum.
* Pulse waves are band-limited with PolyBLEP so high notes don't alias into
  inharmonic whistles (real hardware aliases; the brief asks us to avoid it).
* Optional "pulse width modulation" sweeps the duty across the NES's duty steps.
* A final bit-crusher gives the "intensity" knob: fewer bits plus sample-and-hold.

No audio device is ever opened: everything is rendered to arrays.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy import signal as sps

from . import config
from .audio_decoder import AudioError
from .chip_tables import DRUM_PATCHES, lfsr_cycle, noise_clock_hz

if TYPE_CHECKING:  # annotations only; keeps this module importable without the analysis stack
    from .pitch_tracker import DrumEvent, NoteEvent, TrackAnalysis

log = logging.getLogger(__name__)

NES_DUTIES: tuple[float, ...] = (0.125, 0.25, 0.5)
# The 2A03 triangle sequencer: 32 steps counting 15 -> 0 -> 15 (note the repeated 0 and 15).
TRIANGLE_STEPS: np.ndarray = np.array(list(range(15, -1, -1)) + list(range(0, 16)), dtype=np.float64)
ENVELOPE_RATE_HZ: float = 60.0       # envelope units are clocked about once per video frame
TRIANGLE_DECLICK_S: float = 0.0015   # tiny fade so note edges don't pop (not on hardware, but kind to ears)
MIXERS: tuple[str, ...] = ("nes", "linear")


class SynthesisError(AudioError):
    """Synthesis could not produce audio."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Adsr:
    """
    ADSR envelope. ``sustain`` is a fraction of the note's peak level. The output
    is quantised to integer volume steps (0..15) at 60 Hz by ``adsr_levels``.
    """
    attack_s: float = 0.0
    decay_s: float = 0.0
    sustain: float = 1.0
    release_s: float = 0.0

    def validate(self) -> None:
        if min(self.attack_s, self.decay_s, self.release_s) < 0:
            raise ValueError("envelope times must be >= 0")
        if not 0.0 <= self.sustain <= 1.0:
            raise ValueError("envelope sustain must be within [0, 1]")


@dataclass
class SynthSettings:
    """Everything the GUI can turn. Defaults sound like a classic NES lead + bass + drums."""
    sample_rate: int = config.OUTPUT_SR
    mixer: str = "nes"                 # "nes" non-linear mixer, or "linear" (Game Boy / ideal)
    lead_duty: float = 0.25
    harmony_duty: float = 0.125
    strict_duty: bool = True           # snap duty cycles to the NES's 12.5 / 25 / 50 % steps
    pwm_depth: float = 0.0             # 0 = fixed duty, 1 = sweep across the whole duty range
    pwm_rate_hz: float = 5.0
    lead_env: Adsr = Adsr(0.0, 0.18, 0.75, 0.04)
    harmony_env: Adsr = Adsr(0.0, 0.12, 0.60, 0.03)
    triangle_dynamics: bool = False    # True lets velocity scale the bass (not authentic, but expressive)
    drum_gain: float = 1.0             # scales the noise channel's volume steps
    bitcrush: float = 0.0              # 0 = clean, 1 = 4-bit samples with heavy sample-and-hold
    normalize_dbfs: float = -1.0

    @classmethod
    def for_chip(cls, chip: str, **overrides) -> "SynthSettings":
        """Sensible defaults per emulated chip (the NES uses its non-linear mixer)."""
        base = {"mixer": "nes" if chip == "nes" else "linear"}
        base.update(overrides)
        return cls(**base)

    def validate(self) -> None:
        if not 8_000 <= int(self.sample_rate) <= 192_000:
            raise ValueError("sample_rate must be within 8000..192000 Hz")
        if self.mixer not in MIXERS:
            raise ValueError(f"mixer must be one of {MIXERS}")
        for name in ("lead_duty", "harmony_duty"):
            if not 0.05 <= getattr(self, name) <= 0.5:
                raise ValueError(f"{name} must be within [0.05, 0.5]")
        if not 0.0 <= self.pwm_depth <= 1.0 or not 0.0 <= self.bitcrush <= 1.0:
            raise ValueError("pwm_depth and bitcrush must be within [0, 1]")
        if not 0.1 <= self.pwm_rate_hz <= 30.0:
            raise ValueError("pwm_rate_hz must be within [0.1, 30]")
        if not 0.0 <= self.drum_gain <= 2.0:
            raise ValueError("drum_gain must be within [0, 2]")
        if not -20.0 <= self.normalize_dbfs <= 0.0:
            raise ValueError("normalize_dbfs must be within [-20, 0]")
        self.lead_env.validate()
        self.harmony_env.validate()


@dataclass
class SynthResult:
    audio: np.ndarray                  # mono float32 in [-1, 1]
    sample_rate: int
    duration_s: float
    channels: dict[str, np.ndarray]    # pre-mix 0..15 level signals, for debugging/visualisation
    note_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Oscillators
# --------------------------------------------------------------------------- #
def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    """PolyBLEP residual: smooths the sample nearest each discontinuity (t = phase in [0,1))."""
    out = np.zeros_like(t)
    rising = t < dt
    x = t[rising] / dt
    out[rising] = x + x - x * x - 1.0
    falling = t > 1.0 - dt
    x = (t[falling] - 1.0) / dt
    out[falling] = x * x + x + x + 1.0
    return out


def pulse_wave(freq_hz: float, n: int, sr: int, duty: float | np.ndarray, band_limited: bool = True) -> np.ndarray:
    """
    Unipolar pulse wave in [0, 1] (the chip outputs 0 or the current volume).

    ``duty`` may be a scalar or a per-sample array (for PWM). Band-limiting uses
    PolyBLEP on both edges; above ~quarter of the sample rate it falls back to the
    naive wave because the correction regions would overlap.
    """
    dt = freq_hz / float(sr)
    phase = (np.arange(n, dtype=np.float64) * dt) % 1.0
    bipolar = np.where(phase < duty, 1.0, -1.0)
    if band_limited and 0.0 < dt < 0.25:
        bipolar += _poly_blep(phase, dt) - _poly_blep((phase - duty) % 1.0, dt)
    return (bipolar + 1.0) * 0.5


def triangle_wave(freq_hz: float, n: int, sr: int) -> np.ndarray:
    """The 2A03's 32-step triangle as integer levels 0..15 (stepped, not band-limited, like the chip)."""
    phase = (np.arange(n, dtype=np.float64) * (freq_hz / float(sr))) % 1.0
    return TRIANGLE_STEPS[(phase * 32.0).astype(np.int64) % 32]


_PREFIX_CACHE: dict[bool, np.ndarray] = {}


def _lfsr_prefix(short_mode: bool) -> np.ndarray:
    """Cumulative sum of one LFSR cycle (length L+1), cached because the long cycle is 32,767 steps."""
    key = bool(short_mode)
    if key not in _PREFIX_CACHE:
        _PREFIX_CACHE[key] = np.concatenate(([0.0], np.cumsum(lfsr_cycle(key).astype(np.float64))))
    return _PREFIX_CACHE[key]


def lfsr_noise(n: int, clock_hz: float, sr: int, short_mode: bool, start_step: float = 0.0) -> np.ndarray:
    """
    Unipolar noise in [0, 1] from the 2A03's 15-bit LFSR.

    The LFSR is clocked at ``clock_hz`` (up to ~450 kHz, far above audio rate), so
    each output sample is the *average* of the LFSR bits inside that sample's
    time span (a box filter built from prefix sums), which avoids aliasing.
    ``start_step`` is where in the cycle the generator is when the hit begins.
    """
    prefix = _lfsr_prefix(short_mode)
    cycle_len = prefix.size - 1
    total = prefix[-1]
    rate = clock_hz / float(sr)
    x = start_step + np.arange(n + 1, dtype=np.float64) * rate
    whole = np.floor(x / cycle_len)
    frac = x - whole * cycle_len
    cumulative = whole * total + np.interp(frac, np.arange(cycle_len + 1), prefix)
    return np.diff(cumulative) / rate


# --------------------------------------------------------------------------- #
# Envelopes (4-bit, stepped at 60 Hz)
# --------------------------------------------------------------------------- #
def _hold_curve(tau: np.ndarray, peak: float, env: Adsr) -> np.ndarray:
    """Attack -> decay -> sustain level at (stepped) times ``tau`` since note start."""
    sustain = peak * env.sustain
    a, d = env.attack_s, env.decay_s
    attack = peak * tau / max(a, 1e-9)
    decay = peak + (sustain - peak) * (tau - a) / max(d, 1e-9)
    return np.where(tau < a, attack, np.where(tau < a + d, decay, sustain))


def adsr_levels(
    peak: int, hold_s: float, env: Adsr, sr: int, max_total_s: float | None = None
) -> np.ndarray:
    """
    Integer volume (0..15) for every output sample of one note.

    The note sounds for ``hold_s`` seconds, then the release tail follows. The
    envelope is evaluated at 60 Hz and held between ticks, so volume moves in
    audible steps like the real chip. ``max_total_s`` truncates the whole note
    (used when the next note on this channel starts).
    """
    total = hold_s + env.release_s
    if max_total_s is not None:
        total = min(total, max_total_s)
    n = int(round(total * sr))
    if n <= 0 or peak <= 0:
        return np.zeros(max(n, 0), dtype=np.float64)

    t = np.arange(n, dtype=np.float64) / sr
    tau = np.floor(t * ENVELOPE_RATE_HZ) / ENVELOPE_RATE_HZ
    level = _hold_curve(tau, float(peak), env)

    if env.release_s > 0:
        hold_tick = math.floor(hold_s * ENVELOPE_RATE_HZ) / ENVELOPE_RATE_HZ
        at_release = float(_hold_curve(np.array([hold_tick]), float(peak), env)[0])
        # Evaluate at the END of each 60 Hz tick so the final tick of the release reaches 0
        # (otherwise the tail would hold level 1 and cut off with a small step).
        fade = np.clip(1.0 - (tau + 1.0 / ENVELOPE_RATE_HZ - hold_s) / env.release_s, 0.0, 1.0)
        level = np.where(t >= hold_s, at_release * fade, level)
    elif total > hold_s:
        level = np.where(t >= hold_s, 0.0, level)

    return np.rint(np.clip(level, 0.0, 15.0))


# --------------------------------------------------------------------------- #
# Channel rendering
# --------------------------------------------------------------------------- #
def _duty_curve(base: float, t_abs: np.ndarray, settings: SynthSettings) -> np.ndarray:
    """Per-sample duty: fixed, or swept by an LFO (PWM); optionally snapped to NES duty steps."""
    duty = np.full(t_abs.shape, base, dtype=np.float64)
    if settings.pwm_depth > 0:
        duty = duty + 0.25 * settings.pwm_depth * np.sin(2.0 * np.pi * settings.pwm_rate_hz * t_abs)
    duty = np.clip(duty, 0.06, 0.5)
    if settings.strict_duty:
        steps = np.asarray(NES_DUTIES)
        duty = steps[np.argmin(np.abs(duty[:, None] - steps[None, :]), axis=1)]
    return duty


def _peak_level(velocity: float) -> int:
    """Velocity 0..1 -> volume 1..15 (never silent: a detected note should be audible)."""
    return int(min(15, max(1, round(velocity * 15.0))))


def _render_pulse(
    notes: Sequence["NoteEvent"], total: int, sr: int, base_duty: float, env: Adsr, settings: SynthSettings
) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    ordered = sorted(notes, key=lambda x: x.start_s)
    for k, note in enumerate(ordered):
        start = int(round(note.start_s * sr))
        if start >= total:
            continue
        limit = (ordered[k + 1].start_s - note.start_s) if k + 1 < len(ordered) else None
        hold = note.duration_s if limit is None else min(note.duration_s, limit)
        levels = adsr_levels(_peak_level(note.velocity), hold, env, sr, max_total_s=limit)
        n = min(levels.size, total - start)
        if n <= 0:
            continue
        t_abs = (start + np.arange(n)) / float(sr)
        duty = _duty_curve(base_duty, t_abs, settings)
        buf[start:start + n] += (pulse_wave(note.freq_hz, n, sr, duty) * levels[:n]).astype(np.float32)
    return buf


def _render_triangle(notes: Sequence["NoteEvent"], total: int, sr: int, settings: SynthSettings) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    ordered = sorted(notes, key=lambda x: x.start_s)
    ramp = max(1, int(TRIANGLE_DECLICK_S * sr))
    for k, note in enumerate(ordered):
        start = int(round(note.start_s * sr))
        end_s = note.end_s if k + 1 >= len(ordered) else min(note.end_s, ordered[k + 1].start_s)
        n = min(int(round((end_s - note.start_s) * sr)), total - start)
        if start >= total or n <= 0:
            continue
        wave = triangle_wave(note.freq_hz, n, sr)
        if settings.triangle_dynamics:
            wave = wave * (0.4 + 0.6 * note.velocity)
        idx = np.arange(n)
        wave = wave * np.minimum(1.0, np.minimum(idx + 1, n - idx) / ramp)
        buf[start:start + n] += wave.astype(np.float32)
    return buf


def _render_noise(drums: Sequence["DrumEvent"], total: int, sr: int, settings: SynthSettings) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    ordered = sorted(drums, key=lambda d: d.time_s)
    for k, hit in enumerate(ordered):
        patch = DRUM_PATCHES.get(hit.kind)
        start = int(round(hit.time_s * sr))
        if patch is None or start >= total:
            continue
        limit = (ordered[k + 1].time_s - hit.time_s) if k + 1 < len(ordered) else None   # noise channel is monophonic
        peak = int(min(15, max(1, round(15.0 * (0.4 + 0.6 * hit.strength) * settings.drum_gain))))
        levels = adsr_levels(peak, patch.decay_s, Adsr(0.0, patch.decay_s, 0.0, 0.0), sr, max_total_s=limit)
        n = min(levels.size, total - start)
        if n <= 0:
            continue
        clock = noise_clock_hz(patch.period_index)
        bits = lfsr_noise(n, clock, sr, patch.short_mode, start_step=hit.time_s * clock)
        buf[start:start + n] = (bits * levels[:n]).astype(np.float32)        # new hit cuts the old one
    return buf


# --------------------------------------------------------------------------- #
# Mixing and post-processing
# --------------------------------------------------------------------------- #
def nes_mix(p1: np.ndarray, p2: np.ndarray, tri: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    The 2A03's non-linear mixer (levels 0..15 per channel, DMC unused).

        pulse_out = 95.88 / (8128 / (p1 + p2) + 100)
        tnd_out   = 159.79 / (1 / (tri/8227 + noise/12241) + 100)

    Compared with a linear sum, loud simultaneous channels compress, which is a
    big part of why NES music sounds the way it does.
    """
    pulse_sum = np.asarray(p1, dtype=np.float64) + p2
    pulse_out = np.where(pulse_sum > 0, 95.88 / (8128.0 / np.maximum(pulse_sum, 1e-9) + 100.0), 0.0)
    denom = np.asarray(tri, dtype=np.float64) / 8227.0 + np.asarray(noise, dtype=np.float64) / 12241.0
    tnd_out = np.where(denom > 0, 159.79 / (1.0 / np.maximum(denom, 1e-12) + 100.0), 0.0)
    return pulse_out + tnd_out


def linear_mix(p1: np.ndarray, p2: np.ndarray, tri: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Plain average of the four 0..15 channels (Game Boy style)."""
    return (np.asarray(p1, dtype=np.float64) + p2 + tri + noise) / 60.0


def dc_block(x: np.ndarray, sr: int, cutoff_hz: float = 20.0) -> np.ndarray:
    """One-pole high-pass; removes the DC offset of the unipolar chip signal (the real console does this too)."""
    if x.size == 0:
        return x
    r = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    b, a = [1.0, -1.0], [1.0, -r]
    zi = sps.lfilter_zi(b, a) * x[0]
    return sps.lfilter(b, a, x, zi=zi)[0]


def bitcrush(x: np.ndarray, intensity: float) -> np.ndarray:
    """
    Lo-fi degradation for the "bit-crush intensity" knob.

    intensity 0 returns the input; 1 gives 4-bit samples and an 8x sample-and-hold
    (about 5.5 kHz effective rate at 44.1 kHz). Input is expected in [-1, 1].
    """
    if intensity <= 0 or x.size == 0:
        return x
    bits = int(round(16 - 12 * intensity))
    hold = 1 + int(round(7 * intensity))
    y = x
    if hold > 1:
        y = np.repeat(x[::hold], hold)[: x.size]
    steps = 2.0 ** (bits - 1)
    return np.round(np.clip(y, -1.0, 1.0) * steps) / steps


def _normalize(x: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1e-9:
        return x
    return x * (10.0 ** (peak_dbfs / 20.0) / peak)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def synthesize(analysis: "TrackAnalysis", settings: SynthSettings | None = None) -> SynthResult:
    """
    Render a ``TrackAnalysis`` to chiptune audio.

    Raises ``SynthesisError`` if there is nothing to play. All numeric problems
    (NaN, overs) are removed before the audio is returned.
    """
    settings = settings or SynthSettings.for_chip(analysis.chip)
    settings.validate()
    sr = int(settings.sample_rate)

    lead, harmony, bass, drums = analysis.lead, analysis.harmony, analysis.bass, analysis.drums
    if not (lead or harmony or bass or drums):
        raise SynthesisError("There are no notes or drum hits to synthesize.")

    tails = [n.end_s + 0.1 for n in (*lead, *harmony, *bass)] + [d.time_s + 0.3 for d in drums]
    length_s = max(analysis.duration_s, max(tails)) + 0.1
    total = int(math.ceil(length_s * sr))
    warnings: list[str] = []
    if total > 15 * 60 * sr:
        raise SynthesisError("The rendered track would be longer than 15 minutes.")

    p1 = _render_pulse(lead, total, sr, settings.lead_duty, settings.lead_env, settings)
    p2 = _render_pulse(harmony, total, sr, settings.harmony_duty, settings.harmony_env, settings)
    tri = _render_triangle(bass, total, sr, settings)
    noise = _render_noise(drums, total, sr, settings)

    mix = nes_mix(p1, p2, tri, noise) if settings.mixer == "nes" else linear_mix(p1, p2, tri, noise)
    audio = dc_block(mix, sr)
    audio = _normalize(audio, -1.0)                       # work at a known level before degrading
    audio = bitcrush(audio, settings.bitcrush)
    audio = np.nan_to_num(_normalize(audio, settings.normalize_dbfs), nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    if float(np.max(np.abs(audio))) < 1e-4:
        raise SynthesisError("The rendered audio is silent.")

    return SynthResult(
        audio=audio,
        sample_rate=sr,
        duration_s=audio.size / float(sr),
        channels={"pulse1": p1, "pulse2": p2, "triangle": tri, "noise": noise},
        note_counts={"pulse1": len(lead), "pulse2": len(harmony), "triangle": len(bass), "noise": len(drums)},
        warnings=tuple(warnings),
    )


def render_to_wav(analysis: "TrackAnalysis", path, settings: SynthSettings | None = None) -> SynthResult:
    """Synthesize and write a 16-bit WAV in one call; returns the SynthResult."""
    from .audio_decoder import write_wav  # local import keeps module import light

    result = synthesize(analysis, settings)
    write_wav(path, result.audio, result.sample_rate)
    return result
