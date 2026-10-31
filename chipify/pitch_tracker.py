"""
Musical analysis for Chipify: audio in, quantised chip note events out.

Pipeline (see ``analyze``)
--------------------------
1. HPSS splits the signal into a harmonic part (notes) and a percussive part (drums).
2. The harmonic part is band-split by register and tracked with pYIN:
       bass band  -> triangle channel
       lead band  -> pulse 1 channel
3. Frame-wise pitch is segmented into discrete notes (hysteresis, glitch
   removal, re-attack splitting for repeated notes).
4. Pitch quantisation: each note is snapped toward the nearest equal-tempered
   semitone (blend controlled by ``quant_strictness``) and then to the nearest
   register the emulated chip can really produce.
5. Rhythm quantisation: BPM is user-supplied or estimated, then a grid phase
   is fitted to the detected onsets and notes/drums snap to 1/16 (or other) steps.
6. Optional harmony (pulse 2) built from the lead in the detected key.
7. Percussive onsets are classified as kick / snare / hat for the noise channel.

Everything below ``analyze`` is a small, individually testable function.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import librosa
import numpy as np
from scipy import signal as sps

from .audio_decoder import AudioError, DecodedAudio
from .chip_tables import ChipSpec, FrequencyTable, get_chip, hz_to_midi, midi_to_hz

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]

HARMONY_MODES: tuple[str, ...] = ("off", "third_above", "third_below", "octave_up", "octave_down")
PITCH_CLASS_NAMES: tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class AnalysisError(AudioError):
    """The musical analysis could not be completed."""


class NoNotesFoundError(AnalysisError):
    """No pitched notes or drum hits were found in the track."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisSettings:
    """All tunable analysis parameters. Defaults suit typical pop/game instrumentals."""

    # --- user-facing knobs -------------------------------------------------- #
    chip: str = "nes"                 # "nes" | "gameboy" | "ideal"
    quant_strictness: float = 1.0     # 0 = keep detected pitch/timing, 1 = fully snapped
    bpm: float | None = None          # None = estimate from the audio
    steps_per_beat: int = 4           # 4 = 1/16 notes in 4/4
    harmony_mode: str = "off"         # see HARMONY_MODES
    drums_enabled: bool = True

    # --- analysis front end -------------------------------------------------- #
    hop_length: int = 256             # 11.6 ms at 22.05 kHz
    frame_length: int = 2048          # pYIN window
    hpss_kernel: int = 31
    hpss_margin: tuple[float, float] = (1.5, 3.0)   # (harmonic, percussive); >1 = stricter split

    # --- register split ------------------------------------------------------ #
    split_hz: float = 175.0           # bass fundamentals below, lead fundamentals above
    bass_fmin_hz: float = 32.7        # C1
    lead_fmax_hz: float = 2093.0      # C7
    bass_mask_harmonics: int = 4      # bass partials notched out of the lead band (0 = off)
    lead_mask_harmonics: int = 3      # lead partials notched out of the bass band (0 = off)
    drop_bass_harmonic_leads: bool = True   # drop lead notes that are the 2nd/3rd harmonic of a bass note
    bass_source: str = "harmonic"     # "harmonic" = HPSS output, "mix" = unseparated audio (keeps short bass notes)
    fundamental_min_ratio: float = 0.15   # bass frames need this much energy AT their pitch (0 = off)

    # --- note segmentation --------------------------------------------------- #
    voicing_prob_min: float = 0.25
    gate_db: float = 35.0             # bass: ignore frames this far below the band's loudest frame
    lead_gate_db: float = 25.0        # same for the lead band (lower = stricter, rejects quiet leakage)
    octave_fix_ratio: float = 1.5     # floor frames move up an octave if 2f has this much more energy than f (0 = off)
    octave_fix_zone: float = 1.2      # ...but only for pitches below this multiple of the lead floor frequency
    min_note_s: float = 0.06
    pitch_hysteresis: float = 0.2     # extra semitones of tolerance before a note changes
    reattack_db: float = 6.0          # dip-then-rise of this size splits repeated notes
    max_gap_frames: int = 2           # unvoiced gap tolerated inside one note

    # --- drums --------------------------------------------------------------- #
    drum_delta: float = 0.20          # onset peak-pick threshold (lower = more hits)
    drum_min_gap_s: float = 0.05
    drum_min_flatness: float = 0.01   # snare/hat candidates flatter than this are noisy enough to be drums
    kick_min_strength: float = 0.35   # kick candidates must be at least this strong (0..1)

    def validate(self) -> None:
        """Raise ValueError for programmer errors (the GUI clamps its sliders)."""
        get_chip(self.chip)  # raises ValueError for unknown chips
        if not 0.0 <= self.quant_strictness <= 1.0:
            raise ValueError("quant_strictness must be within [0, 1]")
        if self.bpm is not None and not 30.0 <= float(self.bpm) <= 300.0:
            raise ValueError("bpm must be within [30, 300]")
        if self.steps_per_beat not in (1, 2, 3, 4, 6, 8):
            raise ValueError("steps_per_beat must be one of 1, 2, 3, 4, 6, 8")
        if self.bass_source not in ("harmonic", "mix"):
            raise ValueError("bass_source must be 'harmonic' or 'mix'")
        if self.harmony_mode not in HARMONY_MODES:
            raise ValueError(f"harmony_mode must be one of {HARMONY_MODES}")
        if self.hop_length < 64 or self.frame_length < 1024:
            raise ValueError("hop_length >= 64 and frame_length >= 1024 are required")
        if self.min_note_s <= 0 or self.split_hz <= self.bass_fmin_hz * 1.5:
            raise ValueError("invalid min_note_s or split_hz")


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class NoteEvent:
    """One monophonic note on one chip channel."""
    start_s: float
    end_s: float
    midi: float            # target pitch after strictness blending (fractional)
    midi_note: int         # nearest integer note to what the chip actually plays (for MIDI export)
    register: int          # chip timer register
    freq_hz: float         # exact frequency the chip produces
    cents_error: float     # chip pitch minus target pitch
    velocity: float        # 0..1, relative loudness
    channel: str           # "pulse1" | "pulse2" | "triangle"
    octave_shift: int = 0  # octaves the pitch was moved to fit the chip's range (+up / -down)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class DrumEvent:
    """One percussive hit for the noise channel."""
    time_s: float
    kind: str              # "kick" | "snare" | "hat"
    strength: float        # 0..1


@dataclass
class TrackAnalysis:
    """Everything the synthesiser and exporters need, plus plotting data."""
    sr: int
    duration_s: float
    chip: str
    bpm: float
    bpm_source: str                 # "user" | "estimated"
    grid_step_s: float
    grid_offset_s: float
    grid_fit: float                 # 0..1, how well onsets sit on the grid
    key: str | None
    lead: list[NoteEvent]
    harmony: list[NoteEvent]
    bass: list[NoteEvent]
    drums: list[DrumEvent]
    f0_times: np.ndarray            # frame times for the raw pitch curves (plotting)
    f0_lead_hz: np.ndarray
    f0_bass_hz: np.ndarray
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        kinds = {k: sum(1 for d in self.drums if d.kind == k) for k in ("kick", "snare", "hat")}
        lines = [
            f"chip       : {self.chip}",
            f"tempo      : {self.bpm:.1f} BPM ({self.bpm_source}), grid fit {self.grid_fit:.2f}",
            f"key        : {self.key or 'unknown'}",
            f"lead       : {len(self.lead)} notes (pulse 1)",
            f"harmony    : {len(self.harmony)} notes (pulse 2)",
            f"bass       : {len(self.bass)} notes (triangle)",
            f"drums      : {len(self.drums)} hits "
            f"(kick {kinds['kick']}, snare {kinds['snare']}, hat {kinds['hat']})",
        ]
        lines += [f"warning    : {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage 1: HPSS and band splitting
# --------------------------------------------------------------------------- #
def separate_hpss(y: np.ndarray, settings: AnalysisSettings) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic/percussive source separation via median filtering of the STFT."""
    try:
        harmonic, percussive = librosa.effects.hpss(
            y, kernel_size=settings.hpss_kernel, margin=settings.hpss_margin
        )
    except Exception as exc:
        raise AnalysisError(f"Harmonic/percussive separation failed: {exc}") from exc
    return harmonic.astype(np.float32), percussive.astype(np.float32)


def split_bands(
    harmonic: np.ndarray, sr: int, settings: AnalysisSettings
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-phase Butterworth split of the harmonic signal by register.

    bass: low-pass at ~2.5x the split (keeps a few harmonics, which helps pYIN
          lock on bass notes whose fundamental is weak on small speakers)
    lead: high-pass at 0.8x the split (removes the bass fundamental)
    """
    nyq = sr / 2.0

    def sos(kind: str, hz: float):
        wn = min(max(hz / nyq, 1e-4), 0.99)
        return sps.butter(4, wn, btype=kind, output="sos")

    bass = sps.sosfiltfilt(sos("lowpass", settings.split_hz * 2.5), harmonic)
    lead = sps.sosfiltfilt(sos("highpass", settings.split_hz * 0.8), harmonic)
    return bass.astype(np.float32), lead.astype(np.float32)


def suppress_partials(
    band: np.ndarray, sr: int, f0_hz: np.ndarray, hop: int, n_partials: int
) -> np.ndarray:
    """
    Notch the first ``n_partials`` harmonics of a tracked voice out of a band.

    Pitch trackers follow the loudest periodicity. A loud voice therefore hides
    a quieter one that shares its frequency range (bass partials fool the lead
    tracker; a loud lead fools the bass tracker). Once one voice is tracked, we
    know where its partials are in every frame, so we zero those STFT bins and
    track the other voice on what remains. Only the first few partials are
    removed: higher ones are weak, and removing them would also erase notes of
    the other voice that sit an octave-multiple above.
    """
    if n_partials <= 0 or not np.any(np.isfinite(f0_hz)):
        return band
    n_fft = 2048
    try:
        spec = librosa.stft(band, n_fft=n_fft, hop_length=hop)
    except Exception as exc:
        raise AnalysisError(f"Voice separation failed: {exc}") from exc

    bin_hz = sr / float(n_fft)
    f0 = f0_hz[: spec.shape[1]]
    for i in np.flatnonzero(np.isfinite(f0)):
        for k in range(1, n_partials + 1):
            centre = k * float(f0[i])
            half = max(2.0 * bin_hz, 0.03 * centre)
            lo = max(int(np.floor((centre - half) / bin_hz)), 0)
            hi = int(np.ceil((centre + half) / bin_hz)) + 1
            spec[lo:hi, i] = 0.0
    try:
        cleaned = librosa.istft(spec, hop_length=hop, length=band.size)
    except Exception as exc:
        raise AnalysisError(f"Voice separation failed: {exc}") from exc
    return cleaned.astype(np.float32)


def suppress_bass_harmonics(
    band: np.ndarray, sr: int, bass_f0_hz: np.ndarray, settings: AnalysisSettings
) -> np.ndarray:
    """Notch the first ``bass_mask_harmonics`` partials of the bass out of the lead band."""
    return suppress_partials(band, sr, bass_f0_hz, settings.hop_length, settings.bass_mask_harmonics)


def reject_ghost_pitches(
    band: np.ndarray, sr: int, f0_hz: np.ndarray, settings: AnalysisSettings
) -> np.ndarray:
    """
    Remove sub-octave "ghost" pitch frames (returned as NaN).

    If a loud lead note sits above the tracker's search ceiling, pYIN reports
    its sub-octave (f/2, f/4...), because a periodic signal is also periodic at
    multiples of its period. A genuine note has spectral energy at its own
    fundamental; a ghost has none there. So we keep a frame only if the energy
    near the claimed pitch is at least ``fundamental_min_ratio`` of the
    strongest component below 600 Hz.
    """
    out = np.array(f0_hz, dtype=np.float64, copy=True)
    if settings.fundamental_min_ratio <= 0 or not np.any(np.isfinite(out)):
        return out
    n_fft = 4096                                   # 5.4 Hz bins at 22.05 kHz: resolves low bass
    try:
        mag = np.abs(librosa.stft(band, n_fft=n_fft, hop_length=settings.hop_length))
    except Exception as exc:
        raise AnalysisError(f"Bass verification failed: {exc}") from exc

    bin_hz = sr / float(n_fft)
    ceiling = min(mag.shape[0] - 1, int(600.0 / bin_hz))
    n = min(out.size, mag.shape[1])
    for i in np.flatnonzero(np.isfinite(out[:n])):
        centre = out[i] / bin_hz
        half = max(1.0, 0.04 * centre)             # +/-4 % around the claimed pitch
        lo = max(int(np.floor(centre - half)), 0)
        hi = int(np.ceil(centre + half)) + 1
        at_pitch = float(mag[lo:hi, i].max())
        strongest = float(mag[: ceiling + 1, i].max())
        if strongest <= 0.0 or at_pitch < settings.fundamental_min_ratio * strongest:
            out[i] = np.nan
    out[n:] = np.nan
    return out


def correct_octave_errors(
    band: np.ndarray, sr: int, f0_hz: np.ndarray, settings: AnalysisSettings, fmin_hz: float, fmax_hz: float
) -> np.ndarray:
    """
    Move sub-octave errors near the bottom of the tracker's range up one octave.

    A periodic tone repeats every period but also every *two* periods, so pYIN
    sometimes reports f/2 instead of f; with a repeated F4 it flips to F3 at the
    floor of the search range. For frames within ``octave_fix_zone`` of the floor
    we compare the energy at the claimed pitch with the energy at double it; if
    double dominates (``octave_fix_ratio``), the true pitch is the octave above.
    Restricted to the floor zone on purpose: a real melody note higher up may have
    a strong 2nd harmonic, and shifting it would be worse than the error.
    """
    out = np.array(f0_hz, dtype=np.float64, copy=True)
    if settings.octave_fix_ratio <= 0 or not np.any(np.isfinite(out)):
        return out
    n_fft = 4096
    try:
        mag = np.abs(librosa.stft(band, n_fft=n_fft, hop_length=settings.hop_length))
    except Exception as exc:
        raise AnalysisError(f"Octave verification failed: {exc}") from exc

    bin_hz = sr / float(n_fft)

    def level(frame: int, hz: float) -> float:
        centre = hz / bin_hz
        half = max(1.0, 0.04 * centre)
        lo = max(int(np.floor(centre - half)), 0)
        hi = min(int(np.ceil(centre + half)) + 1, mag.shape[0])
        return float(mag[lo:hi, frame].max()) if hi > lo else 0.0

    zone_top = settings.octave_fix_zone * fmin_hz
    for i in np.flatnonzero(np.isfinite(out[: mag.shape[1]])):
        f = out[i]
        if f > zone_top or 2.0 * f > fmax_hz:
            continue
        if level(i, 2.0 * f) >= settings.octave_fix_ratio * max(level(i, f), 1e-12):
            out[i] = 2.0 * f
    return out


# --------------------------------------------------------------------------- #
# Stage 2: frame-wise pitch tracking
# --------------------------------------------------------------------------- #
@dataclass
class _BandTrack:
    times: np.ndarray       # frame centre times (s)
    f0_hz: np.ndarray       # NaN where unvoiced
    midi: np.ndarray        # fractional MIDI, NaN where unvoiced
    energy_db: np.ndarray   # band energy per frame, dB
    peak_db: float          # loudest frame of the band


def _frame_energy_db(band: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    """Per-frame RMS in dB, forced to exactly ``n_frames`` entries."""
    rms = librosa.feature.rms(y=band, frame_length=1024, hop_length=hop, center=True)[0]
    if rms.size < n_frames:
        rms = np.pad(rms, (0, n_frames - rms.size), mode="edge")
    rms = rms[:n_frames]
    return 20.0 * np.log10(np.maximum(rms, 1e-7))


def track_band(
    band: np.ndarray, sr: int, fmin: float, fmax: float, settings: AnalysisSettings, label: str,
    check_fundamental: bool = False, gate_db: float | None = None, fix_octaves: bool = False,
) -> _BandTrack:
    """Run pYIN on one band and gate the result by voicing probability and energy."""
    try:
        f0, flag, prob = librosa.pyin(
            band,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=settings.frame_length,
            hop_length=settings.hop_length,
            resolution=0.2,     # 20-cent pitch bins: plenty for note quantisation, ~5x faster than 0.1
        )
    except Exception as exc:
        raise AnalysisError(f"Pitch tracking failed on the {label} band: {exc}") from exc

    n = int(f0.size)
    energy_db = _frame_energy_db(band, settings.hop_length, n)
    peak_db = float(np.max(energy_db)) if n else -140.0

    voiced = np.isfinite(f0) & np.asarray(flag, dtype=bool) & (np.nan_to_num(prob) >= settings.voicing_prob_min)
    voiced &= energy_db >= (peak_db - (settings.gate_db if gate_db is None else gate_db))
    if check_fundamental and voiced.any():
        checked = reject_ghost_pitches(band, sr, np.where(voiced, f0, np.nan), settings)
        voiced &= np.isfinite(checked)

    f0_clean = np.where(voiced, f0, np.nan)
    if fix_octaves and voiced.any():
        f0_clean = correct_octave_errors(band, sr, f0_clean, settings, fmin, fmax)
    midi = np.where(voiced, hz_to_midi(np.where(voiced, f0_clean, 440.0)), np.nan)
    times = np.arange(n) * settings.hop_length / float(sr)
    return _BandTrack(times, f0_clean, midi, energy_db, peak_db)


# --------------------------------------------------------------------------- #
# Stage 3: note segmentation (pure function, easy to test)
# --------------------------------------------------------------------------- #
@dataclass
class RawNote:
    """A note in frame units, before any quantisation."""
    start: int              # first frame
    end: int                # one past the last frame
    midi_median: float
    mean_db: float


@dataclass
class _Run:
    start: int
    end: int
    label: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _merge_runs(runs: list[_Run], max_gap: int) -> list[_Run]:
    """Join neighbouring runs with the same semitone label separated by <= max_gap frames."""
    out: list[_Run] = []
    for r in runs:
        if out and out[-1].label == r.label and r.start - out[-1].end <= max_gap:
            out[-1].end = r.end
        else:
            out.append(_Run(r.start, r.end, r.label))
    return out


def _split_at_reattacks(
    run: _Run, energy_db: np.ndarray, reattack_db: float, min_frames: int
) -> list[_Run]:
    """
    Split a constant-pitch run wherever the energy dips and then rises again,
    which is how repeated notes of the same pitch ("F F F F") look to a pitch
    tracker that sees one continuous tone.
    """
    a, b = run.start, run.end
    cuts: list[int] = []
    peak = trough = float(energy_db[a])
    trough_i = last_cut = a
    for i in range(a + 1, b):
        v = float(energy_db[i])
        if v < trough:
            trough, trough_i = v, i
        if (
            v - trough >= reattack_db
            and peak - trough >= reattack_db
            and trough_i - last_cut >= min_frames
            and b - trough_i >= min_frames
        ):
            cuts.append(trough_i)
            last_cut = trough_i
            peak = trough = v
            trough_i = i
        elif v > peak:
            peak = trough = v
            trough_i = i
    edges = [a, *cuts, b]
    return [_Run(edges[k], edges[k + 1], run.label) for k in range(len(edges) - 1)]


def segment_notes(
    midi: np.ndarray,
    energy_db: np.ndarray,
    *,
    hysteresis: float = 0.2,
    min_frames: int = 5,
    max_gap_frames: int = 2,
    reattack_db: float = 6.0,
) -> list[RawNote]:
    """
    Turn a frame-wise MIDI-pitch curve (NaN = unvoiced) into discrete notes.

    1. Label each frame with a semitone, with hysteresis so vibrato around a
       boundary does not flicker between two notes.
    2. Group equal labels into runs and bridge tiny unvoiced gaps.
    3. Remove runs shorter than ``min_frames``: absorb into an adjacent run if
       they touch one (pitch glitches, note transitions), else drop them.
    4. Split runs at energy re-attacks so repeated notes become separate notes.
    """
    midi = np.asarray(midi, dtype=np.float64)
    energy_db = np.asarray(energy_db, dtype=np.float64)
    n = midi.size
    if n == 0 or energy_db.size != n:
        return []
    min_frames = max(1, int(min_frames))

    # 1. hysteretic semitone labels (-1 = unvoiced)
    labels = np.full(n, -1, dtype=np.int64)
    prev: int | None = None
    for i in range(n):
        m = midi[i]
        if not np.isfinite(m):
            prev = None
            continue
        lab = prev if (prev is not None and abs(m - prev) <= 0.5 + hysteresis) else int(round(m))
        labels[i] = lab
        prev = lab

    # 2. runs
    runs: list[_Run] = []
    i = 0
    while i < n:
        if labels[i] < 0:
            i += 1
            continue
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append(_Run(i, j, int(labels[i])))
        i = j
    runs = _merge_runs(runs, max_gap_frames)

    # 3. short-run cleanup
    while True:
        k = next((idx for idx, r in enumerate(runs) if r.length < min_frames), None)
        if k is None:
            break
        r = runs[k]
        prev_touch = k > 0 and r.start - runs[k - 1].end <= max_gap_frames
        next_touch = k + 1 < len(runs) and runs[k + 1].start - r.end <= max_gap_frames
        if prev_touch or next_touch:
            options = []
            if prev_touch:
                options.append((abs(runs[k - 1].label - r.label), -runs[k - 1].length, k - 1))
            if next_touch:
                options.append((abs(runs[k + 1].label - r.label), -runs[k + 1].length, k + 1))
            j = min(options)[2]
            if j < k:
                runs[j].end = r.end
            else:
                runs[j].start = r.start
        del runs[k]
        runs = _merge_runs(runs, max_gap_frames)

    # 4. re-attack splitting
    pieces: list[_Run] = []
    for r in runs:
        pieces.extend(_split_at_reattacks(r, energy_db, reattack_db, min_frames))

    notes: list[RawNote] = []
    for r in pieces:
        seg = midi[r.start:r.end]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            continue
        notes.append(
            RawNote(r.start, r.end, float(np.median(finite)), float(np.mean(energy_db[r.start:r.end])))
        )
    return notes


def f0_from_raw_notes(raw: Sequence[RawNote], n_frames: int) -> np.ndarray:
    """
    Expand segmented notes back into a per-frame pitch curve (NaN outside notes).

    pYIN is only confident on part of each note (attacks are often unvoiced), but
    a whole note is sounding for its full length, so notching its partials over
    the full duration removes the leakage that frame-wise tracking misses.
    """
    f0 = np.full(n_frames, np.nan)
    for r in raw:
        f0[r.start:min(r.end, n_frames)] = float(midi_to_hz(r.midi_median))
    return f0


# Semitone offsets of a bass note's 2nd and 3rd harmonics. The 4th harmonic (+24)
# is deliberately NOT listed: a lead two octaves above the bass is common and real.
_BASS_HARMONIC_OFFSETS = (12, 19)


def drop_bass_harmonic_leads(
    lead: Sequence["NoteEvent"], bass: Sequence["NoteEvent"],
    min_overlap: float = 0.5, tolerance_s: float = 0.15,
) -> list["NoteEvent"]:
    """
    Remove lead notes that are really the 2nd or 3rd harmonic of a simultaneous bass note.

    Partials of a loud bass leak into the lead band and get tracked as short,
    low "lead" notes (e.g. F3 over an F2 bass). A lead note is dropped when at
    least ``min_overlap`` of it overlaps a bass note exactly 12 or 19 semitones
    below it. Each bass note is widened by ``tolerance_s`` on both sides because
    the bass tracker locks on later than the lead tracker (low pitches need
    longer analysis windows), so the two onsets rarely line up exactly.
    """
    kept: list[NoteEvent] = []
    for n in lead:
        ghost = False
        for b in bass:
            overlap = min(n.end_s, b.end_s + tolerance_s) - max(n.start_s, b.start_s - tolerance_s)
            if n.duration_s > 0 and overlap / n.duration_s >= min_overlap \
                    and (n.midi_note - b.midi_note) in _BASS_HARMONIC_OFFSETS:
                ghost = True
                break
        if not ghost:
            kept.append(n)
    return kept


# --------------------------------------------------------------------------- #
# Stage 4: pitch quantisation to chip registers
# --------------------------------------------------------------------------- #
def make_note(
    start_s: float,
    end_s: float,
    midi_pitch: float,
    velocity: float,
    channel: str,
    strictness: float,
    table: FrequencyTable,
) -> NoteEvent:
    """
    Build a chip-quantised NoteEvent.

    ``strictness`` blends between the detected pitch (0) and the nearest
    equal-tempered semitone (1); the result is then snapped to the nearest
    register the chip can produce, folding by octaves if it is out of range.
    """
    snapped = float(round(midi_pitch))
    target = strictness * snapped + (1.0 - strictness) * float(midi_pitch)
    q = table.nearest(midi_to_hz(target))
    return NoteEvent(
        start_s=float(start_s),
        end_s=float(end_s),
        midi=float(hz_to_midi(q.target_hz)),
        midi_note=int(round(hz_to_midi(q.freq_hz))),
        register=q.register,
        freq_hz=q.freq_hz,
        cents_error=q.cents_error,
        velocity=float(np.clip(velocity, 0.0, 1.0)),
        channel=channel,
        octave_shift=q.octave_shift,
    )


def _notes_from_raw(
    raw: Sequence[RawNote],
    track: _BandTrack,
    hop: int,
    sr: int,
    settings: AnalysisSettings,
    table: FrequencyTable,
    channel: str,
) -> list[NoteEvent]:
    """Convert frame-unit RawNotes to second-unit, chip-quantised NoteEvents."""
    frame_s = hop / float(sr)
    out: list[NoteEvent] = []
    for r in raw:
        velocity = 1.0 - (track.peak_db - r.mean_db) / 30.0        # 0 dB -> 1.0, -24 dB -> 0.2
        out.append(
            make_note(
                r.start * frame_s, r.end * frame_s, r.midi_median,
                max(0.2, velocity), channel, settings.quant_strictness, table,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 5: tempo, grid fitting and rhythm quantisation
# --------------------------------------------------------------------------- #
def estimate_tempo(y: np.ndarray, sr: int, hop: int) -> float | None:
    """
    Beat-tracker tempo estimate folded into [80, 180) BPM, or None on failure.
    (Beat trackers often report half or double time; the fold picks the octave
    that suits a 1/16 grid. The GUI's BPM knob overrides this.)
    """
    try:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        tempo, _ = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, start_bpm=120.0)
        bpm = float(np.atleast_1d(tempo)[0])
    except Exception as exc:
        log.warning("tempo estimation failed: %s", exc)
        return None
    if not math.isfinite(bpm) or bpm <= 0:
        return None
    while bpm < 80.0:
        bpm *= 2.0
    while bpm >= 180.0:
        bpm /= 2.0
    return bpm


def fit_grid(times: np.ndarray, weights: np.ndarray, step_s: float) -> tuple[float, float]:
    """
    Find the grid phase that best fits onset times (circular mean).

    Returns (offset_s in [0, step_s), fit in [0, 1]); fit 1.0 means every onset
    sits exactly on a grid line, ~0 means onsets are spread evenly (no grid).
    """
    t = np.asarray(times, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if t.size == 0 or step_s <= 0 or w.sum() <= 0:
        return 0.0, 0.0
    z = np.sum(w * np.exp(2j * np.pi * np.mod(t, step_s) / step_s))
    fit = float(abs(z) / w.sum())
    offset = float((np.angle(z) / (2.0 * np.pi)) * step_s) % step_s
    return offset, fit


def refine_bpm(
    times: np.ndarray, weights: np.ndarray, bpm: float, steps_per_beat: int,
    span: float = 0.03, resolution: float = 1e-4,
) -> float:
    """
    Fine-tune an estimated BPM (+/- ``span``) by maximising grid fit.

    A beat tracker is typically 0.5-1% off, which drifts by whole beats over a
    minute; scanning at 0.01% resolution removes that drift.
    """
    t = np.asarray(times, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if t.size < 8 or w.sum() <= 0:
        return bpm
    ratios = np.arange(1.0 - span, 1.0 + span + 1e-12, resolution)
    best_bpm, best_fit = bpm, -1.0
    for chunk in np.array_split(ratios, max(1, ratios.size // 64)):
        steps = 60.0 / (bpm * chunk * steps_per_beat)                       # (c,)
        phase = 2.0 * np.pi * np.mod(t[None, :], steps[:, None]) / steps[:, None]
        fits = np.abs((w[None, :] * np.exp(1j * phase)).sum(axis=1)) / w.sum()
        k = int(np.argmax(fits))
        if fits[k] > best_fit:
            best_fit, best_bpm = float(fits[k]), float(bpm * chunk[k])
    return best_bpm


def _snap_time(t: float, step: float, offset: float, strictness: float) -> float:
    grid = offset + round((t - offset) / step) * step
    return t + strictness * (grid - t)


def snap_notes_to_grid(
    notes: Sequence[NoteEvent], step_s: float, offset_s: float, strictness: float, min_dur_s: float
) -> list[NoteEvent]:
    """
    Snap note starts/ends to the rhythm grid (blended by ``strictness``), keep
    every note at least one grid step long when fully strict, and trim overlaps
    so each channel stays monophonic like the real hardware.
    """
    out: list[NoteEvent] = []
    for n in sorted(notes, key=lambda x: x.start_s):
        gs = round((n.start_s - offset_s) / step_s)
        ge = round((n.end_s - offset_s) / step_s)
        if ge <= gs:
            ge = gs + 1
        start = n.start_s + strictness * ((offset_s + gs * step_s) - n.start_s)
        end = n.end_s + strictness * ((offset_s + ge * step_s) - n.end_s)
        end = max(end, start + min_dur_s)
        out.append(replace(n, start_s=max(0.0, start), end_s=end))

    # enforce monophony: a note ends where the next one begins
    cleaned: list[NoteEvent] = []
    for n in out:
        if cleaned and n.start_s < cleaned[-1].end_s:
            cleaned[-1] = replace(cleaned[-1], end_s=n.start_s)
        cleaned.append(n)
    return [n for n in cleaned if n.duration_s >= 0.01]


def snap_drums_to_grid(
    drums: Sequence[DrumEvent], step_s: float, offset_s: float, strictness: float
) -> list[DrumEvent]:
    """Snap drum hits to the grid; of two same-kind hits landing together, keep the stronger."""
    snapped = [replace(d, time_s=max(0.0, _snap_time(d.time_s, step_s, offset_s, strictness))) for d in drums]
    best: dict[tuple[str, int], DrumEvent] = {}
    for d in snapped:
        key = (d.kind, int(round(d.time_s * 1000.0)))
        if key not in best or d.strength > best[key].strength:
            best[key] = d
    return sorted(best.values(), key=lambda d: d.time_s)


# --------------------------------------------------------------------------- #
# Stage 6: key estimation and harmony
# --------------------------------------------------------------------------- #
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_SCALE_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_SCALE_MINOR = (0, 2, 3, 5, 7, 8, 10)


def estimate_key(notes: Sequence[NoteEvent]) -> tuple[int, str] | None:
    """
    Krumhansl-Schmuckler key estimate from a duration-weighted pitch-class
    histogram. Returns (tonic pitch class 0-11, "major" | "minor") or None.
    Relative keys share the same notes, so harmony is unaffected by that confusion.
    """
    hist = np.zeros(12)
    for n in notes:
        hist[n.midi_note % 12] += max(n.duration_s, 0.0) * (0.5 + 0.5 * n.velocity)
    if hist.sum() <= 0 or np.count_nonzero(hist) < 3:
        return None
    best: tuple[float, int, str] | None = None
    for tonic in range(12):
        for mode, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            c = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            if math.isfinite(c) and (best is None or c > best[0]):
                best = (c, tonic, mode)
    return None if best is None else (best[1], best[2])


def derive_harmony(
    lead: Sequence[NoteEvent],
    key: tuple[int, str] | None,
    mode: str,
    table: FrequencyTable,
    strictness: float,
) -> list[NoteEvent]:
    """
    Build the pulse-2 line from the lead: a diatonic third above/below (needs a
    key) or an octave above/below. Harmony plays quieter than the lead.
    """
    if mode == "off" or not lead:
        return []
    if mode in ("third_above", "third_below"):
        if key is None:
            return []
        tonic, scale_mode = key
        intervals = _SCALE_MAJOR if scale_mode == "major" else _SCALE_MINOR
        scale = np.array([m for m in range(0, 128) if (m - tonic) % 12 in intervals])
        shift = 2 if mode == "third_above" else -2

    out: list[NoteEvent] = []
    for n in lead:
        if mode == "octave_up":
            target = n.midi_note + 12
        elif mode == "octave_down":
            target = n.midi_note - 12
        else:
            idx = int(np.argmin(np.abs(scale - n.midi_note)))
            idx = min(max(idx + shift, 0), scale.size - 1)
            target = int(scale[idx])
        out.append(
            make_note(n.start_s, n.end_s, float(target), n.velocity * 0.7, "pulse2", strictness, table)
        )
    return out


# --------------------------------------------------------------------------- #
# Stage 7: drums
# --------------------------------------------------------------------------- #
def classify_hit(power_spectrum: np.ndarray, freqs: np.ndarray) -> str:
    """
    Classify one percussive hit from its power spectrum.

    kick  : most energy below 200 Hz (or a very low centroid)
    hat   : spectral centroid above 5 kHz
    snare : everything in between
    """
    p = np.asarray(power_spectrum, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    total = float(p.sum())
    if total <= 0:
        return "snare"
    low_ratio = float(p[f < 200.0].sum() / total)
    centroid = float((f * p).sum() / total)
    if low_ratio >= 0.55 or centroid < 500.0:
        return "kick"
    if centroid >= 5000.0:
        return "hat"
    return "snare"


def spectral_flatness(power_spectrum: np.ndarray) -> float:
    """
    Geometric mean / arithmetic mean of a power spectrum, in [0, 1].

    ~1 for noise (drums), ~0 for pitched tones. Leaked synth-note onsets in the
    percussive component score around 1e-5 or lower, snares and hats 0.1-0.4.
    """
    p = np.asarray(power_spectrum, dtype=np.float64) + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def detect_drums(percussive: np.ndarray, sr: int, settings: AnalysisSettings) -> list[DrumEvent]:
    """Onset-detect the percussive component and classify each hit."""
    hop = settings.hop_length
    try:
        env = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=hop)
        wait = max(1, int(settings.drum_min_gap_s * sr / hop))
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=hop, units="frames",
            backtrack=False, delta=settings.drum_delta, wait=wait,
        )
        n_fft = 1024
        mag = np.abs(librosa.stft(percussive, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    except Exception as exc:
        raise AnalysisError(f"Drum detection failed: {exc}") from exc

    env_max = float(env.max()) if env.size and env.max() > 0 else 1.0
    events: list[DrumEvent] = []
    for f in np.asarray(frames, dtype=int):
        lo = min(max(f, 0), mag.shape[1] - 1)
        window = mag[:, lo:lo + 4]                              # ~46 ms right after the onset
        spectrum = np.mean(window ** 2, axis=1)
        strength = float(np.clip(env[min(f, env.size - 1)] / env_max, 0.0, 1.0))
        kind = classify_hit(spectrum, freqs)
        # HPSS leaks the attack of synth notes into the percussive part. Reject those:
        # kicks must be strong, snares/hats must be noisy rather than pitched.
        if kind == "kick":
            if strength < settings.kick_min_strength:
                continue
        elif spectral_flatness(spectrum) < settings.drum_min_flatness:
            continue
        events.append(DrumEvent(float(f * hop / sr), kind, strength))
    return events


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _fmt_key(key: tuple[int, str] | None) -> str | None:
    return None if key is None else f"{PITCH_CLASS_NAMES[key[0]]} {key[1]}"


def analyze(
    audio: DecodedAudio,
    settings: AnalysisSettings | None = None,
    progress: ProgressFn | None = None,
) -> TrackAnalysis:
    """
    Full musical analysis of a loaded track.

    ``progress(fraction, message)`` is called at each stage so a GUI can show a
    progress bar. Raises ``AnalysisError`` subclasses for expected failures.
    """
    settings = settings or AnalysisSettings()
    settings.validate()
    chip: ChipSpec = get_chip(settings.chip)
    y, sr = audio.samples, int(audio.sr)
    warnings: list[str] = []

    def report(frac: float, msg: str) -> None:
        log.info("[%3.0f%%] %s", frac * 100, msg)
        if progress is not None:
            progress(frac, msg)

    # --- 1. HPSS + band split ---------------------------------------------- #
    report(0.02, "Separating melody from drums (HPSS)")
    harmonic, percussive = separate_hpss(y, settings)
    bass_band, lead_band = split_bands(harmonic, sr, settings)
    if settings.bass_source == "mix":
        bass_band = split_bands(y, sr, settings)[0]

    min_frames = max(2, int(round(settings.min_note_s * sr / settings.hop_length)))
    seg_args = dict(
        hysteresis=settings.pitch_hysteresis, min_frames=min_frames,
        max_gap_frames=settings.max_gap_frames, reattack_db=settings.reattack_db,
    )

    # --- 2. Pitch tracking: loudest voice first, then peel it away ------------ #
    # The lead is usually the loudest voice, so track it first on the raw lead
    # band, remove its partials from the bass band, track the bass on what is
    # left, then clean the bass out of the lead band and re-track the lead.
    report(0.15, "Tracking lead pitch (pass 1)")
    lead_first = track_band(
        lead_band, sr, settings.split_hz, settings.lead_fmax_hz, settings, "lead",
        gate_db=settings.lead_gate_db, fix_octaves=True,
    )
    report(0.35, "Isolating the bass line (removing lead partials)")
    bass_band = suppress_partials(bass_band, sr, lead_first.f0_hz, settings.hop_length, settings.lead_mask_harmonics)
    report(0.40, "Tracking bass pitch (pYIN)")
    bass_track = track_band(
        bass_band, sr, settings.bass_fmin_hz, settings.split_hz, settings, "bass", check_fundamental=True
    )
    raw_bass = segment_notes(bass_track.midi, bass_track.energy_db, **seg_args)
    bass_f0_filled = f0_from_raw_notes(raw_bass, bass_track.f0_hz.size)
    if settings.bass_mask_harmonics > 0 and np.any(np.isfinite(bass_f0_filled)):
        report(0.62, "Refining lead pitch (removing bass harmonics)")
        lead_clean = suppress_bass_harmonics(lead_band, sr, bass_f0_filled, settings)
        lead_track = track_band(
            lead_clean, sr, settings.split_hz, settings.lead_fmax_hz, settings, "lead",
            gate_db=settings.lead_gate_db, fix_octaves=True,
        )
    else:
        lead_track = lead_first

    # --- 3. Segmentation ------------------------------------------------------ #
    report(0.68, "Segmenting notes")
    raw_lead = segment_notes(lead_track.midi, lead_track.energy_db, **seg_args)

    # --- 4. Chip pitch quantisation -------------------------------------------- #
    bass = _notes_from_raw(raw_bass, bass_track, settings.hop_length, sr, settings, chip.bass, "triangle")
    lead = _notes_from_raw(raw_lead, lead_track, settings.hop_length, sr, settings, chip.pulse, "pulse1")

    if settings.drop_bass_harmonic_leads and lead and bass:
        before = len(lead)
        lead = drop_bass_harmonic_leads(lead, bass)
        if len(lead) < before:
            warnings.append(f"Removed {before - len(lead)} lead notes that were bass harmonics.")

    # --- 5. Drums ---------------------------------------------------------------- #
    report(0.78, "Detecting drum hits")
    drums = detect_drums(percussive, sr, settings) if settings.drums_enabled else []

    if not (lead or bass or drums):
        raise NoNotesFoundError(
            "No notes or drum hits could be detected. The track may be too noisy, "
            "too quiet, or not tonal (try a cleaner instrumental)."
        )
    if not lead:
        warnings.append("No lead melody was detected above the bass register.")
    if not bass:
        warnings.append("No bass line was detected.")
    folded = sum(1 for n in (*lead, *bass) if n.octave_shift != 0)
    if folded:
        warnings.append(f"{folded} notes were outside the chip's range and moved by octaves.")

    # --- 6. Tempo + rhythm grid --------------------------------------------------- #
    report(0.88, "Fitting tempo grid")
    onset_times = np.array([n.start_s for n in (*lead, *bass)] + [d.time_s for d in drums], dtype=np.float64)
    onset_weights = np.array([n.velocity for n in (*lead, *bass)] + [max(d.strength, 0.1) for d in drums])

    if settings.bpm is not None:
        bpm, bpm_source = float(settings.bpm), "user"
    else:
        est = estimate_tempo(y, sr, settings.hop_length)
        if est is None:
            est = 120.0
            warnings.append("Tempo could not be estimated; assumed 120 BPM. Set the BPM manually.")
        bpm = refine_bpm(onset_times, onset_weights, est, settings.steps_per_beat)
        bpm_source = "estimated"

    step = 60.0 / (bpm * settings.steps_per_beat)
    offset, fit = fit_grid(onset_times, onset_weights, step)
    if onset_times.size >= 8 and fit < 0.35:
        warnings.append(
            f"Weak rhythm-grid fit ({fit:.2f}): the tempo may be wrong or not steady. "
            "Try setting the BPM manually."
        )

    # --- 7. Rhythm quantisation + harmony -------------------------------------------- #
    s = settings.quant_strictness
    min_dur = settings.min_note_s
    lead = snap_notes_to_grid(lead, step, offset, s, min_dur)
    bass = snap_notes_to_grid(bass, step, offset, s, min_dur)
    drums = snap_drums_to_grid(drums, step, offset, s)

    key = estimate_key([*lead, *bass])
    harmony = derive_harmony(lead, key, settings.harmony_mode, chip.pulse, s)
    if settings.harmony_mode.startswith("third") and key is None:
        warnings.append("Key could not be determined, so no harmony line was generated.")

    report(1.0, "Analysis complete")
    return TrackAnalysis(
        sr=sr,
        duration_s=float(y.size / sr),
        chip=chip.name,
        bpm=bpm,
        bpm_source=bpm_source,
        grid_step_s=step,
        grid_offset_s=offset,
        grid_fit=fit,
        key=_fmt_key(key),
        lead=lead,
        harmony=harmony,
        bass=bass,
        drums=drums,
        f0_times=lead_track.times,
        f0_lead_hz=lead_track.f0_hz,
        f0_bass_hz=bass_track.f0_hz,
        warnings=tuple(warnings),
    )
