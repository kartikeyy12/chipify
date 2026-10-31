"""
Vintage sound-chip frequency tables, noise-channel parameters and LFSR model.

Why this exists
---------------
A real chip cannot play arbitrary frequencies: pitch comes from an integer
timer register, so every note is a slightly detuned version of the equal-
temperament note you wanted. Quantising to those register values is what makes
the output sound like the hardware (the slight sourness of high NES notes is
authentic).

Hardware formulas implemented here
----------------------------------
NES 2A03 (NTSC, CPU = 1,789,773 Hz), 11-bit timer ``t`` (0..2047):
    pulse     f = CPU / (16 * (t + 1))      (timer < 8 is muted by the sweep unit)
    triangle  f = CPU / (32 * (t + 1))      (one octave below pulse for equal t)
    noise     15-bit LFSR clocked at CPU / period[index] (16 fixed periods)

Game Boy LR35902, 11-bit register ``x`` (0..2047):
    pulse     f = 131072 / (2048 - x)
    wave      f = 65536  / (2048 - x)       (32-sample wave RAM, used for bass)

"ideal" is an unconstrained 12-TET table (MIDI 12..120) for the Arcade preset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

A4_HZ: float = 440.0
NES_CPU_HZ_NTSC: float = 1_789_773.0

# Nothing musical lives above this; it also filters out ultrasonic register
# values (e.g. NES triangle with a tiny timer) that real hardware treats as silence.
MAX_CHIP_HZ: float = 12_500.0


# --------------------------------------------------------------------------- #
# Pitch conversion helpers (accept scalars or arrays)
# --------------------------------------------------------------------------- #
def midi_to_hz(midi):
    """MIDI note number (fractional allowed) -> frequency in Hz."""
    out = A4_HZ * np.power(2.0, (np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)
    return float(out) if out.ndim == 0 else out


def hz_to_midi(hz):
    """Frequency in Hz -> fractional MIDI note number (NaN/<=0 -> NaN or -inf)."""
    arr = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 69.0 + 12.0 * np.log2(arr / A4_HZ)
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# Frequency table with nearest-register lookup
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuantizedPitch:
    """Result of snapping a requested frequency to a chip register."""
    register: int          # raw timer/register value to program into the chip
    freq_hz: float         # frequency the chip will really produce
    target_hz: float       # requested frequency AFTER octave folding
    octave_shift: int      # octaves the request was moved to fit (+up / -down)

    @property
    def cents_error(self) -> float:
        """Chip pitch minus requested pitch, in cents (+ = chip is sharp)."""
        return 1200.0 * math.log2(self.freq_hz / self.target_hz)


class FrequencyTable:
    """
    All pitches one channel can produce, sorted by ascending frequency.

    Lookup is done in the log-frequency domain, so "nearest" means nearest in
    cents (what the ear hears), not nearest in Hz.
    """

    def __init__(self, name: str, registers: np.ndarray, freqs: np.ndarray) -> None:
        regs = np.asarray(registers, dtype=np.int64)
        f = np.asarray(freqs, dtype=np.float64)
        if regs.ndim != 1 or regs.shape != f.shape or regs.size < 2:
            raise ValueError("registers and freqs must be 1-D arrays of equal length >= 2")
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            raise ValueError("frequencies must be positive and finite")

        order = np.argsort(f, kind="stable")
        self.name = name
        self.registers: np.ndarray = regs[order]
        self.freqs: np.ndarray = f[order]
        self._log2: np.ndarray = np.log2(self.freqs)

        # Requests up to a quarter-tone outside the table snap to its edge;
        # anything further is moved by whole octaves into range.
        quarter_tone = 2.0 ** (1.0 / 24.0)
        self._lo = float(self.freqs[0]) / quarter_tone
        self._hi = float(self.freqs[-1]) * quarter_tone
        if self._hi / self._lo < 2.0:
            raise ValueError("a frequency table must span at least one octave")

    @property
    def min_hz(self) -> float:
        return float(self.freqs[0])

    @property
    def max_hz(self) -> float:
        return float(self.freqs[-1])

    def fold(self, freq_hz: float) -> tuple[float, int]:
        """Octave-shift ``freq_hz`` into the playable range -> (folded_hz, octaves)."""
        f = float(freq_hz)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError(f"cannot fold non-positive/non-finite frequency {freq_hz!r}")
        octaves = 0
        while f < self._lo:
            f *= 2.0
            octaves += 1
        while f > self._hi:
            f /= 2.0
            octaves -= 1
        return f, octaves

    def nearest(self, freq_hz: float) -> QuantizedPitch:
        """Snap ``freq_hz`` (octave-folded if needed) to the closest register."""
        folded, octaves = self.fold(freq_hz)
        x = math.log2(folded)
        i = int(np.searchsorted(self._log2, x))
        lo, hi = max(i - 1, 0), min(i, self._log2.size - 1)
        j = lo if abs(self._log2[lo] - x) <= abs(self._log2[hi] - x) else hi
        return QuantizedPitch(
            register=int(self.registers[j]),
            freq_hz=float(self.freqs[j]),
            target_hz=folded,
            octave_shift=octaves,
        )


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def _nes_pulse_table(cpu_hz: float = NES_CPU_HZ_NTSC) -> FrequencyTable:
    regs = np.arange(8, 2048)                      # timer < 8 is muted on real hardware
    freqs = cpu_hz / (16.0 * (regs + 1.0))
    keep = freqs <= MAX_CHIP_HZ
    return FrequencyTable("nes_pulse", regs[keep], freqs[keep])


def _nes_triangle_table(cpu_hz: float = NES_CPU_HZ_NTSC) -> FrequencyTable:
    regs = np.arange(0, 2048)
    freqs = cpu_hz / (32.0 * (regs + 1.0))
    keep = freqs <= MAX_CHIP_HZ                     # removes the ultrasonic registers
    return FrequencyTable("nes_triangle", regs[keep], freqs[keep])


def _gb_pulse_table() -> FrequencyTable:
    regs = np.arange(0, 2048)
    freqs = 131_072.0 / (2048.0 - regs)
    keep = freqs <= MAX_CHIP_HZ
    return FrequencyTable("gb_pulse", regs[keep], freqs[keep])


def _gb_wave_table() -> FrequencyTable:
    regs = np.arange(0, 2048)
    freqs = 65_536.0 / (2048.0 - regs)
    keep = freqs <= MAX_CHIP_HZ
    return FrequencyTable("gb_wave", regs[keep], freqs[keep])


def _ideal_table() -> FrequencyTable:
    notes = np.arange(12, 121)                      # C0 .. C9; register == MIDI note
    return FrequencyTable("ideal", notes, midi_to_hz(notes))


@dataclass(frozen=True)
class ChipSpec:
    """The pitch capabilities of one emulated chip."""
    name: str
    description: str
    pulse: FrequencyTable      # lead + harmony channels
    bass: FrequencyTable       # bass channel
    bass_kind: str             # "triangle" | "wave" | "ideal"


CHIP_NAMES: tuple[str, ...] = ("nes", "gameboy", "ideal")


@lru_cache(maxsize=None)
def get_chip(name: str) -> ChipSpec:
    """Return the (cached) ChipSpec for ``name``; ValueError if unknown."""
    key = str(name).strip().lower()
    if key == "nes":
        return ChipSpec("nes", "Ricoh 2A03 (NTSC)", _nes_pulse_table(), _nes_triangle_table(), "triangle")
    if key == "gameboy":
        return ChipSpec("gameboy", "Game Boy LR35902", _gb_pulse_table(), _gb_wave_table(), "wave")
    if key == "ideal":
        table = _ideal_table()
        return ChipSpec("ideal", "Unconstrained 12-TET", table, table, "ideal")
    raise ValueError(f"Unknown chip '{name}'. Choose one of: {', '.join(CHIP_NAMES)}")


# --------------------------------------------------------------------------- #
# NES noise channel: fixed period table, LFSR model and drum patches
# --------------------------------------------------------------------------- #
# CPU cycles between LFSR shifts for noise register values 0..15 (NTSC).
NES_NOISE_PERIODS_NTSC: tuple[int, ...] = (
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
)
NOISE_LFSR_BITS: int = 15
NOISE_LFSR_SEED: int = 1
NOISE_TAP_LONG: int = 1     # feedback = bit0 XOR bit1  -> 32,767-step white-ish noise
NOISE_TAP_SHORT: int = 6    # feedback = bit0 XOR bit6  -> short metallic loop


def noise_clock_hz(period_index: int, cpu_hz: float = NES_CPU_HZ_NTSC) -> float:
    """LFSR shift rate in Hz for noise period index 0..15 (higher = brighter)."""
    if not 0 <= int(period_index) < len(NES_NOISE_PERIODS_NTSC):
        raise ValueError("noise period index must be 0..15")
    return cpu_hz / NES_NOISE_PERIODS_NTSC[int(period_index)]


@lru_cache(maxsize=2)
def lfsr_cycle(short_mode: bool) -> np.ndarray:
    """
    One full cycle of the 2A03 noise generator as uint8 samples (1 = sound,
    0 = silenced), one entry per LFSR shift.

    Hardware behaviour: shift right, new bit 14 = bit0 XOR (bit1 | bit6),
    and the output is silenced while bit0 is set.
    """
    tap = NOISE_TAP_SHORT if short_mode else NOISE_TAP_LONG
    reg = NOISE_LFSR_SEED
    out: list[int] = []
    for _ in range(1 << NOISE_LFSR_BITS):          # hard bound: state space size
        out.append(0 if (reg & 1) else 1)
        feedback = (reg & 1) ^ ((reg >> tap) & 1)
        reg = (reg >> 1) | (feedback << (NOISE_LFSR_BITS - 1))
        if reg == NOISE_LFSR_SEED:
            break
    else:  # pragma: no cover - the LFSR update is a bijection, so the seed always recurs
        raise RuntimeError("LFSR did not return to its seed")
    return np.asarray(out, dtype=np.uint8)


@dataclass(frozen=True)
class DrumPatch:
    """Noise-channel settings that approximate one drum voice (tune by ear in Phase 4)."""
    kind: str
    period_index: int     # 0..15, lower = brighter
    short_mode: bool      # True = metallic 'tssk' texture
    decay_s: float        # envelope length


DRUM_PATCHES: dict[str, DrumPatch] = {
    "kick":  DrumPatch("kick", 12, False, 0.14),
    "snare": DrumPatch("snare", 5, False, 0.16),
    "hat":   DrumPatch("hat", 1, True, 0.045),
}
