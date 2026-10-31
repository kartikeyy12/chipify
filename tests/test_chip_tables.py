"""Tests for chipify.chip_tables: register maths, quantisation error, LFSR."""
from __future__ import annotations

import numpy as np
import pytest

from chipify import chip_tables as ct


def test_midi_hz_conversions():
    assert ct.midi_to_hz(69) == pytest.approx(440.0)
    assert ct.hz_to_midi(261.6255653) == pytest.approx(60.0, abs=1e-4)
    assert np.allclose(ct.hz_to_midi(np.array([220.0, 440.0, 880.0])), [57, 69, 81])


def test_nes_pulse_follows_hardware_formula():
    q = ct.get_chip("nes").pulse.nearest(440.0)
    assert q.register == 253
    assert q.freq_hz == pytest.approx(ct.NES_CPU_HZ_NTSC / (16 * (q.register + 1)))
    assert abs(q.cents_error) < 5


def test_nes_triangle_is_one_octave_below_pulse_for_same_register():
    nes = ct.get_chip("nes")
    pulse = nes.pulse.nearest(440.0)
    tri = nes.bass.nearest(pulse.freq_hz / 2.0)
    assert tri.register == pulse.register
    assert tri.freq_hz == pytest.approx(pulse.freq_hz / 2.0)


def test_nes_playable_ranges():
    nes = ct.get_chip("nes")
    assert 54.0 < nes.pulse.min_hz < 55.0        # lowest pulse note is just below A1
    assert 27.0 < nes.bass.min_hz < 27.6         # lowest triangle note is just below A0
    assert nes.pulse.max_hz <= ct.MAX_CHIP_HZ
    assert nes.bass.max_hz <= ct.MAX_CHIP_HZ


def test_detune_stays_small_across_the_musical_range():
    nes = ct.get_chip("nes")
    for midi in range(33, 97):                    # A1 .. C7
        q = nes.pulse.nearest(ct.midi_to_hz(midi))
        assert abs(q.cents_error) < 20, f"pulse MIDI {midi}: {q.cents_error:.1f} cents"
    for midi in range(24, 85):                    # C1 .. C6
        q = nes.bass.nearest(ct.midi_to_hz(midi))
        assert abs(q.cents_error) < 20, f"triangle MIDI {midi}: {q.cents_error:.1f} cents"


def test_out_of_range_requests_fold_by_octaves():
    pulse = ct.get_chip("nes").pulse
    low = pulse.nearest(20.0)
    assert low.octave_shift == 2
    assert pulse.min_hz <= low.freq_hz <= pulse.max_hz
    high = pulse.nearest(30_000.0)
    assert high.octave_shift < 0
    assert high.freq_hz <= pulse.max_hz


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_nearest_rejects_invalid_frequencies(bad):
    with pytest.raises(ValueError):
        ct.get_chip("nes").pulse.nearest(bad)


def test_gameboy_formulas():
    gb = ct.get_chip("gameboy")
    q = gb.pulse.nearest(440.0)
    assert q.freq_hz == pytest.approx(131_072.0 / (2048 - q.register))
    assert gb.pulse.min_hz == pytest.approx(64.0)
    assert gb.bass.min_hz == pytest.approx(32.0)
    assert gb.bass_kind == "wave"


def test_ideal_chip_is_exact_equal_temperament():
    q = ct.get_chip("ideal").pulse.nearest(261.6255653)
    assert q.register == 60
    assert abs(q.cents_error) < 1e-3


def test_unknown_chip_raises():
    with pytest.raises(ValueError):
        ct.get_chip("commodore64")


def test_noise_period_table():
    periods = ct.NES_NOISE_PERIODS_NTSC
    assert len(periods) == 16
    assert all(b > a for a, b in zip(periods, periods[1:]))
    assert ct.noise_clock_hz(0) == pytest.approx(ct.NES_CPU_HZ_NTSC / 4)
    with pytest.raises(ValueError):
        ct.noise_clock_hz(16)


def test_lfsr_cycle_lengths_match_hardware():
    long_cycle = ct.lfsr_cycle(False)
    short_cycle = ct.lfsr_cycle(True)
    assert long_cycle.size == 32_767           # maximal-length 15-bit sequence
    assert short_cycle.size == 93              # the metallic short-mode loop
    assert set(np.unique(long_cycle)) <= {0, 1}
    assert set(np.unique(short_cycle)) <= {0, 1}


def test_drum_patches_are_valid():
    assert set(ct.DRUM_PATCHES) == {"kick", "snare", "hat"}
    for patch in ct.DRUM_PATCHES.values():
        assert 0 <= patch.period_index < 16
        assert patch.decay_s > 0
