"""Tests for chipify.synth_engine. Everything is synthesised in memory; no audio files needed."""
from __future__ import annotations

import numpy as np
import pytest

from chipify import chip_tables as ct
from chipify import synth_engine as se
from chipify.pitch_tracker import DrumEvent, TrackAnalysis, make_note

SR = 44_100
NES = ct.get_chip("nes")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def lead_note(start, end, midi=69, vel=1.0):
    return make_note(start, end, midi, vel, "pulse1", 1.0, NES.pulse)


def bass_note(start, end, midi=36, vel=1.0):
    return make_note(start, end, midi, vel, "triangle", 1.0, NES.bass)


def analysis(lead=(), harmony=(), bass=(), drums=(), duration=2.0):
    return TrackAnalysis(
        sr=22_050, duration_s=duration, chip="nes", bpm=120.0, bpm_source="user",
        grid_step_s=0.125, grid_offset_s=0.0, grid_fit=1.0, key=None,
        lead=list(lead), harmony=list(harmony), bass=list(bass), drums=list(drums),
        f0_times=np.zeros(0), f0_lead_hz=np.zeros(0), f0_bass_hz=np.zeros(0),
    )


def dominant_hz(x, sr=SR):
    x = x - np.mean(x)                                  # chip waveforms are unipolar: ignore the DC bin
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.fft.rfftfreq(x.size, 1.0 / sr)[int(np.argmax(spec))])


def inharmonic_fraction(x, f0, sr=SR):
    """Share of spectral power that is NOT near a harmonic of f0 (a measure of aliasing)."""
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    harmonic = np.zeros(freqs.size, dtype=bool)
    for k in range(1, int((sr / 2) // f0) + 1):
        harmonic |= np.abs(freqs - k * f0) < 12.0
    return float(power[~harmonic].sum() / power.sum())


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [{"mixer": "tape"}, {"lead_duty": 0.9}, {"pwm_depth": 2.0}, {"bitcrush": -1.0},
     {"pwm_rate_hz": 0.0}, {"drum_gain": 5.0}, {"sample_rate": 100}, {"normalize_dbfs": 3.0}],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        se.SynthSettings(**kwargs).validate()


def test_invalid_envelope_rejected():
    with pytest.raises(ValueError):
        se.Adsr(sustain=1.5).validate()


def test_for_chip_picks_mixer():
    assert se.SynthSettings.for_chip("nes").mixer == "nes"
    assert se.SynthSettings.for_chip("gameboy").mixer == "linear"


# --------------------------------------------------------------------------- #
# Oscillators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("duty", [0.125, 0.25, 0.5])
def test_pulse_mean_equals_duty(duty):
    w = se.pulse_wave(220.0, SR, SR, duty)
    assert float(np.mean(w)) == pytest.approx(duty, abs=0.01)


def test_polyblep_cuts_aliasing_compared_with_naive():
    f = 3135.0
    naive = se.pulse_wave(f, SR, SR, 0.25, band_limited=False)
    smooth = se.pulse_wave(f, SR, SR, 0.25, band_limited=True)
    assert inharmonic_fraction(smooth, f) < 0.5 * inharmonic_fraction(naive, f)


def test_pulse_pitch_is_correct():
    w = se.pulse_wave(440.4, SR, SR, 0.5)
    assert dominant_hz(w) == pytest.approx(440.4, abs=3.0)


def test_pulse_accepts_per_sample_duty_array():
    duty = np.linspace(0.1, 0.5, 4410)
    w = se.pulse_wave(300.0, 4410, SR, duty)
    assert w.shape == (4410,) and np.all((w >= -0.2) & (w <= 1.2))


def test_triangle_table_is_the_2a03_sequence():
    t = se.TRIANGLE_STEPS
    assert t.size == 32 and t.min() == 0 and t.max() == 15
    assert list(t[:16]) == list(range(15, -1, -1)) and list(t[16:]) == list(range(0, 16))


def test_triangle_wave_uses_only_4_bit_steps_at_the_right_pitch():
    w = se.triangle_wave(110.0, SR, SR)
    assert set(np.unique(w)) <= set(range(16))
    assert dominant_hz(w) == pytest.approx(110.0, abs=2.0)


def test_lfsr_noise_statistics_and_modes():
    long_noise = se.lfsr_noise(20_000, 11_025.0, SR, short_mode=False)
    assert 0.4 < float(np.mean(long_noise)) < 0.6
    assert np.all((long_noise >= -1e-9) & (long_noise <= 1 + 1e-9))
    short_noise = se.lfsr_noise(20_000, 11_025.0, SR, short_mode=True)
    lag = 93 * 4                                      # 4 samples per LFSR step, 93-step loop
    a = short_noise - short_noise.mean()
    assert np.dot(a[:-lag], a[lag:]) / np.dot(a, a) > 0.9      # short mode repeats exactly
    b = long_noise - long_noise.mean()
    assert abs(np.dot(b[:-lag], b[lag:]) / np.dot(b, b)) < 0.2  # long mode does not


def test_lfsr_noise_is_deterministic_and_antialiased_at_ultrasonic_clocks():
    a = se.lfsr_noise(1000, 447_443.0, SR, False, start_step=5.0)
    b = se.lfsr_noise(1000, 447_443.0, SR, False, start_step=5.0)
    assert np.array_equal(a, b)
    assert np.all((a >= 0) & (a <= 1))
    assert float(np.std(a)) < 0.25      # averaged bits cluster around 0.5 instead of being 0/1


# --------------------------------------------------------------------------- #
# Envelopes
# --------------------------------------------------------------------------- #
def test_adsr_shape_and_four_bit_steps():
    env = se.Adsr(attack_s=0.1, decay_s=0.1, sustain=0.5, release_s=0.1)
    lv = se.adsr_levels(15, 0.5, env, SR)
    assert lv.size == int(round(0.6 * SR))
    assert set(np.unique(lv)) <= set(range(16))
    assert lv[0] == 0
    assert lv.max() == 15
    assert lv[int(0.35 * SR)] in (7, 8)               # sustain = half of peak
    assert lv[-1] == 0                                # release reaches silence


def test_adsr_is_stepped_at_60_hz():
    env = se.Adsr(0.0, 0.5, 0.0, 0.0)
    lv = se.adsr_levels(15, 0.5, env, SR)
    changes = np.count_nonzero(np.diff(lv))
    assert changes <= int(0.5 * se.ENVELOPE_RATE_HZ) + 1


def test_adsr_truncates_at_next_note():
    env = se.Adsr(0.0, 0.0, 1.0, 0.2)
    assert se.adsr_levels(10, 0.5, env, SR, max_total_s=0.3).size == int(round(0.3 * SR))


def test_adsr_zero_peak_is_silent():
    assert not np.any(se.adsr_levels(0, 0.5, se.Adsr(), SR))


# --------------------------------------------------------------------------- #
# Mixer and post-processing
# --------------------------------------------------------------------------- #
def test_nes_mixer_known_values_and_compression():
    z = np.zeros(1)
    assert se.nes_mix(z, z, z, z)[0] == 0.0
    full = np.full(1, 15.0)
    assert se.nes_mix(full, full, z, z)[0] == pytest.approx(0.2585, abs=1e-3)
    assert se.nes_mix(z, z, full, full)[0] == pytest.approx(0.3733, abs=1e-3)
    one = se.nes_mix(full, z, z, z)[0]
    two = se.nes_mix(full, full, z, z)[0]
    assert one < two < 2 * one                        # non-linear: two channels are less than double


def test_dc_block_removes_offset():
    x = np.full(SR, 0.7)
    assert abs(float(np.mean(se.dc_block(x, SR)))) < 0.01


def test_bitcrush_identity_and_quantisation():
    x = np.sin(np.linspace(0, 40 * np.pi, 20_000))
    assert se.bitcrush(x, 0.0) is x
    y = se.bitcrush(x, 1.0)
    assert len(np.unique(y)) <= 17                    # 4-bit signed: at most 16 levels (+ rounding edge)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_single_lead_note_renders_at_the_chip_pitch():
    n = lead_note(0.2, 1.2, midi=69)
    out = se.synthesize(analysis(lead=[n]))
    seg = out.audio[int(0.4 * SR):int(1.0 * SR)]
    assert dominant_hz(seg) == pytest.approx(n.freq_hz, abs=3.0)
    assert out.audio.dtype == np.float32 and np.all(np.isfinite(out.audio))
    assert float(np.max(np.abs(out.audio))) == pytest.approx(10 ** (-1.0 / 20), abs=1e-3)
    assert out.note_counts["pulse1"] == 1


def test_bass_only_and_drums_only_render():
    bass = se.synthesize(analysis(bass=[bass_note(0.0, 1.0)]))
    assert dominant_hz(bass.audio[int(0.2 * SR):int(0.9 * SR)]) == pytest.approx(65.4, abs=3.0)
    drums = se.synthesize(analysis(drums=[DrumEvent(0.1, "snare", 0.9), DrumEvent(0.6, "hat", 0.5),
                                          DrumEvent(1.0, "kick", 1.0)]))
    assert float(np.max(np.abs(drums.audio))) > 0.5


def test_drum_hits_are_monophonic_and_decay():
    out = se.synthesize(analysis(drums=[DrumEvent(0.5, "snare", 1.0)]))
    noise = out.channels["noise"]
    start = int(0.5 * SR)
    assert noise[start:start + int(0.03 * SR)].max() >= 10
    assert noise[start + int(0.3 * SR):].max() == 0


def test_a_new_note_cuts_the_previous_one_on_the_same_channel():
    a, b = lead_note(0.0, 1.0, 69), lead_note(0.5, 1.0, 72)
    out = se.synthesize(analysis(lead=[a, b]))
    assert np.max(out.channels["pulse1"]) <= 15.0     # no summing beyond one channel's volume range


def test_pwm_changes_the_sound_and_strict_duty_snaps_to_nes_steps():
    t = np.arange(SR) / SR
    s = se.SynthSettings(pwm_depth=1.0, pwm_rate_hz=3.0, strict_duty=True)
    duty = se._duty_curve(0.25, t, s)
    assert set(np.unique(duty)) <= set(se.NES_DUTIES) and len(np.unique(duty)) > 1
    s2 = se.SynthSettings(pwm_depth=0.0, strict_duty=False, lead_duty=0.3)
    assert np.allclose(se._duty_curve(0.3, t, s2), 0.3)


def test_bitcrush_setting_degrades_the_output():
    a = analysis(lead=[lead_note(0.0, 1.0, 72)])
    clean = se.synthesize(a, se.SynthSettings(bitcrush=0.0)).audio
    crushed = se.synthesize(a, se.SynthSettings(bitcrush=1.0)).audio
    assert len(np.unique(crushed)) < len(np.unique(clean))


def test_empty_analysis_raises():
    with pytest.raises(se.SynthesisError):
        se.synthesize(analysis())


def test_linear_mixer_path_for_gameboy_style_output():
    out = se.synthesize(analysis(lead=[lead_note(0.0, 0.5)], bass=[bass_note(0.0, 0.5)]),
                        se.SynthSettings(mixer="linear"))
    assert np.all(np.isfinite(out.audio)) and float(np.max(np.abs(out.audio))) > 0.5


def test_render_to_wav_roundtrip(tmp_path):
    import soundfile as sf
    path = tmp_path / "out.wav"
    se.render_to_wav(analysis(lead=[lead_note(0.0, 0.5)]), path)
    data, sr = sf.read(str(path), dtype="float32")
    assert sr == se.config.OUTPUT_SR and data.size > 0 and np.max(np.abs(data)) <= 1.0
