"""
Tests for chipify.pitch_tracker.

Most tests exercise pure functions with hand-built inputs, so they are fast and
deterministic. The two tests marked "integration" run the real HPSS/pYIN stack
on synthetic audio.
"""
from __future__ import annotations

import numpy as np
import pytest

from chipify import chip_tables as ct
from chipify import pitch_tracker as pt
from chipify.audio_decoder import DecodedAudio

SR = 22_050
NES = ct.get_chip("nes")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def seq(*parts):
    """Build a frame array from (value, n_frames) pairs; value may be nan."""
    return np.concatenate([np.full(n, v, dtype=np.float64) for v, n in parts])


def note(start, end, midi=60.0, vel=1.0, channel="pulse1", strictness=1.0, table=None):
    return pt.make_note(start, end, midi, vel, channel, strictness, table or NES.pulse)


def tone(midi, dur, sr=SR, amp=0.4):
    """Harmonically rich tone (4 partials) with short attack/release."""
    t = np.arange(int(dur * sr)) / sr
    f = float(ct.midi_to_hz(midi))
    y = sum(np.sin(2 * np.pi * f * h * t) / h for h in range(1, 5))
    env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.02))
    return (amp * y * env / 1.6).astype(np.float32)


def render(notes, total_s, sr=SR):
    y = np.zeros(int(total_s * sr), dtype=np.float32)
    for midi, start, dur in notes:
        seg = tone(midi, dur, sr)
        i = int(start * sr)
        y[i:i + seg.size] += seg
    return y


def decoded(y, sr=SR):
    return DecodedAudio(
        samples=y, sr=sr, duration_s=y.size / sr, source_path="synthetic",
        original_sr=sr, original_channels=1, decoder="synthetic",
        was_resampled=False, was_truncated=False,
        gain_db=0.0, peak_dbfs=-1.0, rms_dbfs=-12.0,
    )


# --------------------------------------------------------------------------- #
# Settings validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"quant_strictness": 1.5},
        {"bpm": 10.0},
        {"steps_per_beat": 5},
        {"harmony_mode": "fifths"},
        {"chip": "atari"},
        {"hop_length": 16},
    ],
)
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        pt.AnalysisSettings(**kwargs).validate()


def test_default_settings_are_valid():
    pt.AnalysisSettings().validate()


# --------------------------------------------------------------------------- #
# Note segmentation
# --------------------------------------------------------------------------- #
def test_two_pitches_become_two_notes():
    midi = seq((60.0, 30), (64.0, 30))
    out = pt.segment_notes(midi, np.full(60, -20.0))
    assert len(out) == 2
    assert out[0].midi_median == pytest.approx(60.0)
    assert out[1].midi_median == pytest.approx(64.0)
    assert out[0].end == out[1].start


def test_vibrato_stays_one_note():
    midi = 60.0 + 0.3 * np.sin(np.linspace(0, 12 * np.pi, 80))
    out = pt.segment_notes(midi, np.full(80, -20.0))
    assert len(out) == 1


def test_single_frame_pitch_glitch_is_absorbed():
    midi = seq((60.0, 20), (72.0, 1), (60.0, 20))
    out = pt.segment_notes(midi, np.full(41, -20.0))
    assert len(out) == 1
    assert out[0].midi_median == pytest.approx(60.0)


def test_short_isolated_blip_is_dropped():
    midi = seq((np.nan, 10), (60.0, 2), (np.nan, 10))
    assert pt.segment_notes(midi, np.full(22, -20.0), min_frames=5) == []


def test_long_gap_makes_two_notes_but_tiny_gap_is_bridged():
    energy = np.full(42, -20.0)
    bridged = pt.segment_notes(seq((60.0, 20), (np.nan, 2), (60.0, 20)), energy)
    assert len(bridged) == 1
    energy = np.full(50, -20.0)
    split = pt.segment_notes(seq((60.0, 20), (np.nan, 10), (60.0, 20)), energy)
    assert len(split) == 2


def test_energy_reattack_splits_repeated_same_pitch_notes():
    midi = np.full(44, 60.0)
    energy = seq((-10.0, 20), (-30.0, 4), (-10.0, 20))
    out = pt.segment_notes(midi, energy, reattack_db=6.0, min_frames=5)
    assert len(out) == 2
    assert all(n.midi_median == pytest.approx(60.0) for n in out)


def test_gentle_decay_is_not_split():
    midi = np.full(40, 60.0)
    energy = np.linspace(-10.0, -20.0, 40)
    assert len(pt.segment_notes(midi, energy)) == 1


def test_segment_notes_handles_empty_and_all_unvoiced():
    assert pt.segment_notes(np.array([]), np.array([])) == []
    assert pt.segment_notes(seq((np.nan, 30)), np.full(30, -20.0)) == []


# --------------------------------------------------------------------------- #
# Pitch quantisation
# --------------------------------------------------------------------------- #
def test_full_strictness_snaps_to_semitone_then_chip_register():
    n = note(0, 1, midi=60.4, strictness=1.0)
    assert n.midi_note == 60
    assert n.midi == pytest.approx(60.0, abs=0.05)
    assert abs(n.cents_error) < 20


def test_zero_strictness_keeps_detected_pitch():
    n = note(0, 1, midi=60.4, strictness=0.0)
    assert n.midi == pytest.approx(60.4, abs=0.05)


def test_half_strictness_lands_between():
    n = note(0, 1, midi=60.4, strictness=0.5)
    assert n.midi == pytest.approx(60.2, abs=0.05)


def test_make_note_records_octave_fold_and_stays_in_range():
    n = note(0, 1, midi=20.0)                      # ~26 Hz, below the pulse range
    assert n.octave_shift == 2
    assert NES.pulse.min_hz <= n.freq_hz <= NES.pulse.max_hz


def test_make_note_clamps_velocity():
    assert note(0, 1, vel=5.0).velocity == 1.0
    assert note(0, 1, vel=-1.0).velocity == 0.0


# --------------------------------------------------------------------------- #
# Rhythm grid
# --------------------------------------------------------------------------- #
def test_fit_grid_recovers_phase():
    times = 0.037 + 0.125 * np.arange(40)
    offset, fit = pt.fit_grid(times, np.ones(40), 0.125)
    assert offset == pytest.approx(0.037, abs=1e-6)
    assert fit == pytest.approx(1.0, abs=1e-6)


def test_fit_grid_on_random_times_is_low():
    rng = np.random.default_rng(0)
    times = rng.uniform(0, 60, 400)
    _, fit = pt.fit_grid(times, np.ones(400), 0.125)
    assert fit < 0.3


def test_fit_grid_handles_empty_input():
    assert pt.fit_grid(np.array([]), np.array([]), 0.125) == (0.0, 0.0)


def test_refine_bpm_removes_tempo_estimation_error():
    true_bpm, steps = 137.3, 4
    step = 60.0 / (true_bpm * steps)
    rng = np.random.default_rng(1)
    times = np.arange(0, 500) * step + rng.normal(0, 0.004, 500)
    refined = pt.refine_bpm(times, np.ones(500), 135.0, steps)   # start 1.7% off
    assert refined == pytest.approx(true_bpm, rel=1e-3)


def test_refine_bpm_with_too_few_onsets_returns_input():
    assert pt.refine_bpm(np.array([0.0, 0.5]), np.ones(2), 120.0, 4) == 120.0


def test_strict_snap_lands_on_grid():
    out = pt.snap_notes_to_grid([note(0.13, 0.30), note(0.40, 0.60)], 0.125, 0.0, 1.0, 0.05)
    for n in out:
        assert n.start_s / 0.125 == pytest.approx(round(n.start_s / 0.125), abs=1e-9)
        assert n.end_s / 0.125 == pytest.approx(round(n.end_s / 0.125), abs=1e-9)


def test_zero_strictness_leaves_times_alone():
    out = pt.snap_notes_to_grid([note(0.13, 0.30)], 0.125, 0.0, 0.0, 0.05)
    assert out[0].start_s == pytest.approx(0.13)
    assert out[0].end_s == pytest.approx(0.30)


def test_strict_snap_enforces_one_step_minimum():
    out = pt.snap_notes_to_grid([note(0.13, 0.15)], 0.125, 0.0, 1.0, 0.05)
    assert out[0].duration_s == pytest.approx(0.125)


def test_snapped_notes_never_overlap():
    notes = [note(0.10, 0.50), note(0.20, 0.60), note(0.55, 0.90)]
    out = pt.snap_notes_to_grid(notes, 0.125, 0.0, 1.0, 0.05)
    for a, b in zip(out, out[1:]):
        assert b.start_s >= a.end_s - 1e-9


def test_snap_drums_merges_collisions_keeping_strongest():
    drums = [pt.DrumEvent(0.49, "hat", 0.3), pt.DrumEvent(0.51, "hat", 0.9), pt.DrumEvent(0.50, "kick", 0.5)]
    out = pt.snap_drums_to_grid(drums, 0.125, 0.0, 1.0)
    hats = [d for d in out if d.kind == "hat"]
    assert len(hats) == 1 and hats[0].strength == 0.9
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# Key estimation and harmony
# --------------------------------------------------------------------------- #
def c_major_phrase():
    pitches = [60, 62, 64, 65, 67, 67, 64, 60, 67, 72, 67, 64, 60, 60, 67, 60]
    return [note(i * 0.25, i * 0.25 + 0.25, midi=p) for i, p in enumerate(pitches)]


def test_estimate_key_finds_c_major():
    assert pt.estimate_key(c_major_phrase()) == (0, "major")


def test_estimate_key_needs_enough_pitch_classes():
    assert pt.estimate_key([note(0, 1, 60.0), note(1, 2, 60.0)]) is None
    assert pt.estimate_key([]) is None


def test_harmony_thirds_follow_the_key():
    lead = [note(0, 0.25, 60), note(0.25, 0.5, 62), note(0.5, 0.75, 64)]
    above = pt.derive_harmony(lead, (0, "major"), "third_above", NES.pulse, 1.0)
    below = pt.derive_harmony(lead, (0, "major"), "third_below", NES.pulse, 1.0)
    assert [h.midi_note for h in above] == [64, 65, 67]
    assert [h.midi_note for h in below] == [57, 59, 60]
    assert all(h.channel == "pulse2" for h in above)
    assert all(h.velocity < l.velocity for h, l in zip(above, lead))
    assert [h.start_s for h in above] == [l.start_s for l in lead]


def test_harmony_octaves_and_off_and_missing_key():
    lead = [note(0, 0.25, 64)]
    assert pt.derive_harmony(lead, None, "octave_up", NES.pulse, 1.0)[0].midi_note == 76
    assert pt.derive_harmony(lead, None, "octave_down", NES.pulse, 1.0)[0].midi_note == 52
    assert pt.derive_harmony(lead, (0, "major"), "off", NES.pulse, 1.0) == []
    assert pt.derive_harmony(lead, None, "third_above", NES.pulse, 1.0) == []


# --------------------------------------------------------------------------- #
# Drum classification
# --------------------------------------------------------------------------- #
def test_classify_hit_separates_kick_snare_hat():
    freqs = np.fft.rfftfreq(1024, 1.0 / SR)
    kick = ((freqs < 150)).astype(float)
    snare = ((freqs > 1000) & (freqs < 4000)).astype(float)
    hat = ((freqs > 7000)).astype(float)
    assert pt.classify_hit(kick, freqs) == "kick"
    assert pt.classify_hit(snare, freqs) == "snare"
    assert pt.classify_hit(hat, freqs) == "hat"
    assert pt.classify_hit(np.zeros_like(freqs), freqs) == "snare"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def test_analyze_raises_when_nothing_is_found(monkeypatch):
    empty = pt._BandTrack(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), -140.0)
    monkeypatch.setattr(pt, "separate_hpss", lambda y, s: (y, y))
    monkeypatch.setattr(pt, "track_band", lambda *a, **k: empty)
    monkeypatch.setattr(pt, "detect_drums", lambda *a, **k: [])
    y = np.zeros(SR * 2, dtype=np.float32)
    with pytest.raises(pt.NoNotesFoundError):
        pt.analyze(decoded(y), pt.AnalysisSettings(bpm=120.0))


def test_analyze_reports_progress_in_order(monkeypatch):
    lead_notes = [(72, 0.5, 0.45), (76, 1.0, 0.45)]
    y = render(lead_notes, 2.0)
    seen = []
    pt.analyze(decoded(y), pt.AnalysisSettings(bpm=120.0, drums_enabled=False),
               progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0


# --------------------------------------------------------------------------- #
# Integration: real HPSS + pYIN on synthetic audio
# --------------------------------------------------------------------------- #
def test_integration_lead_melody_is_tracked_and_quantised():
    spec = [(72, 0.5, 0.45), (76, 1.0, 0.45), (79, 1.5, 0.45), (84, 2.0, 0.45)]   # C5 E5 G5 C6
    out = pt.analyze(decoded(render(spec, 3.0)), pt.AnalysisSettings(bpm=120.0, drums_enabled=False))
    assert [n.midi_note for n in out.lead] == [72, 76, 79, 84]
    for n, (_, start, _) in zip(out.lead, spec):
        assert abs(n.start_s - start) < 0.08
    assert out.bpm == 120.0 and out.bpm_source == "user"


def test_integration_bass_line_goes_to_triangle_channel():
    spec = [(36, 0.5, 0.45), (31, 1.0, 0.45), (41, 1.5, 0.45)]                    # C2 G1 F2
    out = pt.analyze(decoded(render(spec, 2.5)), pt.AnalysisSettings(bpm=120.0, drums_enabled=False))
    assert [n.midi_note for n in out.bass] == [36, 31, 41]
    assert all(n.channel == "triangle" for n in out.bass)


def test_integration_noise_bursts_are_detected_as_drum_hits():
    rng = np.random.default_rng(3)
    y = np.zeros(SR * 4, dtype=np.float32)
    hits = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for t0 in hits:
        n = int(0.04 * SR)
        burst = rng.uniform(-1, 1, n) * np.exp(-np.arange(n) / (0.01 * SR))
        burst = burst - np.convolve(burst, np.ones(4) / 4, mode="same")          # crude high-pass
        y[int(t0 * SR):int(t0 * SR) + n] += (0.8 * burst).astype(np.float32)
    events = pt.detect_drums(y, SR, pt.AnalysisSettings())
    found = sum(any(abs(e.time_s - t) < 0.05 for e in events) for t in hits)
    assert found >= 5
    assert sum(e.kind == "hat" for e in events) >= 4


def test_integration_bass_harmonics_do_not_hijack_the_lead():
    """Regression: bass partials (130/196/261 Hz...) used to be reported as the melody."""
    lead = [(72, 0.5, 0.45), (74, 1.0, 0.45), (76, 1.5, 0.45), (77, 2.0, 0.45)]     # C5 D5 E5 F5
    bass = [(36, 0.5, 2.0)]                                                         # C2 held underneath
    out = pt.analyze(decoded(render(lead + bass, 3.0)), pt.AnalysisSettings(bpm=120.0, drums_enabled=False))
    assert [n.midi_note for n in out.lead] == [72, 74, 76, 77]
    assert [n.midi_note for n in out.bass] == [36]


def test_suppress_bass_harmonics_is_a_noop_without_bass():
    y = render([(72, 0.2, 0.5)], 1.0)
    s = pt.AnalysisSettings()
    assert pt.suppress_bass_harmonics(y, SR, np.full(10, np.nan), s) is y
    assert pt.suppress_bass_harmonics(y, SR, np.full(10, 65.0), pt.AnalysisSettings(bass_mask_harmonics=0)) is y


def test_spectral_flatness_separates_noise_from_tones():
    rng = np.random.default_rng(0)
    t = np.arange(1024) / SR
    tone_power = np.abs(np.fft.rfft(np.hanning(1024) * tone_wave(523.0, t))) ** 2
    noise_power = np.abs(np.fft.rfft(np.hanning(1024) * rng.uniform(-1, 1, 1024))) ** 2
    assert pt.spectral_flatness(tone_power) < 1e-3
    assert pt.spectral_flatness(noise_power) > 0.1


def tone_wave(freq, t):
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 5))


def test_integration_tonal_note_onsets_are_not_drums():
    """Regression: synth-note attacks leaking through HPSS were reported as ~150 snares."""
    spec = [(72, 0.5 * k, 0.4) for k in range(1, 7)]                 # six C5 notes
    events = pt.detect_drums(render(spec, 4.0), SR, pt.AnalysisSettings())
    assert len(events) <= 1


def test_reject_ghost_pitches_keeps_real_bass_and_drops_sub_octave_ghosts():
    sr, hop = SR, 256
    t = np.arange(SR * 4) / sr
    real = sum(np.sin(2 * np.pi * 87.3 * k * t) / k for k in range(1, 5))       # genuine F2 bass
    lead = sum(np.sin(2 * np.pi * 349.2 * k * t) / k for k in range(1, 4))      # F4 lead, no energy at 175 Hz
    band = np.where(t < 2.0, real, lead).astype(np.float32) * 0.3
    n = 1 + band.size // hop
    times = np.arange(n) * hop / sr
    claimed = np.where(times < 2.0, 87.3, 174.6)                                # tracker claims F2, then the F3 ghost
    out = pt.reject_ghost_pitches(band, sr, claimed, pt.AnalysisSettings())
    first, second = out[(times > 0.3) & (times < 1.7)], out[(times > 2.3) & (times < 3.7)]
    assert np.all(np.isfinite(first))
    assert np.all(np.isnan(second))


def test_reject_ghost_pitches_can_be_disabled():
    f0 = np.full(10, 100.0)
    out = pt.reject_ghost_pitches(np.zeros(2560, dtype=np.float32), SR, f0,
                                  pt.AnalysisSettings(fundamental_min_ratio=0.0))
    assert np.array_equal(out, f0)


def test_bass_source_is_validated():
    with pytest.raises(ValueError):
        pt.AnalysisSettings(bass_source="vinyl").validate()


def test_integration_lead_alone_creates_no_ghost_bass():
    """Regression from real Axel F: lead notes reappeared as an octave-down 'bass' line."""
    spec = [(65, 0.5, 0.45), (68, 1.0, 0.45), (65, 1.5, 0.45), (70, 2.0, 0.45)]      # F4 Ab4 F4 Bb4
    out = pt.analyze(decoded(render(spec, 3.0)), pt.AnalysisSettings(bpm=120.0, drums_enabled=False))
    assert [n.midi_note for n in out.lead] == [65, 68, 65, 70]
    assert out.bass == []


def test_suppress_partials_removes_the_tracked_voice_but_keeps_the_rest():
    t = np.arange(SR * 2) / SR
    bass = 0.1 * np.sin(2 * np.pi * 87.3 * t)
    lead = 0.4 * np.sin(2 * np.pi * 349.2 * t)
    mixed = (bass + lead).astype(np.float32)
    n = 1 + mixed.size // 256
    cleaned = pt.suppress_partials(mixed, SR, np.full(n, 349.2), 256, 2)
    spec_in = np.abs(np.fft.rfft(mixed))
    spec_out = np.abs(np.fft.rfft(cleaned))
    freqs = np.fft.rfftfreq(mixed.size, 1.0 / SR)
    at = lambda spec, hz: spec[np.argmin(np.abs(freqs - hz))]
    assert at(spec_out, 349.2) < 0.1 * at(spec_in, 349.2)       # lead removed (>20 dB)
    assert at(spec_out, 87.3) > 0.8 * at(spec_in, 87.3)         # bass untouched


def test_suppress_partials_is_a_noop_when_disabled():
    y = render([(72, 0.2, 0.5)], 1.0)
    assert pt.suppress_partials(y, SR, np.full(10, 349.0), 256, 0) is y


def test_integration_quiet_bass_under_a_loud_lead_is_found():
    """Regression from real Axel F: a loud lead inside the bass band hid the bass entirely."""
    t = np.arange(int(3.0 * SR)) / SR
    bass = 0.05 * sum(np.sin(2 * np.pi * 87.3 * k * t) / k for k in range(1, 5))      # F2, about -20 dB
    lead = render([(65, 0.5, 0.45), (68, 1.0, 0.45), (65, 1.5, 0.45), (70, 2.0, 0.45)], 3.0)
    out = pt.analyze(decoded((bass.astype(np.float32) + lead)), pt.AnalysisSettings(bpm=120.0, drums_enabled=False))
    assert [n.midi_note for n in out.lead] == [65, 68, 65, 70]
    assert out.bass and {n.midi_note for n in out.bass} == {41}


def test_drop_bass_harmonic_leads_removes_octave_and_twelfth_ghosts_only():
    bass = [note(0.0, 1.0, 41, table=NES.bass, channel="triangle")]               # F2
    lead = [
        note(0.1, 0.3, 53),     # F3 = 2nd harmonic of F2   -> dropped
        note(0.3, 0.5, 60),     # C4 = 3rd harmonic of F2   -> dropped
        note(0.5, 0.7, 65),     # F4 = a real lead (two octaves up) -> kept
        note(1.5, 1.7, 53),     # F3 but the bass has stopped -> kept
        note(1.0, 1.6, 53),     # only 25 % falls within the bass note +/- tolerance -> kept
    ]
    kept = pt.drop_bass_harmonic_leads(lead, bass)
    assert [n.midi_note for n in kept] == [65, 53, 53]


def test_drop_bass_harmonic_leads_tolerates_late_bass_onsets():
    bass = [note(0.50, 1.00, 41, table=NES.bass, channel="triangle")]
    early_ghost = note(0.38, 0.46, 53)                 # ends 40 ms before the bass tracker locked on
    assert pt.drop_bass_harmonic_leads([early_ghost], bass) == []
    assert pt.drop_bass_harmonic_leads([early_ghost], bass, tolerance_s=0.0) == [early_ghost]


def test_f0_from_raw_notes_fills_whole_note_duration():
    raw = [pt.RawNote(start=2, end=6, midi_median=41.0, mean_db=-20.0)]
    f0 = pt.f0_from_raw_notes(raw, 10)
    assert np.all(np.isnan(f0[:2])) and np.all(np.isnan(f0[6:]))
    assert np.allclose(f0[2:6], ct.midi_to_hz(41.0))


def _octave_fixture():
    t = np.arange(SR * 2) / SR
    n = 1 + t.size // 256
    return t, n


def test_correct_octave_errors_lifts_floor_ghosts_to_the_true_pitch():
    t, n = _octave_fixture()
    lead = sum(np.sin(2 * np.pi * 349.2 * k * t) / k for k in range(1, 4))       # F4 lead
    leak = 0.3 * np.sin(2 * np.pi * 174.6 * t)                                   # bass 2nd-harmonic leakage at F3
    band = (0.3 * (lead + leak)).astype(np.float32)
    out = pt.correct_octave_errors(band, SR, np.full(n, 174.8), pt.AnalysisSettings(), 175.0, 2093.0)
    assert np.allclose(out, 349.6, rtol=1e-3)


def test_correct_octave_errors_keeps_a_genuine_low_note():
    t, n = _octave_fixture()
    square = sum(np.sin(2 * np.pi * 174.6 * k * t) / k for k in (1, 3, 5))       # odd harmonics only
    out = pt.correct_octave_errors((0.3 * square).astype(np.float32), SR, np.full(n, 174.8),
                                   pt.AnalysisSettings(), 175.0, 2093.0)
    assert np.allclose(out, 174.8)


def test_correct_octave_errors_ignores_pitches_above_the_floor_zone():
    t, n = _octave_fixture()
    band = (0.3 * sum(np.sin(2 * np.pi * 349.2 * k * t) / k for k in range(1, 4))).astype(np.float32)
    claimed = np.full(n, 349.2)                       # real F4 with strong partials must never be shifted up
    out = pt.correct_octave_errors(band, SR, claimed, pt.AnalysisSettings(), 175.0, 2093.0)
    assert np.allclose(out, 349.2)
