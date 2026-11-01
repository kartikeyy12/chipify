"""Tests for chipify.visualizer: the numerics are checked directly, the figures for structure and PNG output."""
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from chipify import chip_tables as ct
from chipify import visualizer as vz
from chipify.pitch_tracker import DrumEvent, TrackAnalysis, make_note

NES = ct.get_chip("nes")


def sine(freq, dur, sr, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def analysis(with_notes=True):
    lead = [make_note(0.5, 1.0, 72, 1.0, "pulse1", 1.0, NES.pulse)] if with_notes else []
    bass = [make_note(0.5, 1.5, 36, 0.8, "triangle", 1.0, NES.bass)] if with_notes else []
    drums = [DrumEvent(0.5, "kick", 1.0), DrumEvent(1.0, "hat", 0.4)] if with_notes else []
    times = np.arange(100) * 0.0116
    return TrackAnalysis(
        sr=22_050, duration_s=3.0, chip="nes", bpm=120.0, bpm_source="user", grid_step_s=0.125,
        grid_offset_s=0.0, grid_fit=0.9, key="C major", lead=lead, harmony=[], bass=bass, drums=drums,
        f0_times=times, f0_lead_hz=np.where(times > 0.5, 523.0, np.nan), f0_bass_hz=np.full(100, np.nan),
    )


def is_png(path):
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
def test_note_names():
    assert vz.note_name(60) == "C4" and vz.note_name(69) == "A4" and vz.note_name(36) == "C2"


def test_downsample_envelope_keeps_peaks():
    x = np.zeros(10_000, dtype=np.float32)
    x[1234], x[7777] = 1.0, -1.0
    lo, hi = vz.downsample_envelope(x, 100)
    assert lo.size == hi.size == 100
    assert hi.max() == 1.0 and lo.min() == -1.0                  # plain decimation would likely miss both


def test_downsample_envelope_handles_short_input_and_rejects_empty():
    lo, hi = vz.downsample_envelope(np.array([0.1, 0.2, 0.3], dtype=np.float32), 50)
    assert lo.size == 3
    with pytest.raises(ValueError):
        vz.downsample_envelope(np.array([]), 10)


def test_spectrogram_peak_is_at_the_tone_frequency():
    db, times, freqs = vz.compute_spectrogram_db(sine(1000.0, 1.0, 22_050), 22_050)
    peak_freq = freqs[int(np.argmax(db.mean(axis=1)))]
    assert abs(peak_freq - 1000.0) < 30.0
    assert db.max() == pytest.approx(0.0) and db.min() >= -80.0 - 1e-9
    assert freqs.max() <= 8000.0 and times.size > 10


def test_spectrogram_limits_frequency_to_nyquist_and_rejects_short_input():
    _, _, freqs = vz.compute_spectrogram_db(sine(440.0, 1.0, 8_000), 8_000)
    assert freqs.max() <= 4000.0
    with pytest.raises(ValueError):
        vz.compute_spectrogram_db(np.zeros(100, dtype=np.float32), 22_050)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def test_waveform_comparison_structure_and_png(tmp_path):
    fig = vz.plot_waveform_comparison(sine(220, 2.0, 22_050), 22_050, sine(220, 2.0, 44_100), 44_100)
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    out = vz.save_figure(fig, tmp_path / "plots" / "wave.png")
    assert out.exists() and out.stat().st_size > 1000 and is_png(out)


def test_spectrogram_comparison_structure_and_png(tmp_path):
    fig = vz.plot_spectrogram_comparison(sine(440, 2.0, 22_050), 22_050, sine(440, 2.0, 44_100), 44_100)
    assert isinstance(fig, Figure) and len(fig.axes) == 3         # two spectrograms + colour bar
    out = vz.save_figure(fig, tmp_path / "spec.png")
    assert is_png(out)


def test_piano_roll_structure_and_png(tmp_path):
    fig = vz.plot_piano_roll(analysis(), start_s=0.0, window_s=3.0)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == 2                                   # one lead bar, one bass bar
    assert ax.get_xlim() == (0.0, 3.0)
    assert is_png(vz.save_figure(fig, tmp_path / "roll.png"))


def test_piano_roll_handles_an_analysis_with_no_notes_and_windows_past_the_end(tmp_path):
    assert isinstance(vz.plot_piano_roll(analysis(with_notes=False)), Figure)
    fig = vz.plot_piano_roll(analysis(), start_s=2.99, window_s=20.0)      # window clipped to track length
    assert is_png(vz.save_figure(fig, tmp_path / "edge.png"))


def test_figures_do_not_touch_pyplot_global_state():
    import matplotlib.pyplot as plt
    before = len(plt.get_fignums())
    vz.plot_waveform_comparison(sine(220, 1.0, 22_050), 22_050, sine(220, 1.0, 22_050), 22_050)
    assert len(plt.get_fignums()) == before
