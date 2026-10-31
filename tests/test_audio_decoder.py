"""Tests for chipify.audio_decoder. Every fixture is synthesised on the fly."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from chipify import audio_decoder as ad
from chipify import config


def sine(freq: float = 440.0, dur: float = 2.0, sr: int = 44_100, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write(path, y, sr, subtype="PCM_16"):
    sf.write(str(path), y, sr, subtype=subtype)
    return path


# ------------------------------------------------------------------ happy path
def test_mono_wav_is_canonicalised(tmp_path):
    f = write(tmp_path / "a.wav", sine(), 44_100)
    out = ad.load_audio(f)
    assert out.sr == config.ANALYSIS_SR
    assert out.samples.ndim == 1
    assert out.samples.dtype == np.float32
    assert out.was_resampled is True
    assert out.decoder == "soundfile"
    assert abs(out.duration_s - 2.0) < 0.01
    target_peak = 10 ** (config.NORMALIZE_PEAK_DBFS / 20)
    assert abs(np.max(np.abs(out.samples)) - target_peak) < 0.01


def test_no_resample_when_already_at_target_rate(tmp_path):
    f = write(tmp_path / "a.wav", sine(sr=22_050), 22_050)
    assert ad.load_audio(f).was_resampled is False


def test_quiet_track_is_normalised_up(tmp_path):
    f = write(tmp_path / "quiet.wav", sine(amp=0.01), 44_100)
    out = ad.load_audio(f)
    assert out.gain_db > 30
    assert np.max(np.abs(out.samples)) > 0.85


def test_normalisation_can_be_disabled(tmp_path):
    f = write(tmp_path / "a.wav", sine(amp=0.2, sr=22_050), 22_050)
    out = ad.load_audio(f, normalize=False)
    assert out.gain_db == 0.0
    assert abs(np.max(np.abs(out.samples)) - 0.2) < 0.01


def test_stereo_is_downmixed(tmp_path):
    y = sine()
    f = write(tmp_path / "st.wav", np.stack([y, y], axis=1), 44_100)
    out = ad.load_audio(f)
    assert out.samples.ndim == 1
    assert out.original_channels == 2


@pytest.mark.parametrize("sr", [8_000, 11_025, 32_000, 48_000, 96_000])
def test_odd_sample_rates(tmp_path, sr):
    f = write(tmp_path / f"sr{sr}.wav", sine(sr=sr, dur=1.5), sr)
    out = ad.load_audio(f)
    assert out.sr == config.ANALYSIS_SR
    assert abs(out.duration_s - 1.5) < 0.02
    assert out.original_sr == sr


def test_dc_offset_is_removed(tmp_path):
    f = write(tmp_path / "dc.wav", sine(amp=0.3) + 0.3, 44_100)
    out = ad.load_audio(f)
    assert abs(float(np.mean(out.samples))) < 1e-3


# -------------------------------------------------------------- failure modes
def test_silent_track_rejected(tmp_path):
    f = write(tmp_path / "zeros.wav", np.zeros(44_100 * 2, dtype=np.float32), 44_100)
    with pytest.raises(ad.SilentAudioError):
        ad.load_audio(f)


def test_near_silent_track_rejected(tmp_path):
    f = write(tmp_path / "hush.wav", sine(amp=1e-4), 44_100, subtype="FLOAT")  # -80 dBFS
    with pytest.raises(ad.SilentAudioError):
        ad.load_audio(f)


def test_too_short_rejected(tmp_path):
    f = write(tmp_path / "blip.wav", sine(dur=0.1), 44_100)
    with pytest.raises(ad.AudioTooShortError):
        ad.load_audio(f)


def test_random_bytes_with_wav_extension_rejected(tmp_path):
    f = tmp_path / "garbage.wav"
    f.write_bytes(os.urandom(8192))
    with pytest.raises(ad.CorruptAudioError):
        ad.load_audio(f)


def test_empty_file_rejected(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    with pytest.raises(ad.CorruptAudioError):
        ad.load_audio(f)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ad.AudioError):
        ad.load_audio(tmp_path / "nope.wav")


def test_none_path_rejected():
    with pytest.raises(ad.AudioError):
        ad.load_audio(None)


def test_unsupported_extension_rejected(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(ad.UnsupportedFormatError):
        ad.load_audio(f)


def test_oversized_file_rejected(tmp_path):
    f = write(tmp_path / "big.wav", sine(), 44_100)
    with pytest.raises(ad.FileTooLargeError):
        ad.load_audio(f, max_file_bytes=1_000)


def test_invalid_target_sr_is_programmer_error(tmp_path):
    f = write(tmp_path / "a.wav", sine(), 44_100)
    with pytest.raises(ValueError):
        ad.load_audio(f, target_sr=10)


# ----------------------------------------------------------------- edge cases
def test_long_file_is_truncated(tmp_path):
    f = write(tmp_path / "long.wav", sine(dur=5.0), 44_100)
    out = ad.load_audio(f, max_duration_s=2.0)
    assert out.was_truncated is True
    assert abs(out.duration_s - 2.0) < 0.02
    assert any("truncated" in w.lower() for w in out.warnings)


def test_nan_samples_are_sanitised(tmp_path):
    y = sine()
    y[1000:1010] = np.nan
    f = write(tmp_path / "nan.wav", y, 44_100, subtype="FLOAT")
    out = ad.load_audio(f)
    assert np.all(np.isfinite(out.samples))
    assert any("non-finite" in w for w in out.warnings)


def test_phase_cancellation_warning(tmp_path):
    y = sine()
    f = write(tmp_path / "phase.wav", np.stack([y, -0.8 * y], axis=1), 44_100)
    out = ad.load_audio(f)
    assert any("phase" in w.lower() for w in out.warnings)


def test_mostly_silent_track_warns(tmp_path):
    y = np.zeros(44_100 * 10, dtype=np.float32)
    y[:2_000] = sine(dur=2_000 / 44_100)[:2_000]
    f = write(tmp_path / "sparse.wav", y, 44_100)
    out = ad.load_audio(f)
    assert any("mostly silent" in w for w in out.warnings)


# --------------------------------------------------------------- ffmpeg paths
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
def test_ffmpeg_fallback_decoder_directly(tmp_path):
    f = write(tmp_path / "a.wav", sine(), 44_100)
    raw = ad._decode_with_ffmpeg(f.resolve(), 22_050, 10.0)
    assert raw.decoder == "ffmpeg"
    assert raw.data.shape[1] == 1
    assert abs(raw.data.shape[0] / 22_050 - 2.0) < 0.1


@needs_ffmpeg
def test_mp3_loads(tmp_path):
    wav = write(tmp_path / "a.wav", sine(), 44_100)
    mp3 = tmp_path / "a.mp3"
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), str(mp3)],
                          capture_output=True)
    if proc.returncode != 0 or not mp3.exists():
        pytest.skip("this ffmpeg build has no MP3 encoder")
    out = ad.load_audio(mp3)
    assert out.sr == config.ANALYSIS_SR
    assert abs(out.duration_s - 2.0) < 0.3  # MP3 encoder padding shifts length slightly


# ------------------------------------------------------------------ write_wav
def test_write_wav_roundtrip_and_limiting(tmp_path):
    y = 2.0 * sine(sr=44_100)       # deliberately exceeds full scale
    y[10] = np.nan                   # and contains a poison sample
    path = ad.write_wav(tmp_path / "sub" / "out.wav", y, 44_100)
    back, sr = sf.read(str(path), dtype="float32")
    assert sr == 44_100
    assert np.max(np.abs(back)) <= 1.0
    assert np.all(np.isfinite(back))


def test_write_wav_rejects_empty(tmp_path):
    with pytest.raises(ad.AudioWriteError):
        ad.write_wav(tmp_path / "x.wav", np.array([], dtype=np.float32), 44_100)
