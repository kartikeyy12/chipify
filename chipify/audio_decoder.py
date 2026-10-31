"""
Defensive audio loading for Chipify.

Public API
----------
load_audio(path, ...) -> DecodedAudio
    Decode any supported file into the canonical analysis format:
    mono, float32, ``target_sr`` Hz, finite, DC-free, peak-normalised.

write_wav(path, samples, sr) -> Path
    Safely write a mono/stereo float signal as 16-bit PCM WAV.

Decoding strategy
-----------------
1. ``soundfile`` (libsndfile) handles WAV/FLAC/OGG, and MP3 on libsndfile>=1.1.
2. If that fails for any reason, fall back to an ``ffmpeg`` subprocess, which
   decodes essentially anything and can salvage partially damaged files.
3. If both fail, raise ``CorruptAudioError`` with a message fit for the UI.

No audio *device* is ever opened, so WSL2 needs no ALSA/PulseAudio.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions (messages are written to be shown directly to the end user)
# --------------------------------------------------------------------------- #
class AudioError(Exception):
    """Base class for all expected, user-presentable audio problems."""


class UnsupportedFormatError(AudioError):
    """File extension or sample rate is not supported."""


class FileTooLargeError(AudioError):
    """File exceeds the configured size limit."""


class CorruptAudioError(AudioError):
    """File could not be decoded, or contains no audio."""


class SilentAudioError(AudioError):
    """Decoded audio is effectively silent; there is nothing to convert."""


class AudioTooShortError(AudioError):
    """Decoded audio is shorter than the minimum useful duration."""


class AudioWriteError(AudioError):
    """Writing an output file failed."""


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class _RawDecode:
    """Internal result of a low-level decode, before normalisation."""
    data: np.ndarray            # shape (n_samples, n_channels), float32
    sr: int                     # sample rate of ``data``
    original_sr: int            # sample rate of the source file (for reporting)
    original_channels: int      # channel count of the source file
    truncated: bool             # True if the file was longer than the cap
    decoder: str                # "soundfile" or "ffmpeg"
    notes: tuple[str, ...] = ()


@dataclass
class DecodedAudio:
    """Canonical analysis-ready audio plus provenance and quality metadata."""
    samples: np.ndarray         # mono float32 in [-1, 1]
    sr: int                     # == target_sr
    duration_s: float
    source_path: str
    original_sr: int
    original_channels: int
    decoder: str
    was_resampled: bool
    was_truncated: bool
    gain_db: float              # gain applied by normalisation (0.0 if disabled)
    peak_dbfs: float            # measured BEFORE normalisation (after DC removal)
    rms_dbfs: float             # measured BEFORE normalisation (after DC removal)
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"file       : {Path(self.source_path).name}",
            f"decoder    : {self.decoder}",
            f"source     : {self.original_sr} Hz, {self.original_channels} ch",
            f"canonical  : {self.sr} Hz, mono, {self.duration_s:.2f} s",
            f"level      : peak {self.peak_dbfs:.1f} dBFS, RMS {self.rms_dbfs:.1f} dBFS "
            f"(gain applied {self.gain_db:+.1f} dB)",
        ]
        lines += [f"warning    : {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _to_dbfs(x: float) -> float:
    """Linear amplitude -> dBFS, floored to avoid log(0)."""
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _rms(x: np.ndarray) -> float:
    """RMS computed in float64 so float32 overflow/rounding cannot bite."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _validate_path(path: str | Path | None, max_file_bytes: int) -> Path:
    """Check existence, type, extension, and size before touching a decoder."""
    if path is None or str(path).strip() == "":
        raise AudioError("No audio file was provided.")

    p = Path(str(path)).expanduser()
    if not p.exists():
        raise AudioError(f"File not found: {p.name}")
    if not p.is_file():
        raise AudioError(f"Not a regular file: {p.name}")

    ext = p.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(config.SUPPORTED_EXTENSIONS))
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext or '(none)'}'. Supported types: {allowed}."
        )

    size = p.stat().st_size
    if size == 0:
        raise CorruptAudioError("The file is empty (0 bytes).")
    if size > max_file_bytes:
        raise FileTooLargeError(
            f"File is {size / 1_048_576:.1f} MiB; the limit is "
            f"{max_file_bytes / 1_048_576:.0f} MiB."
        )
    return p.resolve()


# --------------------------------------------------------------------------- #
# Low-level decoders
# --------------------------------------------------------------------------- #
def _decode_with_soundfile(path: Path, max_duration_s: float) -> _RawDecode:
    """
    Decode via libsndfile, reading at most ``max_duration_s`` of audio so a huge
    file never has to be fully loaded into memory.
    """
    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate)
        channels = int(f.channels)
        if sr <= 0 or channels <= 0:
            raise CorruptAudioError("Header reports an invalid sample rate or channel count.")

        limit = max(1, int(round(max_duration_s * sr)))
        data = f.read(frames=limit, dtype="float32", always_2d=True)
        # Peek one more frame to learn whether we cut the file short.
        truncated = f.read(frames=1, dtype="float32", always_2d=True).shape[0] > 0

    if data.shape[0] == 0:
        raise CorruptAudioError("Decoder returned zero audio samples.")
    return _RawDecode(data, sr, sr, channels, truncated, "soundfile")


def _probe_stream(ffprobe: str, path: Path) -> tuple[int, int] | None:
    """Best-effort (sample_rate, channels) lookup for reporting purposes only."""
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=config.FFPROBE_TIMEOUT_S, check=False,
        )
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream["sample_rate"]), int(stream["channels"])
    except Exception:  # purely informational; never let it fail a load
        return None


def _decode_with_ffmpeg(path: Path, target_sr: int, max_duration_s: float) -> _RawDecode:
    """
    Fallback decoder. ffmpeg downmixes to mono and resamples to ``target_sr``
    for us, streaming raw float32 PCM over a pipe (no temp files).
    Partially damaged files often still yield usable audio before the error.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CorruptAudioError(
            "The file could not be decoded natively and ffmpeg is not installed "
            "(install it with: sudo apt install ffmpeg)."
        )

    # +1 s margin lets us detect "file was longer than the cap".
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-t", f"{max_duration_s + 1.0:.3f}",
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=config.FFMPEG_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise CorruptAudioError("Decoding timed out; the file may be corrupt.") from exc
    except OSError as exc:
        raise CorruptAudioError(f"Could not run ffmpeg: {exc}") from exc

    notes: list[str] = []
    if proc.returncode != 0:
        stderr_lines = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        reason = stderr_lines[-1] if stderr_lines else "unknown decoder error"
        if not proc.stdout:
            raise CorruptAudioError(f"The file is corrupt or not valid audio ({reason}).")
        notes.append("The decoder reported errors; the file may be damaged and the audio incomplete.")

    usable_bytes = (len(proc.stdout) // 4) * 4
    data = np.frombuffer(proc.stdout[:usable_bytes], dtype="<f4").astype(np.float32)
    if data.size == 0:
        raise CorruptAudioError("Decoder returned zero audio samples.")

    limit = int(round(max_duration_s * target_sr))
    truncated = data.size > limit
    if truncated:
        data = data[:limit]

    ffprobe = shutil.which("ffprobe")
    probed = _probe_stream(ffprobe, path) if ffprobe else None
    original_sr, original_channels = probed if probed else (target_sr, 1)

    return _RawDecode(data[:, None], target_sr, original_sr, original_channels,
                      truncated, "ffmpeg", tuple(notes))


def _to_mono(data: np.ndarray, warnings: list[str]) -> np.ndarray:
    """
    Average channels to mono. Warns when the mono sum is much quieter than the
    individual channels, which indicates out-of-phase stereo (phase cancellation)
    that would starve the pitch tracker of signal.
    """
    if data.shape[1] == 1:
        return data[:, 0].astype(np.float32, copy=False)

    mono = data.mean(axis=1, dtype=np.float64).astype(np.float32)
    channel_rms = _rms(data)
    if channel_rms > 1e-9 and (_rms(mono) / channel_rms) < config.PHASE_CANCEL_RATIO_WARN:
        warnings.append(
            "Stereo channels partially cancel when summed to mono (phase problem); "
            "pitch detection may be unreliable."
        )
    return mono


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_audio(
    path: str | Path,
    target_sr: int = config.ANALYSIS_SR,
    max_duration_s: float = config.MAX_DURATION_S,
    normalize: bool = True,
    max_file_bytes: int = config.MAX_FILE_BYTES,
) -> DecodedAudio:
    """
    Load an audio file into the canonical analysis format.

    Parameters
    ----------
    path            Path to a .wav/.mp3/.flac/.ogg file.
    target_sr       Output sample rate in Hz.
    max_duration_s  Audio beyond this length is discarded (with a warning).
    normalize       Peak-normalise to ``config.NORMALIZE_PEAK_DBFS``.
    max_file_bytes  Reject files larger than this before decoding.

    Raises
    ------
    AudioError (or a subclass) for every expected failure. ``ValueError`` is
    raised only for programmer errors such as an invalid ``target_sr``.
    """
    if not (config.MIN_VALID_SR <= int(target_sr) <= config.MAX_VALID_SR):
        raise ValueError(f"target_sr {target_sr} is outside the valid range.")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive.")

    src = _validate_path(path, max_file_bytes)
    warnings: list[str] = []

    # --- 1. Decode: soundfile first, ffmpeg as the safety net -------------- #
    raw: _RawDecode | None = None
    try:
        raw = _decode_with_soundfile(src, max_duration_s)
    except Exception as exc:  # libsndfile raises several unrelated types; all mean "fall back"
        log.info("soundfile could not decode %s (%s); trying ffmpeg", src.name, exc)
    if raw is None:
        raw = _decode_with_ffmpeg(src, int(target_sr), max_duration_s)

    warnings.extend(raw.notes)
    if raw.truncated:
        warnings.append(f"Audio longer than {max_duration_s:.0f} s was truncated.")

    # --- 2. Validate sample rate ------------------------------------------- #
    if not (config.MIN_VALID_SR <= raw.sr <= config.MAX_VALID_SR):
        raise UnsupportedFormatError(
            f"Unsupported sample rate ({raw.sr} Hz). "
            f"Expected {config.MIN_VALID_SR}-{config.MAX_VALID_SR} Hz."
        )
    if raw.sr < config.LOW_SR_WARN_BELOW:
        warnings.append(
            f"Low source sample rate ({raw.sr} Hz); high-frequency content is limited."
        )

    # --- 3. Sanitise non-finite samples (NaN/Inf in float WAVs) ------------ #
    data = raw.data
    bad = ~np.isfinite(data)
    if bad.any():
        warnings.append(f"Replaced {int(bad.sum())} non-finite samples with silence.")
        data = np.where(bad, 0.0, data).astype(np.float32)

    # --- 4. Downmix to mono ------------------------------------------------- #
    mono = _to_mono(data, warnings)

    # --- 5. Clipping report (measured on the source-rate signal) ----------- #
    clipped_fraction = float(np.mean(np.abs(mono) >= config.CLIPPING_THRESHOLD))
    if clipped_fraction > config.CLIPPING_WARN_FRACTION:
        warnings.append(
            f"{clipped_fraction * 100:.1f}% of samples are clipped; the source is heavily distorted."
        )

    # --- 6. Resample to the analysis rate ---------------------------------- #
    was_resampled = raw.sr != int(target_sr)
    if was_resampled:
        try:
            mono = librosa.resample(mono, orig_sr=raw.sr, target_sr=int(target_sr), res_type="soxr_hq")
        except Exception as exc:
            raise CorruptAudioError(f"Resampling failed: {exc}") from exc

    # --- 7. Remove DC offset (after resampling so the mean is exactly zero) - #
    mono = (mono - np.mean(mono, dtype=np.float64)).astype(np.float32)

    # --- 8. Length and silence gates --------------------------------------- #
    duration_s = mono.size / float(target_sr)
    if duration_s < config.MIN_DURATION_S:
        raise AudioTooShortError(
            f"Audio is only {duration_s:.2f} s long; at least {config.MIN_DURATION_S:.1f} s is required."
        )

    peak = float(np.max(np.abs(mono)))
    rms = _rms(mono)
    peak_dbfs, rms_dbfs = _to_dbfs(peak), _to_dbfs(rms)
    if peak_dbfs < config.SILENCE_PEAK_DBFS or rms_dbfs < config.SILENCE_RMS_DBFS:
        raise SilentAudioError(
            f"The track is silent (peak {peak_dbfs:.0f} dBFS); there is no music to convert."
        )

    # --- 9. Sparse-content warning ----------------------------------------- #
    frame_rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    if frame_rms.size and frame_rms.max() > 0:
        floor = frame_rms.max() * 10.0 ** (-config.SPARSE_ACTIVITY_DB / 20.0)
        if float(np.mean(frame_rms > floor)) < config.SPARSE_ACTIVITY_WARN_FRACTION:
            warnings.append("The track is mostly silent; few notes will be detected.")

    # --- 10. Peak normalisation -------------------------------------------- #
    gain_db = 0.0
    if normalize:
        gain = (10.0 ** (config.NORMALIZE_PEAK_DBFS / 20.0)) / peak
        mono = (mono * gain).astype(np.float32)
        gain_db = _to_dbfs(gain)

    return DecodedAudio(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sr=int(target_sr),
        duration_s=duration_s,
        source_path=str(src),
        original_sr=raw.original_sr,
        original_channels=raw.original_channels,
        decoder=raw.decoder,
        was_resampled=was_resampled,
        was_truncated=raw.truncated,
        gain_db=gain_db,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        warnings=tuple(warnings),
    )


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    """
    Write ``samples`` (1-D mono or (n, channels)) as a 16-bit PCM WAV.

    Non-finite values are zeroed and the signal is hard-limited to [-1, 1] so a
    synthesis bug can never produce a corrupt or blasting output file.
    """
    out = Path(path)
    try:
        y = np.asarray(samples, dtype=np.float32)
        if y.size == 0:
            raise AudioWriteError("Refusing to write an empty audio buffer.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), y, int(sr), subtype="PCM_16")
    except AudioWriteError:
        raise
    except Exception as exc:
        raise AudioWriteError(f"Could not write '{out.name}': {exc}") from exc
    return out
