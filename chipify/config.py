"""
Central configuration for Chipify.

Every tunable limit and constant lives here so the DSP modules stay free of
magic numbers. Values can be overridden for deployment through environment
variables where it makes sense (currently only the output directory).
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = Path(os.environ.get("CHIPIFY_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# --------------------------------------------------------------------------- #
# Sample rates
# --------------------------------------------------------------------------- #
# Analysis rate: 22.05 kHz keeps energy up to ~11 kHz, far above the highest
# note the NES can play (~12.5 kHz timer limit is irrelevant; musical fundamentals
# top out near 4 kHz), and halves the cost of HPSS/pYIN versus 44.1 kHz.
ANALYSIS_SR: int = 22_050

# Synthesis/export rate used by synth_engine (Phase 4). 44.1 kHz gives the
# band-limited pulses room to breathe and is universally playable.
OUTPUT_SR: int = 44_100

# Reject nonsensical headers (corrupt files often report absurd sample rates).
MIN_VALID_SR: int = 4_000
MAX_VALID_SR: int = 768_000
LOW_SR_WARN_BELOW: int = 16_000

# --------------------------------------------------------------------------- #
# Input limits
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".flac", ".ogg"})
MAX_FILE_BYTES: int = 100 * 1024 * 1024      # 100 MiB
MAX_DURATION_S: float = 180.0                # pYIN is roughly real-time; cap the wait
MIN_DURATION_S: float = 0.5

# --------------------------------------------------------------------------- #
# Level / quality gates (all in dBFS, full scale = 1.0)
# --------------------------------------------------------------------------- #
SILENCE_PEAK_DBFS: float = -60.0     # peak below this => treated as silence
SILENCE_RMS_DBFS: float = -70.0      # RMS below this  => treated as silence
NORMALIZE_PEAK_DBFS: float = -1.0    # target peak after normalization
CLIPPING_THRESHOLD: float = 0.999    # |x| at/above this counts as a clipped sample
CLIPPING_WARN_FRACTION: float = 0.01
SPARSE_ACTIVITY_DB: float = 40.0     # "active" frame = within this many dB of the loudest
SPARSE_ACTIVITY_WARN_FRACTION: float = 0.05
PHASE_CANCEL_RATIO_WARN: float = 0.5  # mono RMS / channel RMS below this => warn

# --------------------------------------------------------------------------- #
# Subprocess limits
# --------------------------------------------------------------------------- #
FFMPEG_TIMEOUT_S: int = 120
FFPROBE_TIMEOUT_S: int = 15

# --------------------------------------------------------------------------- #
# Output management
# --------------------------------------------------------------------------- #
OUTPUT_MAX_AGE_S: int = 3600         # generated files older than this get purged


def ensure_output_dir() -> Path:
    """Create the output directory if needed and return it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def new_output_path(stem: str, suffix: str) -> Path:
    """
    Build a unique, filesystem-safe output path such as
    ``outputs/my_song_chip_3f2a9c1b.wav``.

    The stem is sanitised because it originates from user-supplied filenames.
    """
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "track"
    safe_stem = safe_stem[:60]
    if not suffix.startswith("."):
        suffix = "." + suffix
    return ensure_output_dir() / f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"


def cleanup_old_outputs(max_age_s: float = OUTPUT_MAX_AGE_S) -> int:
    """
    Delete generated files older than ``max_age_s`` seconds (``.gitkeep`` is
    always preserved). Returns the number of files removed. Never raises:
    cleanup problems must not break a user's conversion request.
    """
    removed = 0
    if not OUTPUT_DIR.is_dir():
        return removed
    cutoff = time.time() - max_age_s
    for item in OUTPUT_DIR.iterdir():
        try:
            if item.is_file() and item.name != ".gitkeep" and item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1
        except OSError:
            continue
    return removed
