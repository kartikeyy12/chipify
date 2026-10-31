#!/usr/bin/env python3
"""
Chipify environment smoke test.

Verifies, in order:
  1. Python version and virtualenv activation
  2. Required Python packages import cleanly
  3. System libraries (ffmpeg, libsndfile, supported codecs)
  4. A real DSP round-trip: WAV write/read -> HPSS -> pYIN pitch tracking
  5. Port 7860 is free for Gradio

Exit code is 0 when everything passes, 1 otherwise.
"""
from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

# Headless plotting: must be set before matplotlib is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

GRADIO_PORT = 7860

REQUIRED_PACKAGES = [
    "numpy", "scipy", "librosa", "soundfile",
    "numba", "matplotlib", "mido", "gradio",
]

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def check_python() -> None:
    section("Python")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} detected; 3.10+ is required")

    if sys.prefix != sys.base_prefix:
        ok(f"Virtualenv active: {sys.prefix}")
    else:
        warn("No virtualenv active. Run: source .venv/bin/activate")


def check_packages() -> bool:
    section("Python packages")
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            ok(f"{name:<11} {getattr(mod, '__version__', 'unknown version')}")
        except Exception as exc:  # ImportError, or binary incompatibility errors
            fail(f"{name}: import failed ({type(exc).__name__}: {exc})")
            all_ok = False
    return all_ok


def check_system_libs() -> None:
    section("System libraries")

    # ffmpeg: used as the fallback decoder for mp3 and other compressed formats
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            out = subprocess.run(
                [ffmpeg, "-version"], capture_output=True, text=True, timeout=10
            ).stdout.splitlines()[0]
            ok(out)
        except Exception as exc:
            warn(f"ffmpeg found but could not be executed: {exc}")
    else:
        fail("ffmpeg not found on PATH (sudo apt install ffmpeg)")

    # libsndfile: the backbone of soundfile
    try:
        import soundfile as sf

        ok(f"libsndfile  {sf.__libsndfile_version__}")
        formats = sf.available_formats()
        ok("WAV supported" if "WAV" in formats else "WAV NOT supported")
        if "MP3" in formats:
            ok("MP3 supported natively by libsndfile")
        else:
            warn("MP3 not native in libsndfile; ffmpeg fallback will be used")
    except Exception as exc:
        fail(f"soundfile/libsndfile problem: {exc}")


def check_dsp_roundtrip() -> None:
    section("DSP round-trip (WAV I/O -> HPSS -> pYIN)")
    try:
        import numpy as np
        import soundfile as sf
        import librosa

        sr = 22050
        t = np.arange(sr * 2) / sr
        tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        # 1. WAV write -> read
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "probe.wav")
            sf.write(path, tone, sr, subtype="PCM_16")
            y, sr_in = sf.read(path, dtype="float32")
        if sr_in != sr or len(y) != len(tone):
            fail("WAV round-trip changed sample rate or length")
            return
        ok("WAV write/read round-trip")

        # 2. HPSS (first call may be slow while numba compiles)
        t0 = time.time()
        harmonic, percussive = librosa.effects.hpss(y)
        ok(f"HPSS completed in {time.time() - t0:.2f}s")

        # 3. pYIN pitch tracking on the harmonic component
        t0 = time.time()
        f0, voiced_flag, voiced_prob = librosa.pyin(
            harmonic,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        elapsed = time.time() - t0
        if not np.any(np.isfinite(f0)):
            fail("pYIN returned no voiced frames for a pure 440 Hz tone")
            return
        median_f0 = float(np.nanmedian(f0))
        if abs(median_f0 - 440.0) < 5.0:
            ok(f"pYIN median F0 = {median_f0:.2f} Hz (expected 440) in {elapsed:.2f}s")
        else:
            fail(f"pYIN median F0 = {median_f0:.2f} Hz, expected ~440 Hz")
    except Exception as exc:
        fail(f"DSP round-trip crashed: {type(exc).__name__}: {exc}")


def check_port() -> None:
    section(f"Network (port {GRADIO_PORT})")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", GRADIO_PORT))
        ok(f"0.0.0.0:{GRADIO_PORT} is free")
    except OSError as exc:
        fail(f"Cannot bind 0.0.0.0:{GRADIO_PORT}: {exc}")
    finally:
        sock.close()


def main() -> int:
    print("Chipify environment check")
    check_python()
    packages_ok = check_packages()
    check_system_libs()
    if packages_ok:
        check_dsp_roundtrip()
    else:
        warn("Skipping DSP round-trip because package imports failed")
    check_port()

    print("\n" + "=" * 40)
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("RESULT: all checks passed. Ready for Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
