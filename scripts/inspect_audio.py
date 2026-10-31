#!/usr/bin/env python3
"""Usage: python scripts/inspect_audio.py track1.mp3 [track2.wav ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chipify.audio_decoder import AudioError, load_audio  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    status = 0
    for arg in argv[1:]:
        print(f"\n--- {arg}")
        try:
            print(load_audio(arg).summary())
        except AudioError as exc:
            print(f"REJECTED: {exc}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
