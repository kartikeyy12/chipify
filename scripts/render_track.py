#!/usr/bin/env python3
"""
Convert a track to chiptune: load -> analyse -> synthesise -> write a WAV.

This is the first end-to-end Chipify run (the web UI in Phase 6 wraps the same
calls). It also saves the analysis piano roll so you can compare what was heard
with what was detected.

Examples
--------
    python scripts/render_track.py assets/samples/axel_f_45s.wav
    python scripts/render_track.py assets/samples/axel_f_45s.wav --harmony third_below --pwm 0.4
    python scripts/render_track.py assets/samples/axel_f_45s.wav --bitcrush 0.3 --lead-duty 0.125
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chipify import config  # noqa: E402
from chipify.audio_decoder import AudioError, load_audio  # noqa: E402
from chipify.pitch_tracker import HARMONY_MODES, AnalysisSettings, analyze  # noqa: E402
from chipify.synth_engine import SynthSettings, render_to_wav  # noqa: E402

DUTY_CHOICES = (0.125, 0.25, 0.5)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")

    a = p.add_argument_group("analysis")
    a.add_argument("--chip", default="nes", choices=["nes", "gameboy", "ideal"])
    a.add_argument("--bpm", type=float, default=None, help="force a tempo instead of estimating it")
    a.add_argument("--strictness", type=float, default=1.0, help="0 = loose, 1 = fully quantised")
    a.add_argument("--steps", type=int, default=4, help="grid steps per beat (4 = 1/16 notes)")
    a.add_argument("--harmony", default="off", choices=HARMONY_MODES)
    a.add_argument("--no-drums", action="store_true")
    a.add_argument("--drum-delta", type=float, default=None, help="higher = fewer drum hits")
    a.add_argument("--drum-flatness", type=float, default=None, help="higher = fewer snare/hat hits")
    a.add_argument("--kick-strength", type=float, default=None, help="higher = fewer kicks")

    s = p.add_argument_group("sound")
    s.add_argument("--lead-duty", type=float, default=0.25, choices=DUTY_CHOICES)
    s.add_argument("--harmony-duty", type=float, default=0.125, choices=DUTY_CHOICES)
    s.add_argument("--pwm", type=float, default=0.0, help="pulse-width modulation depth 0-1")
    s.add_argument("--pwm-rate", type=float, default=5.0, help="PWM speed in Hz")
    s.add_argument("--bitcrush", type=float, default=0.0, help="bit-crush intensity 0-1")
    s.add_argument("--drum-gain", type=float, default=1.0, help="drum volume 0-2")
    s.add_argument("--bass-dynamics", action="store_true", help="let velocity change bass volume (not NES-accurate)")
    s.add_argument("--free-duty", action="store_true", help="allow any duty instead of snapping to 12.5/25/50 %%")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])

    analysis_settings = AnalysisSettings(
        chip=args.chip, bpm=args.bpm, quant_strictness=args.strictness, steps_per_beat=args.steps,
        harmony_mode=args.harmony, drums_enabled=not args.no_drums,
    )
    for flag, attr in (("drum_delta", "drum_delta"), ("drum_flatness", "drum_min_flatness"),
                       ("kick_strength", "kick_min_strength")):
        if getattr(args, flag) is not None:
            setattr(analysis_settings, attr, getattr(args, flag))

    synth_settings = SynthSettings.for_chip(
        args.chip, lead_duty=args.lead_duty, harmony_duty=args.harmony_duty, pwm_depth=args.pwm,
        pwm_rate_hz=args.pwm_rate, bitcrush=args.bitcrush, drum_gain=args.drum_gain,
        triangle_dynamics=args.bass_dynamics, strict_duty=not args.free_duty,
    )

    t0 = time.time()
    try:
        analysis_settings.validate()
        synth_settings.validate()
        audio = load_audio(args.path)
        print(audio.summary(), "\n")
        result = analyze(audio, analysis_settings, progress=lambda f, m: print(f"  [{f * 100:3.0f}%] {m}"))
        print("\n" + result.summary())

        print("\n  synthesising ...")
        stem = Path(args.path).stem
        out_path = config.new_output_path(f"{stem}_chip", ".wav")
        rendered = render_to_wav(result, out_path, synth_settings)
    except AudioError as exc:
        print(f"\nREJECTED: {exc}")
        return 1
    except ValueError as exc:
        print(f"\nBAD SETTINGS: {exc}")
        return 2

    counts = rendered.note_counts
    print(f"\n=== Done in {time.time() - t0:.1f} s ===")
    print(f"voices  : pulse1 {counts['pulse1']}, pulse2 {counts['pulse2']}, "
          f"triangle {counts['triangle']}, noise {counts['noise']}")
    print(f"output  : {out_path}  ({rendered.duration_s:.1f} s @ {rendered.sample_rate} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
