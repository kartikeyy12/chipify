#!/usr/bin/env python3
"""
Convert a track to chiptune: load -> analyse -> synthesise -> export.

This is the full Chipify pipeline from the command line (the web UI in Phase 6
wraps the same calls). Pick a preset, then override anything you like.

Outputs (all in outputs/)
-------------------------
  <name>_chip_<id>.wav        the chiptune render
  <name>_chip_<id>.mid        the note data as a Standard MIDI File (with --midi)
  <name>_waveform_<id>.png    original vs chiptune waveform        (with --plots)
  <name>_spectrogram_<id>.png original vs chiptune spectrogram     (with --plots)
  <name>_notes_<id>.png       piano roll of what was detected      (with --plots)

Examples
--------
    python scripts/render_track.py assets/samples/axel_f_60s.wav --midi --plots
    python scripts/render_track.py assets/samples/axel_f_60s.wav --preset "Arcade Synth"
    python scripts/render_track.py assets/samples/axel_f_60s.wav --preset "Game Boy Mono" --bitcrush 0.3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chipify import config  # noqa: E402
from chipify.audio_decoder import AudioError, load_audio  # noqa: E402
from chipify.midi_export import export_midi  # noqa: E402
from chipify.pitch_tracker import HARMONY_MODES, analyze  # noqa: E402
from chipify.presets import DEFAULT_PRESET, PRESET_NAMES, UserKnobs, build_settings  # noqa: E402
from chipify.synth_engine import render_to_wav  # noqa: E402
from chipify.visualizer import (  # noqa: E402
    plot_piano_roll, plot_spectrogram_comparison, plot_waveform_comparison, save_figure,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--preset", default=DEFAULT_PRESET, choices=PRESET_NAMES)
    p.add_argument("--midi", action="store_true", help="also export a .mid file")
    p.add_argument("--plots", action="store_true", help="also save waveform, spectrogram and piano-roll PNGs")

    k = p.add_argument_group("knobs (anything left out keeps the preset's value)")
    k.add_argument("--bpm", type=float, default=None, help="force a tempo instead of detecting it")
    k.add_argument("--strictness", type=float, default=None, help="quantisation strictness 0-1")
    k.add_argument("--harmony", default=None, choices=HARMONY_MODES)
    k.add_argument("--no-drums", action="store_true")
    k.add_argument("--steps", type=int, default=None, help="grid steps per beat (4 = 1/16 notes)")
    k.add_argument("--pwm", type=float, default=None, help="pulse-width modulation depth 0-1")
    k.add_argument("--bitcrush", type=float, default=None, help="bit-crush intensity 0-1")
    k.add_argument("--lead-duty", type=float, default=None, choices=(0.125, 0.25, 0.5))
    k.add_argument("--harmony-duty", type=float, default=None, choices=(0.125, 0.25, 0.5))
    k.add_argument("--drum-gain", type=float, default=None, help="drum volume 0-2")
    k.add_argument("--bass-dynamics", action="store_true", help="let velocity change bass volume (not NES-accurate)")
    k.add_argument("--free-duty", action="store_true", help="allow any duty instead of snapping to 12.5/25/50 %%")

    t = p.add_argument_group("drum detection tuning")
    t.add_argument("--drum-delta", type=float, default=None, help="higher = fewer drum hits")
    t.add_argument("--drum-flatness", type=float, default=None, help="higher = fewer snare/hat hits")
    t.add_argument("--kick-strength", type=float, default=None, help="higher = fewer kicks")
    return p


def knobs_from_args(args: argparse.Namespace) -> UserKnobs:
    return UserKnobs(
        strictness=args.strictness, bpm=args.bpm, harmony=args.harmony,
        drums=False if args.no_drums else None, steps_per_beat=args.steps, pwm_depth=args.pwm,
        bitcrush=args.bitcrush, lead_duty=args.lead_duty, harmony_duty=args.harmony_duty,
        drum_gain=args.drum_gain, bass_dynamics=True if args.bass_dynamics else None,
        strict_duty=False if args.free_duty else None,
    )


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv[1:])
    t0 = time.time()
    try:
        analysis_settings, synth_settings = build_settings(args.preset, knobs_from_args(args))
        for flag, attr in (("drum_delta", "drum_delta"), ("drum_flatness", "drum_min_flatness"),
                           ("kick_strength", "kick_min_strength")):
            if getattr(args, flag) is not None:
                setattr(analysis_settings, attr, getattr(args, flag))
        analysis_settings.validate()

        audio = load_audio(args.path)
        print(audio.summary(), "\n")
        print(f"preset: {args.preset}\n")
        result = analyze(audio, analysis_settings, progress=lambda f, m: print(f"  [{f * 100:3.0f}%] {m}"))
        print("\n" + result.summary())

        print("\n  synthesising ...")
        stem = Path(args.path).stem
        wav_path = config.new_output_path(f"{stem}_chip", ".wav")
        rendered = render_to_wav(result, wav_path, synth_settings)

        mid_path = export_midi(result, wav_path.with_suffix(".mid")) if args.midi else None

        plot_paths: list[Path] = []
        if args.plots:
            print("  drawing plots ...")
            wave = plot_waveform_comparison(audio.samples, audio.sr, rendered.audio, rendered.sample_rate)
            spec = plot_spectrogram_comparison(audio.samples, audio.sr, rendered.audio, rendered.sample_rate)
            roll = plot_piano_roll(result)
            for fig, name in ((wave, "waveform"), (spec, "spectrogram"), (roll, "notes")):
                plot_paths.append(save_figure(fig, config.new_output_path(f"{stem}_{name}", ".png")))
    except AudioError as exc:
        print(f"\nREJECTED: {exc}")
        return 1
    except ValueError as exc:
        print(f"\nBAD SETTINGS: {exc}")
        return 2

    c = rendered.note_counts
    print(f"\n=== Done in {time.time() - t0:.1f} s ===")
    print(f"voices : pulse1 {c['pulse1']}, pulse2 {c['pulse2']}, triangle {c['triangle']}, noise {c['noise']}")
    print(f"audio  : {wav_path}  ({rendered.duration_s:.1f} s @ {rendered.sample_rate} Hz)")
    if mid_path:
        print(f"midi   : {mid_path}")
    for path in plot_paths:
        print(f"plot   : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
