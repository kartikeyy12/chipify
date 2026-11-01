#!/usr/bin/env python3
"""
Phase 3 diagnostic: run the full musical analysis on one track and inspect it.

Outputs
-------
* a console report (tempo, key, note/drum counts, first notes of each channel)
* outputs/<name>_notes.png          piano roll: raw pitch dots vs quantised notes
* outputs/<name>_phase3_preview.wav a crude audition render (naive, aliased waves)

The preview is deliberately throwaway: it only lets you HEAR whether the right
notes were found. The real band-limited synth arrives in Phase 4.

Usage
-----
    python scripts/analyze_track.py assets/samples/axel_f_45s.wav
    python scripts/analyze_track.py assets/samples/axel_f_45s.wav --bpm 115 --harmony third_below
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from chipify import config  # noqa: E402
from chipify.audio_decoder import AudioError, load_audio, write_wav  # noqa: E402
from chipify.chip_tables import hz_to_midi  # noqa: E402
from chipify.pitch_tracker import (  # noqa: E402
    HARMONY_MODES, AnalysisSettings, NoteEvent, TrackAnalysis, analyze,
)
from chipify.visualizer import note_name, plot_piano_roll, save_figure  # noqa: E402

PREVIEW_SR = 44_100


# --------------------------------------------------------------------------- #
# Crude audition render (naive waveforms; replaced by synth_engine in Phase 4)
# --------------------------------------------------------------------------- #
def render_preview(a: TrackAnalysis, sr: int = PREVIEW_SR) -> np.ndarray:
    total = int((a.duration_s + 0.5) * sr)
    out = np.zeros(total, dtype=np.float64)
    rng = np.random.default_rng(0)

    def place(start_s: float, wave: np.ndarray) -> None:
        i = int(start_s * sr)
        if 0 <= i < total:
            j = min(total, i + wave.size)
            out[i:j] += wave[: j - i]

    voices = ((a.lead, "pulse50", 0.18), (a.harmony, "pulse25", 0.12), (a.bass, "tri", 0.30))
    for notes, kind, gain in voices:
        for n in notes:
            m = int(round(n.duration_s * sr))
            if m < 16:
                continue
            idx = np.arange(m)
            phase = (idx / sr * n.freq_hz) % 1.0
            if kind == "pulse50":
                wave = np.where(phase < 0.5, 1.0, -1.0)
            elif kind == "pulse25":
                wave = np.where(phase < 0.25, 1.0, -1.0)
            else:
                wave = 4.0 * np.abs(phase - 0.5) - 1.0
            fade = np.minimum(1.0, np.minimum(idx, idx[::-1]) / (0.003 * sr))   # 3 ms anti-click
            place(n.start_s, wave * fade * gain * (0.3 + 0.7 * n.velocity))

    for d in a.drums:
        length = {"kick": 0.12, "snare": 0.14, "hat": 0.04}[d.kind]
        m = int(length * sr)
        t = np.arange(m) / sr
        if d.kind == "kick":
            wave = np.sin(2 * np.pi * (50 * t + 100 * 0.03 * (1 - np.exp(-t / 0.03)))) * np.exp(-t / 0.05)
        else:
            tau = 0.04 if d.kind == "snare" else 0.012
            wave = rng.uniform(-1, 1, m) * np.exp(-t / tau)
        place(d.time_s, wave * 0.25 * (0.4 + 0.6 * d.strength))

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 0.9 / peak
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_notes(label: str, notes: list[NoteEvent], limit: int = 12) -> None:
    print(f"\n{label} (first {min(limit, len(notes))} of {len(notes)})")
    print("   start    len   note   chip Hz   reg   detune   vel")
    for n in notes[:limit]:
        print(f"  {n.start_s:6.2f} {n.duration_s:6.2f}   {note_name(n.midi_note):<4} "
              f"{n.freq_hz:8.2f} {n.register:5d}  {n.cents_error:+5.1f}c  {n.velocity:4.2f}")


def describe_f0(label: str, f0_hz: np.ndarray) -> None:
    """Diagnostic: how much of the track the tracker called voiced, and at what pitches."""
    voiced = np.isfinite(f0_hz)
    if not voiced.any():
        print(f"{label:<5}: no voiced frames")
        return
    lo, mid, hi = np.percentile(f0_hz[voiced], [10, 50, 90])
    names = [note_name(int(round(float(hz_to_midi(v))))) for v in (lo, mid, hi)]
    print(f"{label:<5}: voiced {100 * voiced.mean():4.1f}% of frames | pitch 10/50/90th pct = "
          f"{lo:.0f}/{mid:.0f}/{hi:.0f} Hz ({names[0]}/{names[1]}/{names[2]})")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--chip", default="nes", choices=["nes", "gameboy", "ideal"])
    p.add_argument("--bpm", type=float, default=None, help="force a tempo instead of estimating it")
    p.add_argument("--strictness", type=float, default=1.0, help="0 = loose, 1 = fully quantised")
    p.add_argument("--steps", type=int, default=4, help="grid steps per beat (4 = 1/16 notes)")
    p.add_argument("--harmony", default="off", choices=HARMONY_MODES)
    p.add_argument("--no-drums", action="store_true")
    p.add_argument("--drum-delta", type=float, default=None, help="onset threshold; higher = fewer hits")
    p.add_argument("--drum-flatness", type=float, default=None, help="min noisiness for snare/hat; higher = fewer hits")
    p.add_argument("--kick-strength", type=float, default=None, help="min strength (0-1) for kicks")
    p.add_argument("--split-hz", type=float, default=None, help="bass/lead register boundary (default 175)")
    p.add_argument("--bass-source", choices=["harmonic", "mix"], default=None,
                   help="'mix' tracks bass on the unseparated audio (try if bass notes go missing)")
    p.add_argument("--hpss-kernel", type=int, default=None, help="HPSS median-filter size (default 31; smaller keeps shorter notes)")
    p.add_argument("--lead-gate-db", type=float, default=None,
                   help="ignore lead frames this far below the loudest (default 25; lower rejects more leakage)")
    p.add_argument("--octave-fix", type=float, default=None, help="floor octave-error correction ratio (0 = off)")
    p.add_argument("--lead-mask", type=int, default=None, help="lead partials removed from the bass band (0 = off)")
    p.add_argument("--fundamental-ratio", type=float, default=None, help="ghost-pitch check strength (0 = off)")
    p.add_argument("--plot-start", type=float, default=0.0)
    p.add_argument("--plot-seconds", type=float, default=20.0)
    args = p.parse_args(argv[1:])

    settings = AnalysisSettings(
        chip=args.chip, bpm=args.bpm, quant_strictness=args.strictness,
        steps_per_beat=args.steps, harmony_mode=args.harmony, drums_enabled=not args.no_drums,
    )
    for flag, attr in (("drum_delta", "drum_delta"), ("drum_flatness", "drum_min_flatness"),
                       ("kick_strength", "kick_min_strength"), ("split_hz", "split_hz"),
                       ("bass_source", "bass_source"), ("hpss_kernel", "hpss_kernel"),
                       ("fundamental_ratio", "fundamental_min_ratio"),
                       ("lead_mask", "lead_mask_harmonics"), ("lead_gate_db", "lead_gate_db"),
                       ("octave_fix", "octave_fix_ratio")):
        if getattr(args, flag) is not None:
            setattr(settings, attr, getattr(args, flag))

    t0 = time.time()
    try:
        audio = load_audio(args.path)
        print(audio.summary())
        print()
        result = analyze(audio, settings, progress=lambda f, m: print(f"  [{f * 100:3.0f}%] {m}"))
        settings.validate()
    except AudioError as exc:
        print(f"\nREJECTED: {exc}")
        return 1
    except ValueError as exc:
        print(f"\nBAD SETTINGS: {exc}")
        return 2

    print(f"\n=== Analysis finished in {time.time() - t0:.1f} s ===")
    print(result.summary())
    print("\nRaw pitch-tracker output (before note segmentation):")
    describe_f0("lead", result.f0_lead_hz)
    describe_f0("bass", result.f0_bass_hz)
    print_notes("LEAD (pulse 1)", result.lead)
    print_notes("BASS (triangle)", result.bass)
    if result.harmony:
        print_notes("HARMONY (pulse 2)", result.harmony, limit=6)

    stem = Path(args.path).stem
    png = config.new_output_path(f"{stem}_notes", ".png")
    wav = config.new_output_path(f"{stem}_phase3_preview", ".wav")
    save_figure(plot_piano_roll(result, args.plot_start, args.plot_seconds), png)
    write_wav(wav, render_preview(result), PREVIEW_SR)
    print(f"\npiano roll : {png}")
    print(f"preview wav: {wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
