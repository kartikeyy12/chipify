"""
Figures for Chipify: waveform and spectrogram comparisons plus the analysis piano roll.

Every function returns a ``matplotlib.figure.Figure`` that Gradio's ``gr.Plot`` can
show directly, and ``save_figure`` writes it to PNG.

Figures are built with the object-oriented ``Figure`` class instead of ``pyplot`` on
purpose: ``pyplot`` keeps every figure alive in global state, which would leak memory
in a long-running web server. Nothing here needs a display (Agg only).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import signal as sps

from .chip_tables import hz_to_midi

if TYPE_CHECKING:  # annotations only
    from .pitch_tracker import TrackAnalysis

PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHANNEL_COLORS = {"pulse1": "#d62728", "pulse2": "#ff9f1c", "triangle": "#1f77b4"}
DRUM_COLORS = {"kick": "#2ca02c", "snare": "#9467bd", "hat": "#7f7f7f"}
ORIGINAL_COLOR = "#4c78a8"
CHIP_COLOR = "#e4572e"


def note_name(midi_note: int) -> str:
    """60 -> 'C4'."""
    return f"{PITCH_CLASS_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


# --------------------------------------------------------------------------- #
# Numerics (pure, tested separately from the drawing)
# --------------------------------------------------------------------------- #
def downsample_envelope(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Min/max per bin, so a 3-minute waveform can be drawn with a few thousand
    points without losing peaks (plain decimation would hide transients).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("cannot plot an empty signal")
    n_bins = int(max(1, min(n_bins, x.size)))
    edges = np.linspace(0, x.size, n_bins + 1).astype(int)[:-1]
    return np.minimum.reduceat(x, edges), np.maximum.reduceat(x, edges)


def compute_spectrogram_db(
    x: np.ndarray, sr: int, fmax_hz: float = 8000.0, nperseg: int = 1024, floor_db: float = 80.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Power spectrogram in dB relative to its own peak, clipped to ``floor_db`` below it.

    Returns (db[freq, time], times_s, freqs_hz) limited to ``fmax_hz``.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size < nperseg:
        raise ValueError(f"signal is too short for a spectrogram (needs >= {nperseg} samples)")
    freqs, times, power = sps.spectrogram(
        x, fs=sr, window="hann", nperseg=nperseg, noverlap=nperseg * 3 // 4, scaling="spectrum", mode="psd"
    )
    keep = freqs <= min(fmax_hz, sr / 2.0)
    db = 10.0 * np.log10(power[keep] + 1e-12)
    db = np.maximum(db - db.max(), -floor_db)
    return db, times, freqs[keep]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_waveform_comparison(
    original: np.ndarray, original_sr: int, chip: np.ndarray, chip_sr: int, max_bins: int = 3000
) -> Figure:
    """Original and chiptune waveforms stacked on one shared time axis."""
    fig = Figure(figsize=(11, 4.6), dpi=100)
    top, bottom = fig.subplots(2, 1, sharex=True)
    for ax, sig, sr, color, title in (
        (top, original, original_sr, ORIGINAL_COLOR, "Original"),
        (bottom, chip, chip_sr, CHIP_COLOR, "Chiptune"),
    ):
        lo, hi = downsample_envelope(sig, max_bins)
        duration = len(sig) / float(sr)
        t = np.linspace(0.0, duration, lo.size)
        ax.vlines(t, lo, hi, colors=color, linewidth=0.8)       # one min/max bar per bin: crisp, no hairline blending
        ax.set_xlim(0.0, duration)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    bottom.set_xlabel("time (s)")
    fig.suptitle("Waveform")
    fig.tight_layout()
    return fig


def plot_spectrogram_comparison(
    original: np.ndarray, original_sr: int, chip: np.ndarray, chip_sr: int, fmax_hz: float = 8000.0
) -> Figure:
    """Original and chiptune spectrograms (dB, 0-8 kHz by default) one above the other."""
    fig = Figure(figsize=(11, 6.2), dpi=100)
    axes = fig.subplots(2, 1, sharex=True)
    for ax, sig, sr, title in ((axes[0], original, original_sr, "Original"), (axes[1], chip, chip_sr, "Chiptune")):
        db, times, freqs = compute_spectrogram_db(sig, sr, fmax_hz)
        image = ax.imshow(
            db, origin="lower", aspect="auto", cmap="magma", vmin=-80.0, vmax=0.0,
            extent=[times[0], times[-1], freqs[0] / 1000.0, freqs[-1] / 1000.0],
        )
        ax.set_ylabel(f"{title}\nkHz")
    axes[1].set_xlabel("time (s)")
    fig.colorbar(image, ax=list(axes), label="dB below peak", pad=0.01)
    fig.suptitle("Spectrogram")
    return fig


def plot_piano_roll(analysis: "TrackAnalysis", start_s: float = 0.0, window_s: float = 20.0) -> Figure:
    """
    Quantised chip notes (bars) over the tracker's raw pitch (black dots), the
    rhythm grid (faint lines) and drum hits (ticks along the bottom).
    """
    end_s = min(analysis.duration_s, start_s + window_s)
    notes = [*analysis.lead, *analysis.harmony, *analysis.bass]
    in_view = [n for n in notes if n.end_s >= start_s and n.start_s <= end_s]
    lo = min((n.midi_note for n in in_view), default=36) - 6
    hi = max((n.midi_note for n in in_view), default=84) + 6      # headroom so the legend never covers a note

    fig = Figure(figsize=(14, 6), dpi=100)
    ax = fig.subplots()

    if (end_s - start_s) <= 40 and analysis.grid_step_s > 0:
        k0 = int(np.floor((start_s - analysis.grid_offset_s) / analysis.grid_step_s))
        k1 = int(np.ceil((end_s - analysis.grid_offset_s) / analysis.grid_step_s))
        ax.vlines([analysis.grid_offset_s + k * analysis.grid_step_s for k in range(k0, k1 + 1)],
                  lo, hi, colors="#bbbbbb", linewidth=0.4, zorder=0)

    for n in in_view:
        ax.barh(n.midi_note, n.duration_s, left=n.start_s, height=0.8,
                color=CHANNEL_COLORS.get(n.channel, "#444444"), alpha=0.35 + 0.65 * n.velocity, zorder=2)

    for hz in (analysis.f0_lead_hz, analysis.f0_bass_hz):
        if len(hz) and len(hz) == len(analysis.f0_times):
            ax.plot(analysis.f0_times, hz_to_midi(hz), ".", color="black", markersize=1.6, alpha=0.4, zorder=3)

    for d in analysis.drums:
        if start_s <= d.time_s <= end_s:
            ax.vlines(d.time_s, lo, lo + 2.5, colors=DRUM_COLORS.get(d.kind, "#444444"), linewidth=1.6, zorder=2)

    first_c = int(lo) - int(lo) % 12 + 12
    ticks = list(range(first_c, int(hi) + 1, 12))
    ax.set_yticks(ticks, [note_name(t) for t in ticks])
    ax.set_xlim(start_s, max(end_s, start_s + 1e-3))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"{analysis.bpm:.1f} BPM ({analysis.bpm_source}) | key {analysis.key or '?'} | grid fit "
        f"{analysis.grid_fit:.2f}  -  bars: quantised chip notes, dots: raw pitch, bottom ticks: drums"
    )
    handles = [Line2D([0], [0], color=c, lw=6, label=k) for k, c in CHANNEL_COLORS.items()]
    handles += [Line2D([0], [0], color=c, lw=2, label=k) for k, c in DRUM_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", ncol=6, fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 110) -> Path:
    """Write a figure to PNG (creating parent folders) and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    return out
