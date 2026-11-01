"""
MIDI export for Chipify: note events in, a Standard MIDI File (.mid) out.

The file is Type 1 (parallel tracks) and lines up with the source audio:
tick 0 is audio time 0, and one constant tempo (the analysis BPM) maps seconds
to ticks, so loading the .mid beside the original in a DAW keeps notes on the beat.

Track layout
------------
  track 0   tempo, time signature, key signature
  track 1   lead      (pulse 1)  channel 1,  GM program 81 "Lead 1 (square)"
  track 2   harmony   (pulse 2)  channel 2,  GM program 81
  track 3   bass      (triangle) channel 3,  GM program 39 "Synth Bass 1"
  track 4   drums     (noise)    channel 10, General MIDI kit: kick 36, snare 38, closed hat 42

Tracks with no notes are omitted. The writer is self-contained (no third-party
MIDI library) and follows the SMF spec directly, so it is easy to audit; the test
suite also reads the files back with ``mido`` when it is installed.
"""
from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .audio_decoder import AudioError

if TYPE_CHECKING:  # annotations only
    from .pitch_tracker import DrumEvent, NoteEvent, TrackAnalysis

log = logging.getLogger(__name__)

DEFAULT_PPQ: int = 480                      # ticks per quarter note
MIN_PPQ, MAX_PPQ = 24, 960
MIN_BPM, MAX_BPM = 20.0, 400.0

# 0-based MIDI channels / GM program numbers (program change data is 0-based: 80 == "Lead 1 (square)")
CHANNELS = {"pulse1": 0, "pulse2": 1, "triangle": 2}
PROGRAMS = {"pulse1": 80, "pulse2": 80, "triangle": 38}
TRACK_NAMES = {"pulse1": "Lead (pulse 1)", "pulse2": "Harmony (pulse 2)", "triangle": "Bass (triangle)"}
DRUM_CHANNEL: int = 9
GM_DRUM_NOTES = {"kick": 36, "snare": 38, "hat": 42}
DRUM_NOTE_SECONDS: float = 0.06             # drum "note length"; GM drum voices ignore it, DAWs show it

_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Event tuple: (absolute_tick, priority, raw_bytes). Priority 0 sorts first at equal ticks
# (note-offs before note-ons so repeated notes retrigger cleanly).
_Event = tuple[int, int, bytes]


class MidiExportError(AudioError):
    """The MIDI file could not be produced."""


# --------------------------------------------------------------------------- #
# Low-level SMF encoding
# --------------------------------------------------------------------------- #
def vlq(value: int) -> bytes:
    """Variable-length quantity: 7 bits per byte, high bit set on all but the last."""
    if value < 0:
        raise ValueError("VLQ values must be non-negative")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))


def _meta(kind: int, data: bytes) -> bytes:
    return b"\xFF" + bytes([kind]) + vlq(len(data)) + data


def _track_chunk(events: Sequence[_Event]) -> bytes:
    """Delta-encode events and wrap them in an ``MTrk`` chunk ending with End-of-Track."""
    body = bytearray()
    last = 0
    for tick, _prio, raw in sorted(events, key=lambda e: (e[0], e[1])):
        body += vlq(tick - last) + raw
        last = tick
    body += vlq(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def _key_signature(key: str | None) -> bytes | None:
    """Meta event for a key string like 'F minor' / 'C major', or None if unparseable."""
    if not key:
        return None
    parts = key.split()
    if len(parts) != 2 or parts[0] not in _PITCH_CLASSES or parts[1] not in ("major", "minor"):
        return None
    tonic = _PITCH_CLASSES.index(parts[0])
    minor = parts[1] == "minor"
    major_tonic = (tonic + 3) % 12 if minor else tonic           # relative major shares the signature
    fifths = (major_tonic * 7) % 12                              # position on the circle of fifths
    if fifths > 6:
        fifths -= 12                                             # prefer fewer accidentals (flats side)
    return _meta(0x59, struct.pack(">bB", fifths, 1 if minor else 0))


# --------------------------------------------------------------------------- #
# Event building
# --------------------------------------------------------------------------- #
def _velocity(v: float) -> int:
    return int(min(127, max(1, round(1 + 126 * min(1.0, max(0.0, v))))))


def _channel_events(notes: Sequence["NoteEvent"], channel: str, to_ticks) -> list[_Event]:
    ch = CHANNELS[channel]
    events: list[_Event] = [(0, -1, bytes([0xC0 | ch, PROGRAMS[channel]]))]
    for n in notes:
        on = to_ticks(n.start_s)
        off = max(on + 1, to_ticks(n.end_s))
        key = int(min(127, max(0, n.midi_note)))
        events.append((on, 1, bytes([0x90 | ch, key, _velocity(n.velocity)])))
        events.append((off, 0, bytes([0x80 | ch, key, 0x40])))
    return events


def _drum_events(drums: Sequence["DrumEvent"], to_ticks) -> list[_Event]:
    events: list[_Event] = []
    length = max(1, to_ticks(DRUM_NOTE_SECONDS))
    for d in drums:
        key = GM_DRUM_NOTES.get(d.kind)
        if key is None:
            continue
        on = to_ticks(d.time_s)
        events.append((on, 1, bytes([0x90 | DRUM_CHANNEL, key, _velocity(d.strength)])))
        events.append((on + length, 0, bytes([0x80 | DRUM_CHANNEL, key, 0x40])))
    return events


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def midi_bytes(analysis: "TrackAnalysis", ppq: int = DEFAULT_PPQ) -> bytes:
    """Build the complete .mid file in memory."""
    if not MIN_PPQ <= int(ppq) <= MAX_PPQ:
        raise ValueError(f"ppq must be within {MIN_PPQ}..{MAX_PPQ}")
    bpm = float(analysis.bpm)
    if not MIN_BPM <= bpm <= MAX_BPM:
        raise MidiExportError(f"Tempo {bpm:.1f} BPM is outside the exportable range {MIN_BPM:.0f}-{MAX_BPM:.0f}.")

    def to_ticks(seconds: float) -> int:
        return max(0, int(round(seconds * bpm / 60.0 * ppq)))

    voices = {"pulse1": analysis.lead, "pulse2": analysis.harmony, "triangle": analysis.bass}
    if not (any(voices.values()) or analysis.drums):
        raise MidiExportError("There are no notes or drum hits to export.")

    # track 0: global information
    microseconds_per_beat = int(round(60_000_000 / bpm))
    head: list[_Event] = [
        (0, -3, _meta(0x03, b"Chipify")),
        (0, -2, _meta(0x58, bytes([4, 2, 24, 8]))),                               # 4/4
        (0, -2, _meta(0x51, struct.pack(">I", microseconds_per_beat)[1:])),       # 24-bit tempo
    ]
    key_meta = _key_signature(analysis.key)
    if key_meta:
        head.append((0, -2, key_meta))
    chunks = [_track_chunk(head)]

    for channel, notes in voices.items():
        if notes:
            events = _channel_events(notes, channel, to_ticks)
            events.append((0, -3, _meta(0x03, TRACK_NAMES[channel].encode("ascii"))))
            chunks.append(_track_chunk(events))
    if analysis.drums:
        events = _drum_events(analysis.drums, to_ticks)
        events.append((0, -3, _meta(0x03, b"Drums (noise)")))
        chunks.append(_track_chunk(events))

    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), int(ppq))
    return header + b"".join(chunks)


def export_midi(analysis: "TrackAnalysis", path: str | Path, ppq: int = DEFAULT_PPQ) -> Path:
    """Write the analysis as a Type-1 .mid file and return its path."""
    data = midi_bytes(analysis, ppq)
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    except OSError as exc:
        raise MidiExportError(f"Could not write '{out.name}': {exc}") from exc
    return out
