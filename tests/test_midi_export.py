"""Tests for chipify.midi_export. An independent byte-level SMF parser checks the writer."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from chipify import chip_tables as ct
from chipify import midi_export as me
from chipify.pitch_tracker import DrumEvent, TrackAnalysis, make_note

NES = ct.get_chip("nes")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def lead(start, end, midi=69, vel=1.0):
    return make_note(start, end, midi, vel, "pulse1", 1.0, NES.pulse)


def bass(start, end, midi=36, vel=1.0):
    return make_note(start, end, midi, vel, "triangle", 1.0, NES.bass)


def analysis(lead_notes=(), harmony=(), bass_notes=(), drums=(), bpm=120.0, key="F minor"):
    return TrackAnalysis(
        sr=22_050, duration_s=4.0, chip="nes", bpm=bpm, bpm_source="user",
        grid_step_s=60 / bpm / 4, grid_offset_s=0.0, grid_fit=1.0, key=key,
        lead=list(lead_notes), harmony=list(harmony), bass=list(bass_notes), drums=list(drums),
        f0_times=np.zeros(0), f0_lead_hz=np.zeros(0), f0_bass_hz=np.zeros(0),
    )


def read_vlq(data, pos):
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def parse_smf(data):
    """Minimal independent SMF reader -> (format, ppq, [track events]); events are dicts with abs tick."""
    assert data[:4] == b"MThd"
    length, fmt, ntrks, ppq = struct.unpack(">IHHH", data[4:14])
    assert length == 6
    pos, tracks = 14, []
    for _ in range(ntrks):
        assert data[pos:pos + 4] == b"MTrk"
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        pos += 8
        end, tick, status, events = pos + size, 0, None, []
        while pos < end:
            delta, pos = read_vlq(data, pos)
            tick += delta
            if data[pos] == 0xFF:
                kind = data[pos + 1]
                n, pos = read_vlq(data, pos + 2)
                events.append({"tick": tick, "meta": kind, "data": data[pos:pos + n]})
                pos += n
            else:
                status = data[pos]
                kind = status & 0xF0
                nbytes = 1 if kind in (0xC0, 0xD0) else 2
                events.append({"tick": tick, "type": kind, "ch": status & 0x0F,
                               "args": tuple(data[pos + 1:pos + 1 + nbytes])})
                pos += 1 + nbytes
        assert pos == end
        assert events[-1].get("meta") == 0x2F            # every track ends with End-of-Track
        tracks.append(events)
    assert pos == len(data)
    return fmt, ppq, tracks


def notes_of(track, kind):
    return [(e["tick"], e["args"]) for e in track if e.get("type") == kind]


# --------------------------------------------------------------------------- #
# Encoding primitives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [(0, b"\x00"), (127, b"\x7f"), (128, b"\x81\x00"), (16383, b"\xff\x7f"),
     (16384, b"\x81\x80\x00"), (0x0FFFFFFF, b"\xff\xff\xff\x7f")],
)
def test_vlq_matches_the_spec_examples(value, expected):
    assert me.vlq(value) == expected


def test_vlq_rejects_negative_values():
    with pytest.raises(ValueError):
        me.vlq(-1)


def test_key_signatures():
    sf = lambda key: struct.unpack(">bB", me._key_signature(key)[3:5])
    assert sf("C major") == (0, 0)
    assert sf("F minor") == (-4, 1)          # four flats
    assert sf("A minor") == (0, 1)
    assert sf("G major") == (1, 0)
    assert sf("D minor") == (-1, 1)
    assert me._key_signature(None) is None
    assert me._key_signature("nonsense") is None


# --------------------------------------------------------------------------- #
# File structure and content
# --------------------------------------------------------------------------- #
def test_header_tracks_tempo_and_signatures():
    a = analysis([lead(0.0, 0.5)], [lead(0.0, 0.5, 64)], [bass(0.0, 1.0)], [DrumEvent(0.0, "kick", 1.0)])
    fmt, ppq, tracks = parse_smf(me.midi_bytes(a))
    assert (fmt, ppq, len(tracks)) == (1, 480, 5)                       # tempo + lead + harmony + bass + drums
    meta = {e["meta"]: e["data"] for e in tracks[0] if "meta" in e}
    assert int.from_bytes(meta[0x51], "big") == 500_000                  # 120 BPM
    assert meta[0x58] == bytes([4, 2, 24, 8])
    assert struct.unpack(">bB", meta[0x59]) == (-4, 1)                   # F minor


def test_note_timing_follows_the_tempo():
    a = analysis([lead(1.0, 1.5, 69)], bpm=120.0)                        # 1 s = 2 beats = 960 ticks
    _, _, tracks = parse_smf(me.midi_bytes(a, ppq=480))
    on = notes_of(tracks[1], 0x90)
    off = notes_of(tracks[1], 0x80)
    assert on[0][0] == 960 and off[0][0] == 1440
    assert on[0][1][0] == off[0][1][0] == 69                             # NES pulse A4 rounds to MIDI 69


def test_tempo_scales_ticks():
    a = analysis([lead(1.0, 2.0)], bpm=90.0)                              # 1 s = 1.5 beats
    _, _, tracks = parse_smf(me.midi_bytes(a, ppq=96))
    assert notes_of(tracks[1], 0x90)[0][0] == 144
    meta = {e["meta"]: e["data"] for e in tracks[0] if "meta" in e}
    assert int.from_bytes(meta[0x51], "big") == round(60_000_000 / 90)


def test_channels_programs_and_names():
    a = analysis([lead(0, 1)], [lead(0, 1, 64)], [bass(0, 1)])
    _, _, tracks = parse_smf(me.midi_bytes(a))
    lead_t, harm_t, bass_t = tracks[1], tracks[2], tracks[3]
    assert [e["ch"] for e in lead_t if e.get("type") == 0x90] == [0]
    assert [e["ch"] for e in harm_t if e.get("type") == 0x90] == [1]
    assert [e["ch"] for e in bass_t if e.get("type") == 0x90] == [2]
    assert [e["args"] for e in lead_t if e.get("type") == 0xC0] == [(80,)]
    assert [e["args"] for e in bass_t if e.get("type") == 0xC0] == [(38,)]
    assert next(e["data"] for e in lead_t if e.get("meta") == 0x03) == b"Lead (pulse 1)"


def test_drums_use_general_midi_on_channel_10():
    drums = [DrumEvent(0.0, "kick", 1.0), DrumEvent(0.5, "snare", 0.5), DrumEvent(1.0, "hat", 0.1)]
    _, _, tracks = parse_smf(me.midi_bytes(analysis(drums=drums)))
    ons = notes_of(tracks[1], 0x90)
    assert [args[0] for _, args in ons] == [36, 38, 42]
    assert all(e["ch"] == 9 for e in tracks[1] if e.get("type") in (0x90, 0x80))


def test_velocity_mapping_and_pitch_clamping():
    a = analysis([lead(0, 1, vel=1.0), lead(1, 2, vel=0.0)])
    _, _, tracks = parse_smf(me.midi_bytes(a))
    assert [args[1] for _, args in notes_of(tracks[1], 0x90)] == [127, 1]


def test_note_off_sorts_before_note_on_at_the_same_tick():
    a = analysis([lead(0.0, 0.5, 69), lead(0.5, 1.0, 69)])
    _, _, tracks = parse_smf(me.midi_bytes(a))
    at_480 = [e for e in tracks[1] if e["tick"] == 480 and e.get("type") in (0x80, 0x90)]
    assert [e["type"] for e in at_480] == [0x80, 0x90]


def test_zero_length_notes_get_one_tick():
    _, _, tracks = parse_smf(me.midi_bytes(analysis([lead(1.0, 1.0)])))
    on, off = notes_of(tracks[1], 0x90)[0][0], notes_of(tracks[1], 0x80)[0][0]
    assert off == on + 1


def test_empty_tracks_are_omitted():
    _, _, tracks = parse_smf(me.midi_bytes(analysis(bass_notes=[bass(0, 1)])))
    assert len(tracks) == 2


# --------------------------------------------------------------------------- #
# Errors and file output
# --------------------------------------------------------------------------- #
def test_empty_analysis_raises():
    with pytest.raises(me.MidiExportError):
        me.midi_bytes(analysis())


def test_out_of_range_tempo_raises():
    with pytest.raises(me.MidiExportError):
        me.midi_bytes(analysis([lead(0, 1)], bpm=5.0))


def test_invalid_ppq_is_a_programmer_error():
    with pytest.raises(ValueError):
        me.midi_bytes(analysis([lead(0, 1)]), ppq=1)


def test_export_midi_writes_a_parsable_file(tmp_path):
    out = me.export_midi(analysis([lead(0, 1)]), tmp_path / "nested" / "song.mid")
    assert out.exists()
    fmt, ppq, tracks = parse_smf(out.read_bytes())
    assert fmt == 1 and ppq == 480 and len(tracks) == 2


def test_real_mido_can_read_the_file(tmp_path):
    mido = pytest.importorskip("mido")
    a = analysis([lead(0.0, 0.5, 69), lead(0.5, 1.0, 72)], bass_notes=[bass(0.0, 1.0)],
                 drums=[DrumEvent(0.0, "kick", 1.0)], bpm=100.0)
    path = me.export_midi(a, tmp_path / "x.mid")
    mid = mido.MidiFile(str(path))
    assert mid.type == 1 and mid.ticks_per_beat == 480 and len(mid.tracks) == 4
    tempos = [m.tempo for m in mid.tracks[0] if m.type == "set_tempo"]
    assert mido.tempo2bpm(tempos[0]) == pytest.approx(100.0, rel=1e-3)
    ons = [m for t in mid.tracks for m in t if m.type == "note_on"]
    assert len(ons) == 4
    assert {m.channel for m in ons} == {0, 2, 9}
