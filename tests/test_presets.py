"""Tests for chipify.presets: presets are valid and user knobs override them correctly."""
from __future__ import annotations

import pytest

from chipify import presets as pr


def test_three_presets_exist_with_the_documented_names():
    assert pr.PRESET_NAMES == ("NES 8-Bit", "Game Boy Mono", "Arcade Synth")
    assert pr.DEFAULT_PRESET in pr.PRESETS
    for p in pr.PRESETS.values():
        assert p.description


@pytest.mark.parametrize("name", list(pr.PRESET_NAMES))
def test_every_preset_builds_valid_settings(name):
    analysis, synth = pr.build_settings(name)
    analysis.validate()
    synth.validate()


def test_presets_target_the_right_chips_and_mixers():
    nes_a, nes_s = pr.build_settings("NES 8-Bit")
    gb_a, gb_s = pr.build_settings("Game Boy Mono")
    arc_a, arc_s = pr.build_settings("Arcade Synth")
    assert (nes_a.chip, nes_s.mixer) == ("nes", "nes")
    assert (gb_a.chip, gb_s.mixer) == ("gameboy", "linear")
    assert (arc_a.chip, arc_s.mixer) == ("ideal", "linear")
    assert nes_s.strict_duty and not arc_s.strict_duty
    assert arc_s.pwm_depth > 0 and nes_s.pwm_depth == 0
    assert arc_a.quant_strictness < nes_a.quant_strictness


def test_no_knobs_means_pure_preset_values():
    analysis, synth = pr.build_settings("NES 8-Bit")
    d = pr.preset_defaults("NES 8-Bit")
    assert analysis.quant_strictness == d["strictness"]
    assert analysis.harmony_mode == d["harmony"]
    assert synth.pwm_depth == d["pwm_depth"] and synth.bitcrush == d["bitcrush"]
    assert analysis.bpm is None


def test_knobs_override_the_preset():
    knobs = pr.UserKnobs(strictness=0.4, bpm=130.0, harmony="octave_up", drums=False, steps_per_beat=2,
                         pwm_depth=0.9, bitcrush=0.7, lead_duty=0.125, harmony_duty=0.5,
                         drum_gain=1.5, bass_dynamics=True, strict_duty=False)
    analysis, synth = pr.build_settings("NES 8-Bit", knobs)
    assert (analysis.quant_strictness, analysis.bpm, analysis.harmony_mode) == (0.4, 130.0, "octave_up")
    assert analysis.drums_enabled is False and analysis.steps_per_beat == 2
    assert (synth.pwm_depth, synth.bitcrush, synth.lead_duty, synth.harmony_duty) == (0.9, 0.7, 0.125, 0.5)
    assert synth.drum_gain == 1.5 and synth.triangle_dynamics is True and synth.strict_duty is False
    assert analysis.chip == "nes" and synth.mixer == "nes"          # chip identity still comes from the preset


def test_none_knobs_do_not_disturb_the_preset():
    _, synth = pr.build_settings("Arcade Synth", pr.UserKnobs(bitcrush=0.3))
    assert synth.bitcrush == 0.3
    assert synth.pwm_depth == 0.5                                    # untouched preset value


def test_building_settings_never_mutates_the_shared_presets():
    pr.build_settings("NES 8-Bit", pr.UserKnobs(bitcrush=0.9, strictness=0.1))
    _, synth = pr.build_settings("NES 8-Bit")
    assert synth.bitcrush == 0.0
    assert pr.build_settings("NES 8-Bit")[0].quant_strictness == 1.0


def test_unknown_preset_lists_valid_names():
    with pytest.raises(ValueError) as err:
        pr.build_settings("Commodore 64")
    assert "NES 8-Bit" in str(err.value)


@pytest.mark.parametrize(
    "knobs",
    [pr.UserKnobs(strictness=2.0), pr.UserKnobs(bpm=5.0), pr.UserKnobs(harmony="fifths"),
     pr.UserKnobs(bitcrush=-1.0), pr.UserKnobs(pwm_depth=3.0), pr.UserKnobs(lead_duty=0.9)],
)
def test_out_of_range_knobs_are_rejected(knobs):
    with pytest.raises(ValueError):
        pr.build_settings("NES 8-Bit", knobs)


def test_preset_defaults_has_everything_the_gui_needs():
    keys = {"strictness", "harmony", "pwm_depth", "bitcrush", "lead_duty", "harmony_duty", "drum_gain"}
    for name in pr.PRESET_NAMES:
        d = pr.preset_defaults(name)
        assert keys <= set(d)
        assert d["harmony"] in pr.HARMONY_MODES
