"""
Presets and user knobs for Chipify.

A *preset* is a complete, named configuration (emulated chip + analysis settings +
synth settings). A *knob* is something the user can turn on top of it (quantisation
strictness, PWM, bit-crush, BPM...). ``build_settings`` merges the two, with any
knob left as ``None`` falling back to the preset's value, so the GUI can pass
whatever its widgets currently say without caring about defaults.

The GUI should call ``preset_defaults`` when the preset dropdown changes, to move
its sliders to the new preset's values.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .pitch_tracker import HARMONY_MODES, AnalysisSettings
from .synth_engine import Adsr, SynthSettings


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    analysis: dict[str, Any] = field(default_factory=dict)    # AnalysisSettings field overrides
    synth: dict[str, Any] = field(default_factory=dict)       # SynthSettings field overrides


PRESETS: dict[str, Preset] = {
    "NES 8-Bit": Preset(
        name="NES 8-Bit",
        description=(
            "Ricoh 2A03: hardware pitch registers, 25% and 12.5% pulse duty, 4-bit stepped "
            "triangle bass, LFSR drums and the console's non-linear mixer."
        ),
        analysis={"chip": "nes", "quant_strictness": 1.0, "harmony_mode": "third_below"},
        synth={
            "mixer": "nes", "lead_duty": 0.25, "harmony_duty": 0.125, "strict_duty": True,
            "pwm_depth": 0.0, "bitcrush": 0.0, "drum_gain": 1.0, "triangle_dynamics": False,
            "lead_env": Adsr(0.0, 0.18, 0.75, 0.04), "harmony_env": Adsr(0.0, 0.12, 0.60, 0.03),
        },
    ),
    "Game Boy Mono": Preset(
        name="Game Boy Mono",
        description=(
            "LR35902: pitch registers of the Game Boy, wide 50% pulse lead, wave-channel bass, "
            "linear mixing and a touch of speaker grit."
        ),
        analysis={"chip": "gameboy", "quant_strictness": 1.0, "harmony_mode": "third_below"},
        synth={
            "mixer": "linear", "lead_duty": 0.5, "harmony_duty": 0.25, "strict_duty": True,
            "pwm_depth": 0.0, "bitcrush": 0.12, "drum_gain": 0.8, "triangle_dynamics": False,
            "lead_env": Adsr(0.0, 0.25, 0.65, 0.05), "harmony_env": Adsr(0.0, 0.18, 0.55, 0.04),
        },
    ),
    "Arcade Synth": Preset(
        name="Arcade Synth",
        description=(
            "Unconstrained equal temperament, looser timing, sweeping pulse-width modulation, "
            "expressive bass dynamics and smoother envelopes: chiptune character, arcade-cabinet polish."
        ),
        analysis={"chip": "ideal", "quant_strictness": 0.85, "harmony_mode": "third_above"},
        synth={
            "mixer": "linear", "lead_duty": 0.25, "harmony_duty": 0.25, "strict_duty": False,
            "pwm_depth": 0.5, "pwm_rate_hz": 4.0, "bitcrush": 0.0, "drum_gain": 0.9,
            "triangle_dynamics": True,
            "lead_env": Adsr(0.003, 0.25, 0.70, 0.12), "harmony_env": Adsr(0.003, 0.20, 0.55, 0.10),
        },
    ),
}

PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)
DEFAULT_PRESET: str = "NES 8-Bit"


@dataclass
class UserKnobs:
    """GUI-facing overrides. ``None`` means "use the preset's value"."""
    strictness: float | None = None        # 0..1 quantisation strictness (pitch + timing)
    bpm: float | None = None               # None = auto-detect tempo
    harmony: str | None = None             # one of HARMONY_MODES
    drums: bool | None = None
    steps_per_beat: int | None = None
    pwm_depth: float | None = None         # 0..1
    bitcrush: float | None = None          # 0..1
    lead_duty: float | None = None
    harmony_duty: float | None = None
    drum_gain: float | None = None
    bass_dynamics: bool | None = None
    strict_duty: bool | None = None


def get_preset(name: str) -> Preset:
    """Look up a preset by name; ValueError lists the valid names."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset '{name}'. Choose one of: {', '.join(PRESET_NAMES)}") from None


def preset_defaults(name: str) -> dict[str, Any]:
    """
    Values the GUI should show for a preset: strictness, harmony, pwm_depth,
    bitcrush, lead_duty, harmony_duty, drum_gain. (BPM has no preset value: auto.)
    """
    p = get_preset(name)
    base_analysis, base_synth = AnalysisSettings(), SynthSettings()
    return {
        "strictness": p.analysis.get("quant_strictness", base_analysis.quant_strictness),
        "harmony": p.analysis.get("harmony_mode", base_analysis.harmony_mode),
        "pwm_depth": p.synth.get("pwm_depth", base_synth.pwm_depth),
        "bitcrush": p.synth.get("bitcrush", base_synth.bitcrush),
        "lead_duty": p.synth.get("lead_duty", base_synth.lead_duty),
        "harmony_duty": p.synth.get("harmony_duty", base_synth.harmony_duty),
        "drum_gain": p.synth.get("drum_gain", base_synth.drum_gain),
    }


def build_settings(
    preset_name: str = DEFAULT_PRESET, knobs: UserKnobs | None = None
) -> tuple[AnalysisSettings, SynthSettings]:
    """
    Merge a preset with user knobs into validated (AnalysisSettings, SynthSettings).

    Raises ValueError for unknown presets or out-of-range knob values.
    """
    preset = get_preset(preset_name)
    knobs = knobs or UserKnobs()

    analysis = replace(AnalysisSettings(), **preset.analysis)
    synth = replace(SynthSettings(), **preset.synth)

    if knobs.strictness is not None:
        analysis.quant_strictness = float(knobs.strictness)
    if knobs.bpm is not None:
        analysis.bpm = float(knobs.bpm)
    if knobs.harmony is not None:
        analysis.harmony_mode = knobs.harmony
    if knobs.drums is not None:
        analysis.drums_enabled = bool(knobs.drums)
    if knobs.steps_per_beat is not None:
        analysis.steps_per_beat = int(knobs.steps_per_beat)

    overrides = {
        "pwm_depth": knobs.pwm_depth, "bitcrush": knobs.bitcrush, "lead_duty": knobs.lead_duty,
        "harmony_duty": knobs.harmony_duty, "drum_gain": knobs.drum_gain,
        "triangle_dynamics": knobs.bass_dynamics, "strict_duty": knobs.strict_duty,
    }
    for attr, value in overrides.items():
        if value is not None:
            setattr(synth, attr, value)

    # The synth's mixer follows the chip unless the preset said otherwise.
    synth.mixer = preset.synth.get("mixer", "nes" if analysis.chip == "nes" else "linear")

    analysis.validate()
    synth.validate()
    return analysis, synth


__all__ = [
    "DEFAULT_PRESET", "HARMONY_MODES", "PRESETS", "PRESET_NAMES", "Preset", "UserKnobs",
    "build_settings", "get_preset", "preset_defaults",
]
