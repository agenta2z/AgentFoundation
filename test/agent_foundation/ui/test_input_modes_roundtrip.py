"""Round-trip tests for InputModeConfig serialization.

Covers ChoiceOption.description, the 'multiple_choices' backward-compat alias,
and full to_dict/from_dict round-trips for all input modes.
"""
import pytest

from agent_foundation.ui.input_modes import (
    ChoiceOption,
    InputMode,
    InputModeConfig,
    press_to_continue,
    exact_string,
    single_choice,
    multiple_choices,
)


def test_choice_option_description_field():
    """ChoiceOption has a description field that defaults to empty."""
    opt = ChoiceOption(label="A", value="a")
    assert opt.description == ""

    opt_with_desc = ChoiceOption(label="B", value="b", description="B option")
    assert opt_with_desc.description == "B option"


def test_description_serialization():
    """ChoiceOption.description round-trips through to_dict/from_dict."""
    config = InputModeConfig(
        mode=InputMode.SINGLE_CHOICE,
        options=[
            ChoiceOption(label="A", value="a", description="First option"),
            ChoiceOption(label="B", value="b"),  # no description
        ],
    )
    d = config.to_dict()
    assert d["options"][0]["description"] == "First option"
    assert "description" not in d["options"][1]  # empty descriptions omitted

    restored = InputModeConfig.from_dict(d)
    assert restored.options[0].description == "First option"
    assert restored.options[1].description == ""


def test_multiple_choices_alias_from_dict():
    """from_dict accepts 'multiple_choices' (plural) as a backward-compat alias."""
    d = {"mode": "multiple_choices", "prompt": "Select items"}
    config = InputModeConfig.from_dict(d)
    assert config.mode == InputMode.MULTIPLE_CHOICE
    assert config.prompt == "Select items"


def test_invalid_mode_raises():
    """from_dict raises ValueError for truly invalid mode strings."""
    with pytest.raises(ValueError):
        InputModeConfig.from_dict({"mode": "nonexistent_mode"})


def test_free_text_roundtrip():
    """FREE_TEXT mode round-trips."""
    config = InputModeConfig(mode=InputMode.FREE_TEXT, prompt="Enter text")
    d = config.to_dict()
    restored = InputModeConfig.from_dict(d)
    assert restored.mode == InputMode.FREE_TEXT
    assert restored.prompt == "Enter text"


def test_exact_string_roundtrip():
    """EXACT_STRING mode round-trips with expected_string and case_sensitive."""
    config = exact_string("YES", prompt="Confirm", case_sensitive=True)
    d = config.to_dict()
    restored = InputModeConfig.from_dict(d)
    assert restored.mode == InputMode.EXACT_STRING
    assert restored.expected_string == "YES"
    assert restored.case_sensitive is True


def test_single_choice_roundtrip():
    """SINGLE_CHOICE mode round-trips with options."""
    config = single_choice(
        [
            ChoiceOption(label="Yes", value="yes", description="Approve it"),
            ChoiceOption(label="No", value="no", follow_up_prompt="Why not?"),
        ],
        allow_custom=False,
        prompt="Choose",
    )
    d = config.to_dict()
    restored = InputModeConfig.from_dict(d)
    assert restored.mode == InputMode.SINGLE_CHOICE
    assert len(restored.options) == 2
    assert restored.options[0].description == "Approve it"
    assert restored.options[1].follow_up_prompt == "Why not?"
    assert restored.allow_custom is False


def test_multiple_choices_roundtrip_with_select_all():
    """MULTIPLE_CHOICE mode round-trips with show_select_all/select_all_text."""
    config = multiple_choices(
        [ChoiceOption(label="A", value="a"), ChoiceOption(label="B", value="b")],
        show_select_all=False,
        select_all_text="Pick all",
    )
    d = config.to_dict()
    assert d["show_select_all"] is False
    assert d["select_all_text"] == "Pick all"

    restored = InputModeConfig.from_dict(d)
    assert restored.show_select_all is False
    assert restored.select_all_text == "Pick all"


def test_press_to_continue_roundtrip():
    """PRESS_TO_CONTINUE mode round-trips."""
    config = press_to_continue("Press enter to continue")
    d = config.to_dict()
    restored = InputModeConfig.from_dict(d)
    assert restored.mode == InputMode.PRESS_TO_CONTINUE
    assert restored.prompt == "Press enter to continue"


def test_metadata_roundtrip():
    """Metadata dict round-trips."""
    config = InputModeConfig(
        mode=InputMode.FREE_TEXT,
        metadata={"widget_type": "confirmation", "extra": 42},
    )
    d = config.to_dict()
    assert d["metadata"]["widget_type"] == "confirmation"

    restored = InputModeConfig.from_dict(d)
    assert restored.metadata["extra"] == 42
