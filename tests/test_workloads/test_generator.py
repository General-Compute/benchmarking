import pytest

from benchmarking.workloads.generator import PromptGenerator


def test_generate_prompt_hits_target_token_count() -> None:
    generator = PromptGenerator()

    prompt = generator.generate_prompt(target_tokens=256, template="qa")

    assert generator.validate_token_count(prompt, expected=256)


def test_generate_prompt_rejects_unknown_template() -> None:
    generator = PromptGenerator()

    with pytest.raises(ValueError, match="Unknown template type"):
        generator.generate_prompt(target_tokens=256, template="unknown")


def test_generate_prompt_rejects_tiny_target() -> None:
    generator = PromptGenerator()

    with pytest.raises(ValueError, match="at least 10"):
        generator.generate_prompt(target_tokens=9, template="qa")
