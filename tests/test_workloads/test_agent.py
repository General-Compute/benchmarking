from benchmarking.workloads.agent import (
    AGENT_SYSTEM_PREAMBLE,
    AGENT_TASK,
    agent_system_prompt,
    continue_transcript,
    initial_messages,
    tool_result,
)
from benchmarking.workloads.generator import PromptGenerator


def test_agent_system_prompt_size_includes_task() -> None:
    gen = PromptGenerator()

    system = agent_system_prompt(gen, 2048, "code")

    assert system.startswith(AGENT_SYSTEM_PREAMBLE)
    total = gen._count_tokens(system) + gen._count_tokens(AGENT_TASK)
    # The preamble/body boundary can merge a token or two when re-encoded.
    assert abs(total - 2048) <= 2


def test_tool_results_are_exact_and_distinct() -> None:
    gen = PromptGenerator()

    results = [tool_result(gen, step, 512) for step in (1, 2, 3)]

    assert [gen._count_tokens(r) for r in results] == [512, 512, 512]
    assert len(set(results)) == 3
    assert results[1].startswith("Tool result for step 2:")


def test_tiny_tool_result_still_valid() -> None:
    gen = PromptGenerator()

    assert gen._count_tokens(tool_result(gen, 7, 12)) == 12


def test_continue_transcript_appends_reply_and_result() -> None:
    messages = initial_messages("system text")

    nxt = continue_transcript(messages, ' {"tool": "search"} ', "tool output")

    assert [m["role"] for m in nxt] == ["system", "user", "assistant", "user"]
    assert nxt[2]["content"] == '{"tool": "search"}'
    assert nxt[3]["content"] == "tool output"
    assert len(messages) == 2  # input transcript is not mutated


def test_continue_transcript_replaces_empty_reply() -> None:
    nxt = continue_transcript(initial_messages("s"), "", "out")

    assert nxt[2]["content"]
