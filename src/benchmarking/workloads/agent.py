"""Multi-step agent transcripts for ``kind: agent`` workloads.

An agent step is not a chatbot turn: each request re-sends a long, mostly
unchanged context (system prompt, tool catalog, task, earlier tool results)
and asks for a short, structured reply such as a tool call. These helpers
build that transcript deterministically so every provider sees the same
requests in the same order.
"""

from .generator import PromptGenerator

AGENT_SYSTEM_PREAMBLE = (
    "You are a coding agent working through a multi-step task. At each step, reply with "
    "exactly one tool call as a JSON object with the keys 'tool' and 'arguments', and "
    "nothing else. Available tools: read_file(path), search(query), run_tests(target), "
    "edit_file(path, patch).\n\nReference material:\n"
)
AGENT_TASK = (
    "Task: find the cause of the failing test in the reference material and propose a fix. "
    "Start with your first tool call."
)


def agent_system_prompt(generator: PromptGenerator, tokens: int, template: str) -> str:
    """System prompt of roughly ``tokens`` tokens: agent instructions plus reference text."""
    preamble_tokens = generator._count_tokens(AGENT_SYSTEM_PREAMBLE)
    task_tokens = generator._count_tokens(AGENT_TASK)
    body_tokens = max(10, tokens - preamble_tokens - task_tokens)
    return AGENT_SYSTEM_PREAMBLE + generator.generate_prompt(body_tokens, template)


def tool_result(generator: PromptGenerator, step: int, tokens: int) -> str:
    """Deterministic tool output of exactly ``tokens`` tokens for the given step.

    Each step's output starts with a different header, so consecutive tool
    results are distinct text rather than repeats of the same block.
    """
    header = f"Tool result for step {step}:\n"
    header_tokens = generator._count_tokens(header)
    if tokens <= header_tokens + 10:
        return generator.generate_prompt(max(10, tokens), "code")
    return header + generator.generate_prompt(tokens - header_tokens, "code")


def initial_messages(system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AGENT_TASK},
    ]


def continue_transcript(
    messages: list[dict[str, str]], reply: str, result: str
) -> list[dict[str, str]]:
    """Transcript for the next step: append the model's reply and the tool result.

    An empty reply (for example a reasoning model that spent its budget
    thinking) is replaced with a placeholder so the transcript stays valid.
    """
    return [
        *messages,
        {"role": "assistant", "content": reply.strip() or '{"tool": "read_file"}'},
        {"role": "user", "content": result},
    ]
