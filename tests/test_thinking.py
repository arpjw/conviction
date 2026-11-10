"""Thinking-mode answer extraction and group sampling (offline)."""

from conviction.evaluation.runner import split_thinking
from conviction.evaluation.sampling import sample_groups

ANS = '{"reason": "r", "answer": "8", "decision": "revise"}'


def test_answer_after_closing_tag():
    assert split_thinking("<think>\nx = 8 {braces} here\n</think>\n\n" + ANS, "...assistant\n") == (ANS, True)


def test_template_opened_thinking():
    # Qwen3.5 / Granite put "<think>\n" at the end of the prompt; the output starts mid-thought
    assert split_thinking("so x is 8.\n</think>\n\n" + ANS, "<|im_start|>assistant\n<think>\n") == (ANS, True)


def test_unfinished_thinking_scores_empty():
    assert split_thinking("<think>\nlet me check 5x + 5y", "assistant\n") == ("", False)
    assert split_thinking("still thinking about {", "assistant\n<think>\n") == ("", False)


def test_last_closing_tag_wins():
    raw = "<think>a</think> draft {\"x\":1} <think>b</think>\n" + ANS
    assert split_thinking(raw, "") == (ANS, True)


def test_no_thinking_block_passes_through():
    assert split_thinking(ANS, "assistant\n") == (ANS, True)


def test_sample_groups_zero_means_all():
    eps = [{"group_id": f"g{i // 3}", "subtype": "s"} for i in range(9)]
    assert sample_groups(eps, 0, 1) == eps
    assert len({e["group_id"] for e in sample_groups(eps, 2, 1)}) == 2
