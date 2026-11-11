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


def test_generate_vllm_with_fake_engine(monkeypatch):
    """The vLLM path passes the same token ids as the HF path and returns texts + generated lengths."""
    import sys
    import types
    from conviction.evaluation.runner import generate_vllm

    class SP:
        def __init__(self, **kw):
            self.kw = kw
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SP))

    class Tok:
        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(c) for c in text]}

    seen = {}

    class LLM:
        def generate(self, prompts, sp):
            seen["prompts"], seen["sp"] = prompts, sp.kw
            mk = lambda t, n: types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t, token_ids=list(range(n)))])
            return [mk("a", 3), mk("b", 5)]

    lens = []
    out = generate_vllm(LLM(), Tok(), ["hi", "yo"], 4096,
                        {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}, 1, lens)
    assert out == ["a", "b"] and lens == [3, 5]
    assert seen["prompts"][0] == {"prompt_token_ids": [104, 105]}
    assert seen["sp"] == {"max_tokens": 4096, "seed": 1, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
