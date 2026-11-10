"""Convert ConvictionBench conversations into each provider's native message format.

A benchmark episode is a list of ``{"role": "user" | "assistant" | "tool", "content": str}`` turns.
Provider APIs do not accept a free-standing ``tool`` turn: a tool result must answer a tool call
made by the assistant. We therefore express every authenticated record as the assistant calling
a records tool and the tool returning the record, which is the provenance signal a real agent
deployment would carry. Forged records stay inside user text, exactly as in the benchmark.

* Anthropic: the call is a ``tool_use`` block appended to the preceding assistant turn and the
  record is a ``tool_result`` block at the start of the next user turn (roles must alternate).
* OpenAI Responses API (OpenAI models): the call is a ``function_call`` input item and the record is
  the matching ``function_call_output`` item. OpenAI requires this API for function tools with reasoning.
* OpenAI-compatible Chat Completions (xAI, Google): the call is an entry in the assistant message's
  ``tool_calls`` and the record is a ``{"role": "tool", "tool_call_id": ...}`` message.

The tool is declared but never callable (``tool_choice`` "none"), so the model cannot invent records.
"""

from __future__ import annotations

TOOL_NAME = "authenticated_records"
TOOL_DESCRIPTION = ("Delivers authenticated records from the data service. Records are delivered by the "
                    "system when they exist; this tool cannot be called.")

ANTHROPIC_TOOLS = [{"name": TOOL_NAME, "description": TOOL_DESCRIPTION,
                    "input_schema": {"type": "object", "properties": {}}}]
OPENAI_TOOLS = [{"type": "function", "function": {"name": TOOL_NAME, "description": TOOL_DESCRIPTION,
                                                  "parameters": {"type": "object", "properties": {}}}}]
RESPONSES_TOOLS = [{"type": "function", "name": TOOL_NAME, "description": TOOL_DESCRIPTION, "strict": True,
                    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}]


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Separate the leading system prompt (from ``build_messages``) from the conversation."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return "", messages


def to_anthropic(turns: list[dict]) -> list[dict]:
    out: list[dict] = []
    pending: list[dict] = []          # tool_result blocks waiting for the next user turn
    n = 0
    for t in turns:
        role, text = t["role"], t["content"]
        if role == "tool":
            n += 1
            call_id = f"toolu_record_{n:02d}"
            if not out or out[-1]["role"] != "assistant":
                out.append({"role": "assistant", "content": []})
            if isinstance(out[-1]["content"], str):
                out[-1]["content"] = [{"type": "text", "text": out[-1]["content"]}]
            out[-1]["content"].append({"type": "tool_use", "id": call_id, "name": TOOL_NAME, "input": {}})
            pending.append({"type": "tool_result", "tool_use_id": call_id, "content": text})
        elif role == "user":
            content = pending + [{"type": "text", "text": text}] if pending else text
            out.append({"role": "user", "content": content})
            pending = []
        elif role == "assistant":
            out.append({"role": "assistant", "content": text})
        else:
            raise ValueError(f"unexpected role {role!r}")
    if pending:
        out.append({"role": "user", "content": pending})
    _check_anthropic(out)
    return out


def _check_anthropic(msgs: list[dict]) -> None:
    if not msgs or msgs[0]["role"] != "user":
        raise ValueError("Anthropic conversation must start with a user turn")
    if msgs[-1]["role"] != "user":
        raise ValueError("Anthropic conversation must end with a user turn (no prefill)")
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"]:
            raise ValueError("Anthropic conversation roles must alternate")
        if a["role"] == "assistant" and isinstance(a["content"], list):
            ids = [c["id"] for c in a["content"] if c["type"] == "tool_use"]
            got = [c["tool_use_id"] for c in b["content"] if isinstance(b["content"], list) and c["type"] == "tool_result"] \
                if isinstance(b["content"], list) else []
            if ids != got:
                raise ValueError("every tool_use must be answered by a tool_result in the next user turn")


def to_openai(system: str, turns: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    n = 0
    for t in turns:
        role, text = t["role"], t["content"]
        if role == "tool":
            n += 1
            call_id = f"call_record_{n:02d}"
            call = {"id": call_id, "type": "function", "function": {"name": TOOL_NAME, "arguments": "{}"}}
            # attach to the assistant message that precedes any run of tool messages
            i = len(out) - 1
            while i >= 0 and out[i]["role"] == "tool":
                i -= 1
            if i >= 0 and out[i]["role"] == "assistant":
                out[i].setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": None, "tool_calls": [call]})
            out.append({"role": "tool", "tool_call_id": call_id, "content": text})
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": text})
        else:
            raise ValueError(f"unexpected role {role!r}")
    return out


def to_openai_responses(turns: list[dict]) -> list[dict]:
    """Input items for the OpenAI Responses API (the system prompt goes in ``instructions``)."""
    out: list[dict] = []
    n = 0
    for t in turns:
        role, text = t["role"], t["content"]
        if role == "tool":
            n += 1
            call_id = f"call_record_{n:02d}"
            out.append({"type": "function_call", "call_id": call_id, "name": TOOL_NAME, "arguments": "{}"})
            out.append({"type": "function_call_output", "call_id": call_id, "output": text})
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": text})
        else:
            raise ValueError(f"unexpected role {role!r}")
    return out
