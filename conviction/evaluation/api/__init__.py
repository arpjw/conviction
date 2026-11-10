"""Evaluate frontier models on ConvictionBench through provider APIs (``conviction-eval-api``).

Separate from the HF runner because API models differ in ways that change the protocol:
no greedy decoding on several providers, reasoning that cannot always be disabled, and tool
records that must be expressed in each provider's native tool-call format. Results are written
in the same layout as ``conviction-eval`` so the analysis and site export work unchanged.
"""
