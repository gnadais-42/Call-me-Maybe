"""Constrained-decoding function calling tool.

Translates natural language prompts into structured function calls by
driving a small LLM's token generation through a grammar-aware state
machine, guaranteeing syntactically and semantically valid JSON output.
"""
