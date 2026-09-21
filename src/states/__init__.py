"""State machine states implementing the constrained JSON grammar.

Each module defines one State (see State.py) representing one segment of
the target JSON document, e.g. the opening brace, the function name, a
parameter value, etc. States are chained together to form the full
grammar for:

{
  "name": "<function name>",
  "parameters": {
    "<param name>": <param value>,
    ...
  }
}
"""
