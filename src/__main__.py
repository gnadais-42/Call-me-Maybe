import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import FunctionCallingError
from .context import build_token_cache
from .generation import generate_function_call
from .parser import load_function_definitions, load_prompts
from llm_sdk import Small_LLM_Model

DEFAULT_FUNCTIONS_DEFINITION = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the function calling tool."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural language prompts into structured "
        "function calls using constrained decoding.",
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the natural language prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write the function calling results JSON file.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """Load inputs, run generation for every prompt, write the output file.

    Returns:
        Process exit code: 0 on success, 1 on a handled, reported error.
    """
    args = parse_args(argv)

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompts(args.input)
    except FunctionCallingError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        model = Small_LLM_Model()
    except Exception as error:  # SDK init failures are not ours to predict
        print(f"Error: could not load the language model: {error}", file=sys.stderr)
        return 1

    # Built once and reused for every prompt: the vocabulary is identical
    # across the whole run, so this turns an O(num_prompts * vocab_size)
    # cost into O(vocab_size). See context.build_token_cache.
    token_cache = build_token_cache(model)

    results: list[dict[str, Any]] = []

    for prompt in prompts:
        try:
            call = generate_function_call(model, functions, prompt, token_cache)
        except FunctionCallingError as error:
            print(
                f"Warning: skipping prompt {prompt!r} due to a generation "
                f"error: {error}",
                file=sys.stderr,
            )
            continue

        results.append(
            {
                "prompt": prompt,
                "name": call.get("name"),
                "parameters": call.get("parameters", {}),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except OSError as error:
        print(f"Error: could not write output file {args.output}: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} function call(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
