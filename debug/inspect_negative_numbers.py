"""One-off diagnostic: run this against your real model (with llm_sdk
copied in) to see exactly what the model's masked logits look like at the
first character of a number, for a prompt containing a negative value.

Usage (from the project root):
    uv run python debug/inspect_negative_numbers.py
"""
import sys
from pathlib import Path

# Running a script inside debug/ puts debug/ on sys.path, not the project
# root, so `import src...` would fail. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context import build_token_cache  # noqa: E402
from src.decoder import ConstrainedDecoder  # noqa: E402
from src.models import FunctionDefinition, Parameter, ReturnType  # noqa: E402
from src.prompt import build_function_selection_prompt  # noqa: E402
from src.state_machine import StateMachine  # noqa: E402
from llm_sdk import Small_LLM_Model  # noqa: E402


def main() -> None:
    model = Small_LLM_Model()
    token_cache = build_token_cache(model)

    functions = [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={"a": Parameter(type="number"), "b": Parameter(type="number")},
            returns=ReturnType(type="number"),
        )
    ]

    user_prompt = "What is the sum of -2 and 0?"
    context_prompt = build_function_selection_prompt(user_prompt, functions)
    prompt_tokens = model.encode(context_prompt)[0].tolist()

    machine = StateMachine(model, functions, token_cache)
    generated: list[int] = []

    # Drive generation up to, but not past, the very first character of
    # the first number's value.
    for _ in range(200):
        just_entered_number_state = (
            type(machine.state).__name__ == "NumberState"
            and len(generated) == machine.state.start_position  # type: ignore[attr-defined]
        )
        if just_entered_number_state:
            break

        allowed = machine.get_allowed_tokens(generated)
        if not allowed:
            break

        if len(allowed) == 1:
            next_token = allowed[0]
        else:
            logits = model.get_logits_from_input_ids(prompt_tokens + generated)
            masked = ConstrainedDecoder.mask_logits(logits, allowed)
            next_token = max(range(len(masked)), key=lambda i: masked[i])

        generated.append(next_token)
        machine.update(generated)

    print("Text so far:", repr(machine.context.decode_tokens(generated)))
    print("Current parameter:", machine.context.current_parameter)

    allowed = machine.get_allowed_tokens(generated)
    logits = model.get_logits_from_input_ids(prompt_tokens + generated)
    masked = ConstrainedDecoder.mask_logits(logits, allowed)

    scored = sorted(
        ((masked[token_id], model.decode([token_id])) for token_id in allowed),
        reverse=True,
    )

    print(f"\n{len(allowed)} allowed tokens for the first character of the value:")
    for logit, text in scored[:15]:
        print(f"  logit={logit:>8.3f}  text={text!r}")

    minus_candidates = [
        (logit, text) for logit, text in scored if "-" in text or "\u2212" in text
    ]

    print(f"\n{len(minus_candidates)} allowed candidate(s) containing a minus sign:")
    for logit, text in minus_candidates:
        rank = scored.index((logit, text)) + 1
        print(f"  rank={rank:>3}  logit={logit:>8.3f}  text={text!r}")

    if not minus_candidates:
        print("  (none -- the grammar never offered any negative-number token)")

    print("\n--- What the tokenizer actually does with negative numbers ---")
    for sample in ("-2", " -2", "-2.5", '"a": -2', "-"):
        ids = model.encode(sample)[0].tolist()
        pieces = [model.decode([i]) for i in ids]
        print(f"  encode({sample!r}) -> {pieces}")


if __name__ == "__main__":
    main()
