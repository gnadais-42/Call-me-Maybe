import json
from typing import Any

from .context import TokenCache
from .decoder import ConstrainedDecoder
from .errors import GenerationError
from .models import FunctionDefinition
from .prompt import build_function_selection_prompt
from .state_machine import StateMachine
from llm_sdk import Small_LLM_Model

# Safety cap so a grammar bug (or a genuinely pathological case) can never
# hang the process forever -- a real function call should need nowhere
# near this many generated tokens.
MAX_GENERATED_TOKENS = 512


def generate_function_call(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
    user_prompt: str,
    token_cache: TokenCache | None = None,
) -> dict[str, Any]:
    """Run constrained decoding to produce one function call for one prompt.

    This is the generation pipeline described in the spec (V.3.2/V.3.3),
    applied at every step:
      1. Ask the state machine which token ids are currently legal.
      2. If more than one is legal, ask the model for logits over its
         whole vocabulary, mask them down to only the legal ids, and pick
         the highest-scoring legal token (argmax). If only one token is
         legal, skip the model call entirely and use it directly -- the
         grammar has already made the decision, so there's nothing for
         the model to decide.
      3. Append the chosen token, tell the state machine to update, and
         repeat.
    The loop stops the moment the state machine has nothing left it
    permits -- at that point the generated tokens form one complete,
    schema-valid JSON object by construction, not by luck.

    Args:
        model: The wrapped LLM used both for the initial context and for
            constrained decoding.
        functions: The available function definitions.
        user_prompt: The natural language request to translate.
        token_cache: Optional pre-built vocabulary cache (see
            context.build_token_cache). Callers processing multiple
            prompts should build this once and pass it to every call,
            instead of paying the full vocabulary-decoding cost per
            prompt.

    Returns:
        A dict with "name" and "parameters" keys.

    Raises:
        GenerationError: If generation exceeds the safety cap, or (should
            be unreachable given a correct grammar) the result fails to
            parse as JSON.
    """
    context_prompt = build_function_selection_prompt(user_prompt, functions)
    prompt_tokens: list[int] = model.encode(context_prompt)[0].tolist()

    machine = StateMachine(model, functions, token_cache)
    generated_tokens: list[int] = []

    for _ in range(MAX_GENERATED_TOKENS):
        allowed_tokens = machine.get_allowed_tokens(generated_tokens)

        if not allowed_tokens:
            break

        if len(allowed_tokens) == 1:
            # The grammar leaves no real choice here (forced literal text,
            # e.g. mid-way through `{\n  "name": "`) -- skip the model
            # forward pass entirely rather than spend a full inference
            # step confirming an answer that was never in question. Since
            # masking would force this exact token regardless of the
            # model's actual logits, this changes nothing about the
            # result, only how many times the model is actually queried.
            next_token = allowed_tokens[0]
        else:
            logits = model.get_logits_from_input_ids(prompt_tokens + generated_tokens)
            masked_logits = ConstrainedDecoder.mask_logits(logits, allowed_tokens)
            next_token = max(range(len(masked_logits)), key=lambda i: masked_logits[i])

        generated_tokens.append(next_token)
        machine.update(generated_tokens)
    else:
        raise GenerationError(
            f"Generation for prompt {user_prompt!r} did not terminate within "
            f"{MAX_GENERATED_TOKENS} tokens."
        )

    raw_json = machine.context.decode_tokens(generated_tokens)

    try:
        parsed: dict[str, Any] = json.loads(raw_json)
    except json.JSONDecodeError as error:
        # Should be unreachable: constrained decoding guarantees valid
        # JSON at every step. Kept as a defensive check, not a real
        # control-flow path.
        raise GenerationError(
            f"Constrained decoding produced invalid JSON for prompt "
            f"{user_prompt!r}: {error}\nRaw output: {raw_json}"
        ) from error

    return parsed
