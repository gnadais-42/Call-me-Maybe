from .models import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the natural language context shown to the model before generation.

    Constrained decoding controls the *shape* of the output regardless of
    what the model would say unprompted -- but it can only restrict *which*
    of the grammatically legal tokens are chosen, it can't invent semantic
    understanding the model wasn't given. This prompt's only job is to give
    the model enough context (the user's request, plus every candidate
    function's name/description/parameters) to make a good choice at each
    of those decision points. The actual choice among legal tokens still
    comes from the model's own logits at generation time.

    Args:
        user_prompt: The end user's natural language request.
        functions: The available function definitions to choose from.

    Returns:
        A single prompt string ready to be tokenized.
    """
    lines = [
        "You are a function calling assistant.",
        "Given a user request, choose the single best matching function "
        "from the list below and provide correct arguments for it.",
        "",
        "Available functions:",
    ]

    for function in functions:
        params = ", ".join(
            f"{name}: {parameter.type}"
            for name, parameter in function.parameters.items()
        )
        lines.append(f"- {function.name}({params}): {function.description}")

    lines.extend(
        [
            "",
            f'User request: "{user_prompt}"',
            "",
            "Respond with a JSON object describing the function call.",
        ]
    )

    return "\n".join(lines)
