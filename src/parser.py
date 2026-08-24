import json

from .models import FunctionDefinition


def load_function_definitions(
    path: str,
) -> list[FunctionDefinition]:
    with open(path, "r") as file:
        data = json.load(file)

    functions: list[FunctionDefinition] = []

    for function in data:
        functions.append(FunctionDefinition(**function))

    return functions

