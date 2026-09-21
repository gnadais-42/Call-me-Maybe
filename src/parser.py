import json
from pathlib import Path

from .errors import InputFileError
from .models import FunctionDefinition


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to a JSON file containing an array of function
            definitions (see functions_definition.json in the spec).

    Returns:
        The parsed and validated function definitions.

    Raises:
        InputFileError: If the file is missing, unreadable, not valid
            JSON, or does not match the expected schema.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise InputFileError(f"Function definitions file not found: {path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise InputFileError(
            f"Could not read function definitions from {path}: {error}"
        ) from error

    if not isinstance(data, list):
        raise InputFileError(
            f"Function definitions file {path} must contain a JSON array."
        )

    functions: list[FunctionDefinition] = []

    for index, entry in enumerate(data):
        try:
            functions.append(FunctionDefinition(**entry))
        except (TypeError, ValueError) as error:
            raise InputFileError(
                f"Invalid function definition at index {index} in {path}: {error}"
            ) from error

    if not functions:
        raise InputFileError(f"Function definitions file {path} is empty.")

    return functions


def load_prompts(path: str) -> list[str]:
    """Load natural language prompts from a JSON test file.

    Args:
        path: Path to a JSON file containing an array of
            `{"prompt": "..."}` objects (see function_calling_tests.json).

    Returns:
        A list of natural language prompt strings, in file order.

    Raises:
        InputFileError: If the file is missing, unreadable, not valid
            JSON, or does not match the expected schema.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise InputFileError(f"Prompts file not found: {path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise InputFileError(f"Could not read prompts from {path}: {error}") from error

    if not isinstance(data, list):
        raise InputFileError(f"Prompts file {path} must contain a JSON array.")

    prompts: list[str] = []

    for index, entry in enumerate(data):
        if not isinstance(entry, dict) or "prompt" not in entry:
            raise InputFileError(
                f"Entry at index {index} in {path} is missing a 'prompt' field."
            )
        prompts.append(str(entry["prompt"]))

    return prompts
