from pydantic import BaseModel


class Parameter(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: The JSON-schema-style type of the parameter, e.g. "string",
            "number", or "boolean".
    """

    type: str


class ReturnType(BaseModel):
    """Schema for a function's return value.

    Attributes:
        type: The JSON-schema-style type of the return value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """A single callable function, as described in functions_definition.json.

    Attributes:
        name: The function's identifier, e.g. "fn_add_numbers".
        description: A human-readable description used to help the model
            pick the right function for a given prompt.
        parameters: Mapping of parameter name to its schema.
        returns: The function's return type schema.
    """

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class FunctionCall(BaseModel):
    """A single resolved function call, ready to be written to the output file.

    Attributes:
        prompt: The original natural language request.
        name: The name of the function chosen by the model.
        parameters: The argument values generated for that function.
    """

    prompt: str
    name: str
    parameters: dict[str, object]
