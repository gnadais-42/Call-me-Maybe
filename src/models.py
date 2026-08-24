from pydantic import BaseModel


class Parameter(BaseModel):
    type: str


class ReturnType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: ReturnType


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict
