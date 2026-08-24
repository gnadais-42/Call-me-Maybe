import json

from src.models import FunctionDefinition, FunctionCall


with open("data/input/functions_definition.json") as file:
    data = json.load(file)

functions = [
    FunctionDefinition.model_validate(item)
    for item in data
]

for function in functions:
    print(function)

call = FunctionCall(
    prompt="What is the sum of 2 and 3?",
    name="fn_add_numbers",
    parameters={
        "a": 2,
        "b": 3,
    },
)

print(call)

text = "fn_add_numbers"
