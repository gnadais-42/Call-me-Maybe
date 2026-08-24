from llm_sdk import Small_LLM_Model
from src.parser import load_function_definitions
from src.state_machine import StateMachine


model = Small_LLM_Model()

functions = load_function_definitions(
    "data/input/functions_definition.json"
)

machine = StateMachine(
    model,
    functions
)

generated = []

while True:
    machine.update(generated)

    allowed = machine.get_allowed_tokens(generated)

    if not allowed:
        break

    generated.append(allowed[0])

print(model.decode(generated))
print(machine.context.selected_function)