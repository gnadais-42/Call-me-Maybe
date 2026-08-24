from llm_sdk import Small_LLM_Model
from src.parser import load_function_definitions
from src.state_machine import DecoderState, StateMachine
from src.trie import TokenTrie


model = Small_LLM_Model()

functions = load_function_definitions(
    "data/input/functions_definition.json"
)

trie = TokenTrie()

for function in functions:
    tokens = model.encode(function.name)[0].tolist()
    trie.insert(tokens)

machine = StateMachine(
    model,
    trie,
    functions
)

generated = []

while True:
    machine.update_state(generated)

    allowed = machine.get_allowed_tokens(generated)

    if not allowed:
        break

    generated.append(allowed[0])

print(model.decode(generated))
print(machine.context.selected_function)