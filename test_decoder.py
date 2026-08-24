from llm_sdk import Small_LLM_Model

from src.decoder import ConstrainedDecoder
from src.trie import TokenTrie
from src.parser import load_function_definitions


model = Small_LLM_Model()

functions = load_function_definitions("data/input/functions_definition.json")

function_names = [
    function.name for function in functions
]

trie = TokenTrie()

for name in function_names:
    tokens = model.encode(name)[0].tolist()
    trie.insert(tokens)


prompt = "What is the sum of 2 and 3?"

prompt_tokens = model.encode(prompt)[0].tolist()
generated_tokens = []

print(f"Prompt: {prompt}")
print("\nGenerating function name...\n")


while not trie.is_terminal(generated_tokens):

    # The model needs to see the prompt AND everything generated so far.
    input_ids = prompt_tokens + generated_tokens

    # Get the model's scores for every possible next token.
    logits = model.get_logits_from_input_ids(input_ids)

    # Ask the trie which tokens are valid at this point.
    allowed_tokens = trie.get_next_tokens(generated_tokens)

    # Remove the logits for all forbidden tokens.
    masked_logits = ConstrainedDecoder.mask_logits(
        logits,
        allowed_tokens,
    )

    # Pick the highest-scoring allowed token.
    next_token_id = max(
        range(len(masked_logits)),
        key=lambda i: masked_logits[i]
    )

    generated_tokens.append(next_token_id)

    print(
        f"Token: {next_token_id:5d} "
        f"{model.decode([next_token_id])!r}"
    )


print("\nFinal function name:")
print(model.decode(generated_tokens))