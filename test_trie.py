from llm_sdk import Small_LLM_Model

from src.trie import TokenTrie


model = Small_LLM_Model()

function_names = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
]

trie = TokenTrie()

for name in function_names:
    token_ids = model.encode(name)[0].tolist()
    trie.insert(token_ids)


print("Next tokens from root:")
print(trie.get_next_tokens([]))

print("\nNext tokens after 'fn':")
print(trie.get_next_tokens([8822]))

print("\nNext tokens after 'fn_add':")
print(trie.get_next_tokens([8822, 2891]))

print("\nIs fn_add_numbers complete?")
print(trie.is_terminal([8822, 2891, 32964]))

print("\nIs fn_add complete?")
print(trie.is_terminal([8822, 2891]))