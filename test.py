from llm_sdk import Small_LLM_Model


model = Small_LLM_Model()

text = '",\n   "parameters": {\n'

tokens = model.encode(text)[0].tolist()

print(tokens)

for token in tokens:
    print(token, repr(model.decode([token])))