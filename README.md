*This project has been created as part of the 42 curriculum by <login1>.*

# call me maybe — Function calling via constrained decoding

## Description

This project turns natural language requests ("What is the sum of 2 and 3?")
into structured function calls (`{"name": "fn_add_numbers", "parameters":
{"a": 2, "b": 3}}`) using a small local LLM (Qwen/Qwen3-0.6B). Rather than
hoping the model produces valid JSON when prompted, generation is driven
token-by-token through a grammar-aware state machine that restricts, at
every step, which tokens the model is even allowed to choose from. The
result is 100% syntactically and schema-valid JSON, regardless of how
reliable the underlying model would be left to its own devices.

## Instructions

```bash
# 1. Copy the school-provided llm_sdk package into the project root,
#    next to src/ (not committed to this repository).
cp -r /path/to/llm_sdk .

# 2. Install dependencies.
make install        # runs `uv sync`

# 3. Run against the example data in data/input/.
make run            # runs `uv run python -m src`

# Or with explicit paths:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Other Makefile targets: `make debug` (runs under pdb), `make lint` /
`make lint-strict` (flake8 + mypy), `make clean` (removes caches and
`data/output/`).

## Algorithm explanation

Generation is driven by a `StateMachine` (`src/state_machine.py`) whose
current `State` (`src/states/`) decides, at every single generated token,
exactly which token ids are grammatically legal right now
(`get_allowed_tokens`) and whether the current grammar segment is finished
(`update`). The main loop (`src/generation.py`) repeats:

1. Ask the current state which token ids are legal.
2. Ask the model (`get_logits_from_input_ids`) for logits over its entire
   vocabulary.
3. Mask every illegal token's logit to `-inf` (`ConstrainedDecoder`).
4. Pick the highest-scoring legal token (argmax over the masked logits).
5. Append it, let the state machine advance, and repeat.

The loop stops the moment the current state has nothing left it permits —
at that point the generated tokens already form one complete, valid JSON
object, by construction rather than by luck.

States fall into three categories:

- **Fixed text** (`StartState`, `SuffixState`, `ParameterPrefixState`,
  `ParameterValuePrefixState`, the loop-back/close branch of
  `ParameterEndState`): the surrounding JSON punctuation never varies, so
  exactly one token id is legal at each position — the next token of a
  pre-encoded literal string.
- **Closed vocabulary via `TokenTrie`** (`FunctionNameState`,
  `ParameterNameState`, `BooleanState`): the model must choose among a
  small, known set of strings (function names, remaining parameter names,
  `true`/`false`). The trie restricts legal continuations to exactly the
  paths of valid entries; the model's own logits, masked to those paths,
  decide which one is actually taken.
- **Grammar-checked free text** (`NumberState`, `StringState`): there's no
  small fixed set of valid numbers or strings, so instead, every token in
  the vocabulary is checked individually: would appending its text keep
  the value a valid JSON number/string *prefix*? Only tokens that pass are
  allowed.

A `DecoderContext` (`src/context.py`) is shared across every state for one
prompt: it caches every vocabulary token's decoded text up front (so
states don't call `decode` per candidate at every step), and tracks which
function was selected and which parameters have already been generated,
so the grammar knows whether to loop back for another parameter or close
the object.

Choosing *which* function and *which* argument values, however, is still
entirely the model's decision: the trie/prefix-check machinery only ever
restricts the token *set* on offer; masked logits are what the model
actually picks from. A `build_function_selection_prompt` (`src/prompt.py`)
gives the model natural-language context (the user's request, plus every
candidate function's name/description/parameter types) before generation
starts, so it has something to reason about at each of those decision
points.

## Design decisions

- **One `DecoderContext`/`StateMachine` per prompt.** Function selection
  and parameter bookkeeping (`selected_function`, `generated_parameters`)
  are inherently per-request state; creating them fresh avoids any
  cross-prompt leakage.
- **Pydantic only for actual data models** (`FunctionDefinition`,
  `Parameter`, `ReturnType`, `FunctionCall`). The `State` classes are
  behaviour, not data, so they aren't pydantic models — validating the
  grammar's internal bookkeeping wouldn't add anything a data model is
  meant to provide.
- **Numbers need an explicit "stop" option; strings don't.** A string has
  a natural terminator (the closing quote), so "stop" can simply be one
  of the normally-offered tokens. A JSON number has no such character —
  `"4"` is already a complete, valid number, but so is `"42"` — so
  `NumberState` explicitly offers the first token of whatever comes next
  (a comma or a closing brace) as one more legal choice once the value is
  already complete, and lets the model's logits decide whether to keep
  extending it or stop. See "Challenges faced" below.
- **`get_parameter_end_text` on `DecoderContext`.** Both `NumberState`
  (to preview the "stop" option) and `ParameterEndState` (to actually
  force it) need to compute the same "what comes after this parameter"
  text. Factoring it out keeps that logic in one place instead of
  duplicating the remaining-parameters check.
- **Boolean support was added.** The provided spec mentions boolean as a
  possible argument type, but no `BooleanState` existed. Since `true` and
  `false` are a closed two-entry set, it reuses the same `TokenTrie`
  pattern as function/parameter name selection rather than inventing a
  new mechanism.

## Performance and reliability

- **100% valid JSON is structural, not probabilistic.** Every token
  emitted is one the grammar explicitly allowed; there is no path through
  the state machine that reaches an unparseable result. `generate_function_call`
  still wraps the final `json.loads` in a defensive check (raising
  `GenerationError` rather than crashing) in case of an unexpected
  interaction with tokenizer edge cases, but this should be unreachable
  in practice.
- **Skipping the model call for forced tokens is what actually mattered
  for speed.** The first hypothesis tried was that `DecoderContext`'s
  per-prompt vocabulary-decoding cache (see `build_token_cache`) was the
  bottleneck, since it was originally rebuilt from scratch for every
  prompt; sharing it across prompts turned out to change runtime by
  essentially nothing (measured: 1:54 before, 1:54 after, on 5 prompts
  against the real Qwen3-0.6B model). The actual cost was the number of
  *model forward passes*: most of a typical output is grammar-forced
  literal JSON punctuation (`{\n  "name": "`, `",\n  "parameters": {\n`,
  the parameter prefixes, the closing braces...) where `get_allowed_tokens`
  returns exactly one legal token id -- there was never a real decision
  for the model to make, yet the original loop queried a full forward
  pass anyway, every single step. `generate_function_call` now skips the
  model call whenever exactly one token is legal and appends it directly;
  masking would have forced that exact token regardless of the model's
  actual logits, so this changes nothing about the result, only how many
  times the model is actually invoked. `tests/test_integration.py::test_forced_tokens_skip_the_model_call`
  asserts this directly, by spying on the number of calls made.
- **A `MAX_GENERATED_TOKENS` safety cap** (`src/generation.py`) guarantees
  the process can never hang indefinitely even if a grammar edge case were
  found later; it raises `GenerationError`, which is caught per-prompt so
  one bad prompt doesn't abort the whole batch.
- **Function selection accuracy** depends entirely on the underlying
  model's ability to match the natural language request to the right
  function name/parameters given the context in `build_function_selection_prompt`
  — constrained decoding guarantees the *shape* of the output, not its
  semantic correctness. See "Testing strategy" for how the two concerns
  are tested separately.
- **Measured against the real model:** all 5 example prompts produced
  correct, valid output (`fn_add_numbers` with correctly extracted
  multi-digit arguments, `fn_greet`, `fn_reverse_string`) in under 2
  minutes total, comfortably inside the "under 5 minutes" requirement.

## Challenges faced

- **Numbers have no terminator.** The most subtle bug in this project:
  `"4"` is already a complete JSON number, so naively transitioning out of
  `NumberState` the instant the value becomes valid would stop every
  number after its first digit. Fixed by offering the "what comes next"
  boundary token as an explicit extra choice once the value is already
  complete, and letting the model choose between continuing and stopping.
- **Missing opening quote for string values.** `ParameterValuePrefixState`
  originally forced `'": '` regardless of parameter type and handed off to
  `StringState`, which only ever emits a *closing* quote — string values
  came out with a single, unbalanced quote and invalid JSON. Fixed by
  making the forced prefix type-aware: `'": "'` (with the opening quote)
  specifically for string parameters.
- **Invalid escape sequences weren't actually rejected.** The original
  escape-validity check inspected the *already-concatenated* candidate
  string's trailing backslash count, which "resolves" the instant any
  single character is appended after a backslash — so `\x` (not a legal
  JSON escape) was never actually filtered out. Fixed by scanning the
  whole candidate string from the start instead of only its tail.
- **A missing case in the exponent grammar.** `_is_valid_prefix` rejected
  perfectly valid numbers like `"1e+5"`, because the sign-handling branch
  for scientific notation never accounted for digits following the sign.
  Caught by a small table-driven unit test before it ever reached the
  model.
- **Known limitation, not fixed:** if one function's name is a strict
  prefix of another's, `FunctionNameState` commits to the shorter name the
  instant it becomes reachable-terminal, even if a longer valid name is
  still reachable in the trie. Not an issue for the provided example
  function sets, but a stricter implementation would need to keep both
  options open until the trie actually forks or dead-ends.
- **A wrong first guess at the performance bottleneck.** Measured against
  the real model, 5 prompts took 1:54. The first hypothesis was that
  `DecoderContext`'s per-prompt vocabulary-decode cache was the cause, so
  it was refactored to build once per process and be shared across
  prompts. Runtime didn't change at all (1:54, ±0.2s). Re-examining the
  generation loop instead of the cache revealed the real cost: a full
  model forward pass was being spent on every single forced literal
  token, even when the grammar already left only one legal choice.
  Skipping the model call in that case (kept behaviourally identical,
  since masking would have forced that same token anyway) is what
  actually mattered. Worth remembering: a plausible-sounding hypothesis
  about where time goes should still be measured, not assumed.

## Testing strategy

Testing is split by what actually needs the real model versus what
doesn't:

- **Pure grammar logic** (`tests/test_number_state.py`,
  `tests/test_string_state.py`, `tests/test_trie.py`,
  `tests/test_decoder.py`) tests static, model-independent logic directly:
  which strings are valid JSON number/string prefixes, trie traversal, and
  logit masking. These caught the exponent-sign and escape-validation bugs
  above without ever touching a model.
- **End-to-end grammar tests** (`tests/test_integration.py`) exercise the
  full `generate_function_call` pipeline against a small, deterministic
  fake `llm_sdk` (`tests/stubs/llm_sdk/`, test-only, not part of the
  submission): a character-level tokenizer with a mild logit bias toward
  closing delimiters, so string/number generation terminates
  deterministically instead of running forever on an untrained stub. These
  confirm the *mechanics* (valid JSON, correct keys, correct types, zero-
  and multi-parameter functions, all three value types) independently of
  whether a real model would pick semantically correct values. One test
  (`test_forced_tokens_skip_the_model_call`) spies on the number of model
  calls made and asserts it stays small, as a regression guard for the
  forced-token optimization described above.
- **Manual driving scripts** (not included) were used during development
  to print the token-by-token state transitions for a given prompt/model
  pair — the fastest way to catch a grammar bug without spinning up the
  real 0.6B model each time.
- **Not yet covered by automated tests:** semantic accuracy of function
  selection and argument extraction against the real Qwen3-0.6B model —
  that requires the actual model weights and is validated by running
  `make run` against `data/input/` and inspecting
  `data/output/function_calling_results.json` by hand.

## Example usage

```bash
$ make run
Wrote 5 function call(s) to data/output/function_calling_results.json
```

Given `data/input/function_calling_tests.json` containing:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

`data/output/function_calling_results.json` will contain:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Resources

- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- Background reading on constrained/structured generation for LLMs (grammar-
  and schema-constrained decoding, e.g. the general approach used by
  libraries like Outlines and llama.cpp's GBNF grammars) — used only for
  conceptual background, no external decoding library was used or allowed
  in this project.

**How AI was used:** Claude (Anthropic) was used throughout this project
as a collaborative reviewer and pair-programmer: identifying and
explaining specific bugs in the state machine logic (the missing
parameter-loop/close logic in `ParameterEndState`, the number-termination
bug in `NumberState`, the missing string opening quote in
`ParameterValuePrefixState`, and the escape-validation and exponent-sign
bugs), suggesting the `TokenTrie`-based approach for closed-vocabulary
choices (function/parameter names, booleans), and helping design the test
suite (including the deterministic fake `llm_sdk` stub used for
model-independent testing). Every change was reviewed, tested, and
understood before being kept — see "Challenges faced" above for the
specific bugs this process caught, several of which only surfaced once
actual tests were run rather than by inspection alone.
