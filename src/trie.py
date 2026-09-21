class TrieNode:
    """A single node in a TokenTrie."""

    def __init__(self) -> None:
        self.children: dict[int, "TrieNode"] = {}
        self.is_terminal: bool = False


class TokenTrie:
    """A trie over sequences of token ids.

    Used whenever generation must be restricted to a closed set of known
    strings (function names, parameter names, boolean literals): each
    candidate string is encoded to its token ids and inserted once, then
    `get_next_tokens` tells a state exactly which token ids are valid
    continuations of what has been generated so far.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, tokens: list[int]) -> None:
        """Insert one token sequence, marking its end as terminal."""
        node = self.root

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]

        node.is_terminal = True

    def get_next_tokens(self, tokens: list[int]) -> list[int]:
        """Return the valid next token ids after the given prefix.

        Returns an empty list if the prefix is not present in the trie at
        all (should not happen if the state machine only ever feeds back
        tokens it previously allowed).
        """
        node = self.root

        for token in tokens:
            if token not in node.children:
                return []
            node = node.children[token]

        return list(node.children.keys())

    def is_terminal(self, tokens: list[int]) -> bool:
        """Return True if the given token sequence is a complete entry."""
        node = self.root

        for token in tokens:
            if token not in node.children:
                return False

            node = node.children[token]

        return node.is_terminal
