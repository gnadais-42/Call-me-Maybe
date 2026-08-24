class TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.is_terminal: bool = False


class TokenTrie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, tokens: list[int]) -> None:
        node = self.root

        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]

        node.is_terminal = True

    def get_next_tokens(self, tokens: list[int]) -> list[int]:
        node = self.root

        for token in tokens:
            if token not in node.children:
                return []
            node = node.children[token]

        return list(node.children.keys())

    def is_terminal(self, tokens: list[int]) -> bool:
        node = self.root

        for token in tokens:
            if token not in node.children:
                return False

            node = node.children[token]

        return node.is_terminal