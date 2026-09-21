import math


class ConstrainedDecoder:
    """Applies grammar constraints to raw model logits.

    This is the actual "constrained decoding" step described in the
    project: given the logits the model produced for every token in its
    vocabulary, and the (much smaller) set of token ids the current
    grammar state allows, every disallowed token's logit is set to
    negative infinity so it can never be selected by an argmax (or any
    other sampling strategy applied afterwards).
    """

    @staticmethod
    def mask_logits(
        logits: list[float],
        allowed_tokens: list[int],
    ) -> list[float]:
        """Return a copy of logits with every disallowed entry set to -inf.

        Args:
            logits: Raw logits, one per vocabulary token id (index == id).
            allowed_tokens: Token ids permitted as the next token.

        Returns:
            A new list the same length as logits, safe to argmax over.
        """
        allowed = set(allowed_tokens)

        masked: list[float] = []

        for token_id, logit in enumerate(logits):
            masked.append(logit if token_id in allowed else -math.inf)

        return masked
