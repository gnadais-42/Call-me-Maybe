import math


class ConstrainedDecoder:
    @staticmethod
    def mask_logits(
        logits: list[float],
        allowed_tokens: list[int],
    ) -> list[float]:
        allowed = set(allowed_tokens)

        masked: list[float] = []

        for token_id, logit in enumerate(logits):
            masked.append(logit if token_id in allowed else -math.inf)
        return masked