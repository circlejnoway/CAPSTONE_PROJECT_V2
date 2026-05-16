from typing import List
from rapidfuzz import fuzz, process

class ProductMatcher:
    def __init__(self, config):
        self.min_score = config.min_fuzzy_score

    def find_best_match(self, product: str,
                        candidates: List[str]) -> tuple[str, float]:
        if not candidates:
            return product, 0.0

        matches = process.extract(
            product, candidates,
            limit=3,
            scorer=fuzz.token_sort_ratio
        )
        if matches and matches[0][1] > self.min_score:
            return matches[0][0], matches[0][1] / 100.0

        return product, 0.0