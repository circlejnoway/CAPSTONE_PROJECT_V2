import json
import logging

logger = logging.getLogger(__name__)

class AbbreviationHandler:
    def __init__(self, abbr_file: str = "data/abbreviations.json", config=None):
        self.abbr_map = self._load_abbreviations(abbr_file)
        self.config = config
        self._ai_disabled = False

    def _load_abbreviations(self, path: str) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "chkn": "chicken", "brst": "breast",
                "whl": "whole", "mlk": "milk",
                "org": "organic", "frzn": "frozen",
                "veg": "vegetable", "frt": "fruit",
                "jc": "juice", "yog": "yogurt",
                "chz": "cheese", "bttr": "butter",
                "sg": "sugar", "flr": "flour",
            }

    def expand(self, text: str) -> tuple[str, bool]:
        expanded = self._expand_dictionary(text)

        # Only call AI if text looks abbreviated
        # (short words, all caps, or dictionary changed something)
        ai_enabled = (
            self.config and
            self.config.enable_ai_abbreviation_fallback and
            not self._ai_disabled and
            self._looks_abbreviated(text)  # ← add this guard
        )

        if ai_enabled:
            try:
                ai_expanded = self._expand_with_ai(expanded)
                if ai_expanded.lower() != text.lower():
                    return ai_expanded, True
            except Exception as e:
                logger.warning(f"Ollama fallback failed: {e}")
                self._ai_disabled = True

        return expanded, False

    def _looks_abbreviated(self, text: str) -> bool:
        """Only use AI if text seems to need it"""
        words = text.strip().split()
        # Trigger AI if: any word is all caps, or any word is 4 chars or less
        return any(w.isupper() or len(w) <= 4 for w in words)

    def _expand_dictionary(self, text: str) -> str:
        words = text.lower().strip().split()
        expanded = [self.abbr_map.get(w, w) for w in words]
        return " ".join(expanded)

    def _expand_with_ai(self, text: str) -> str:
        import ollama

        prompt = f"""You are a grocery receipt parser.
Expand this abbreviated food product name to its full, proper name.
Text: "{text}"
Rules:
- Return ONLY the expanded product name, nothing else
- No punctuation, no explanation
- Keep it simple and specific
Examples:
"chkn brst" -> "chicken breast"
"whl mlk" -> "whole milk"
"org ban" -> "organic banana"
"frzn veg" -> "frozen vegetables"
"chz slc" -> "cheese slices"
"""
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()