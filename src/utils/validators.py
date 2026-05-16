from typing import Any


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
