from typing import Optional

class PortionEstimator:
    def __init__(self):
        self.default_portions = {
            "banana": {"quantity": 1, "unit": "piece", "weight_g": 120},
            "apple": {"quantity": 1, "unit": "piece", "weight_g": 182},
            "egg": {"quantity": 1, "unit": "piece", "weight_g": 60},
            "yogurt": {"quantity": 1, "unit": "container", "weight_g": 200},
            "rice": {"quantity": 1, "unit": "cup", "weight_g": 158},
            "milk": {"quantity": 1, "unit": "cup", "weight_g": 244},
            "chicken": {"quantity": 1, "unit": "serving", "weight_g": 150},
            "bread": {"quantity": 2, "unit": "slice", "weight_g": 60},
        }

    def estimate(self, product: str, quantity: Optional[float],
                 unit: Optional[str]) -> tuple[float, str, bool]:
        # Already has both — no estimation needed
        if quantity is not None and unit is not None:
            return quantity, unit, False

        product_lower = product.lower()
        for key, portion in self.default_portions.items():
            if key in product_lower:
                return portion["quantity"], portion["unit"], True

        # Generic fallback
        return 1.0, "portion", True