from typing import Optional, Dict
from src.models.schemas import NutrientData
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    def validate_nutrition(self, nutrition: Dict,
                           source: str) -> tuple[NutrientData, bool]:
        try:
            data = NutrientData(
                energy_kcal_100g=nutrition.get("energy_kcal_100g"),
                proteins_100g=nutrition.get("proteins_100g"),
                fat_100g=nutrition.get("fat_100g"),
                carbohydrates_100g=nutrition.get("carbohydrates_100g"),
            )
            return data, True
        except Exception:
            logger.warning(f"Invalid data from {source}")
            return NutrientData(), False

    def handle_missing_data(self, product: str) -> Optional[NutrientData]:
        # Category-based fallback estimates
        p = product.lower()
        if "chicken" in p or "beef" in p or "pork" in p:
            return NutrientData(energy_kcal_100g=200, proteins_100g=25,
                                fat_100g=10, carbohydrates_100g=0)
        elif "milk" in p or "yogurt" in p:
            return NutrientData(energy_kcal_100g=60, proteins_100g=3,
                                fat_100g=3, carbohydrates_100g=5)
        elif "bread" in p or "rice" in p or "pasta" in p:
            return NutrientData(energy_kcal_100g=350, proteins_100g=8,
                                fat_100g=2, carbohydrates_100g=70)
        return None