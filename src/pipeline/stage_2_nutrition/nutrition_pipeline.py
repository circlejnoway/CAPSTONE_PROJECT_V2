from src.models.schemas import (ProductItem, ProductNutrition,
                                 ReceiptEntity, NutritionResult,
                                 NutrientData)
from .abbreviation_handler import AbbreviationHandler
from .portion_estimator import PortionEstimator
from .product_matcher import ProductMatcher
from .api_client import APIClientFactory
from .data_validator import DataValidator
import logging

logger = logging.getLogger(__name__)

class NutritionPipeline:
    def __init__(self, config):
        self.config = config
        self.abbreviation_handler = AbbreviationHandler(config=config)
        self.portion_estimator = PortionEstimator()
        self.product_matcher = ProductMatcher(config)
        self.api_clients = APIClientFactory.create_clients(config)
        self.data_validator = DataValidator()

    def _scale_nutrients(self, nutrients: NutrientData,
                         quantity: float, unit: str) -> dict:
        factor = quantity / 100.0
        return {
            "energy_kcal": (nutrients.energy_kcal_100g or 0) * factor,
            "proteins_g": (nutrients.proteins_100g or 0) * factor,
            "fat_g": (nutrients.fat_100g or 0) * factor,
            "carbohydrates_g": (nutrients.carbohydrates_100g or 0) * factor,
        }

    def process_item(self, item: ProductItem) -> ProductNutrition:
        warnings = []

        # 1. Expand abbreviations
        try:
            expanded, was_ai = self.abbreviation_handler.expand(item.product)
            if was_ai:
                warnings.append("abbreviation_expanded_by_ai")
        except Exception:
            expanded = item.product

        # 2. Estimate portion if missing
        quantity, unit, is_estimated = self.portion_estimator.estimate(
            expanded, item.quantity, item.unit
        )
        if is_estimated:
            warnings.append("portion_estimated")

        # 3. Try each API source
        raw_nutrition = None
        source_used = None
        for client in self.api_clients:
            raw_nutrition = client.search(expanded, quantity, unit)
            if raw_nutrition:
                source_used = type(client).__name__
                break

        # 4. Validate or fall back
        if raw_nutrition:
            nutrient_data, valid = self.data_validator.validate_nutrition(
                raw_nutrition, source_used
            )
            if not valid:
                nutrient_data = self.data_validator.handle_missing_data(
                    expanded) or NutrientData()
                source_used = "estimated"
        else:
            nutrient_data = self.data_validator.handle_missing_data(
                expanded) or NutrientData()
            source_used = "estimated"
            warnings.append("nutrition_not_found")

        # 5. Scale nutrients
        scaled = self._scale_nutrients(nutrient_data, quantity, unit)

        # 6. Always return a valid object
        return ProductNutrition(
            product_original=item.product,
            product_expanded=expanded,
            quantity=quantity,
            unit=unit,
            source=source_used or "estimated",
            nutrients_per100=nutrient_data,
            scaled_nutrients=scaled,
            is_estimated=(source_used == "estimated"),
            warnings=[w for w in warnings if w],
        )

    def process_receipt(self, receipt: ReceiptEntity) -> NutritionResult:
        items, missing, warnings = [], [], []

        for product_item in receipt.items:
            try:
                nutrition = self.process_item(product_item)
                if nutrition is None:
                    missing.append(product_item.product)
                    continue
                items.append(nutrition)
                if nutrition.is_estimated:
                    missing.append(product_item.product)
            except Exception as e:
                logger.error(f"Failed: {product_item.product}: {e}")
                missing.append(product_item.product)

        totals = {}
        for item in items:
            if item is None:
                continue
            for key, val in item.scaled_nutrients.items():
                totals[key] = totals.get(key, 0) + val

        return NutritionResult(
            items=items, totals=totals,
            warnings=warnings, missing_products=missing
        )