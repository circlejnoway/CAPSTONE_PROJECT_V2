from src.config.nutrition import NutritionConfig
from src.models.schemas import ReceiptEntity, ProductItem
from src.pipeline.stage_2_nutrition.nutrition_pipeline import NutritionPipeline

# Mock receipt — no OCR needed
mock_receipt = ReceiptEntity(
    merchant="Test Store",
    date="2025-01-01",
    total="20.00",
    items=[
        ProductItem(raw="CHKN BRST 500G", product="CHKN BRST",
                    quantity=500, unit="g"),
        ProductItem(raw="WHOLE MILK 1L", product="whole milk",
                    quantity=1000, unit="ml"),
        ProductItem(raw="BANANA", product="banana",
                    quantity=None, unit=None),  # tests portion estimator
    ]
)

config = NutritionConfig()
pipeline = NutritionPipeline(config)
result = pipeline.process_receipt(mock_receipt)

print("=== TOTALS ===")
for k, v in result.totals.items():
    print(f"  {k}: {v:.1f}")

print("\n=== ITEMS ===")
for item in result.items:
    print(f"  {item.product_original} → {item.product_expanded} "
          f"[{item.source}] warnings: {item.warnings}")

print("\n=== MISSING ===", result.missing_products)