"""
nutrition_retrieval.py
----------------------
Retrieve nutritional information for products listed on a receipt.

Uses:
- Open Food Facts API (primary source, filtered for Qatar)
- OpenAI as a fallback when no product is found
- Abbreviation expansion & fuzzy matching to handle receipt shorthand
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import openai
import requests
from rapidfuzz import fuzz, process

# openfoodfacts Python package (you already have it installed)
import openfoodfacts

# Load environment variables (e.g., OPENAI_API_KEY)
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
OPENFOODFACTS_COUNTRY = "qatar"           # "world" for global search
SEARCH_PAGE_SIZE = 5                      # how many results to fuzzy-match against
MIN_FUZZY_SCORE = 75                      # minimum similarity to accept a match

# If you don't have an OpenAI key, the fallback will be skipped.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------------------------------------------------
# Abbreviation Handling and Expansion
# ----------------------------------------------------------------------
def load_abbreviations(filepath: str = "data/abbreviations.json") -> Dict[str, str]:
    """Load abbreviations JSON; supports flat dict or nested under 'abbr_map'."""
    import os
    if not os.path.exists(filepath):
        logging.warning(f"Abbreviations file not found at {filepath}. Using empty mapping.")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If the file is a flat dictionary already, use it directly
        if isinstance(data, dict) and "abbr_map" not in data:
            return data

        # If wrapped inside "abbr_map", extract that
        if isinstance(data, dict) and "abbr_map" in data and isinstance(data["abbr_map"], dict):
            return data["abbr_map"]

        logging.warning("Abbreviations file is not in the expected format. Returning empty mapping.")
        return {}
    except json.JSONDecodeError as e:
        logging.warning(f"Invalid JSON in {filepath}: {e}. Using empty mapping.")
        return {}
    
def expand_abbreviations(text: str, mapping: Dict[str, str]) -> str:
    lowered = text.lower().strip()
    # Try whole phrase first
    if lowered in mapping:
        return mapping[lowered]
    # Then word-by-word fallback
    words = lowered.split()
    expanded = [mapping.get(w, w) for w in words]
    return " ".join(expanded)


# ----------------------------------------------------------------------
# Receipt Line Parsing
# ----------------------------------------------------------------------
# Regex to extract quantity and unit.
# Supports: kg, g, l, ml, litre, liter, oz, lb, piece(s), pcs.
QTY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|litre|liter|oz|lb|piece|pcs|pc)\b",
    re.IGNORECASE,
)

# Unit normalisation map (to g or ml for easier scaling, if needed)
UNIT_NORMALIZE = {
    "kg": ("kg", 1000),   # we keep kg but note conversion if we want grams
    "g": "g",
    "l": "l",
    "ml": "ml",
    "litre": "l",
    "liter": "l",
    "oz": "oz",
    "lb": "lb",
    "piece": "piece",
    "pcs": "piece",
    "pc": "piece",
}


def parse_receipt_line(line: str) -> Dict[str, Optional[str]]:
    """
    Extract product name, quantity, and unit from a single receipt line.

    Returns:
        {
            "raw": ...,
            "product": str,       # cleaned product name
            "quantity": float | None,
            "unit": str | None
        }
    """
    line = line.strip()
    # Remove common price patterns (e.g., "@ 25.00", "25.00")
    # Keep only the part before any price indication.
    # A simple approach: split on " @ " and take the first part, or remove trailing numbers.
    product_part = line
    if " @ " in product_part:
        product_part = product_part.split(" @ ")[0]
    # Remove trailing currency amounts
    product_part = re.sub(r"\s+\d+\.?\d*\s*$", "", product_part)

    # Find quantity/unit in the product part
    qty_match = QTY_PATTERN.search(product_part)
    quantity = None
    unit = None
    product_name = product_part

    if qty_match:
        qty_str = qty_match.group(1)
        unit_raw = qty_match.group(2).lower()
        quantity = float(qty_str)
        unit_info = UNIT_NORMALIZE.get(unit_raw, unit_raw)
        if isinstance(unit_info, tuple):
            unit = unit_info[0]  # "kg"
        else:
            unit = unit_info
        # Remove the quantity part from product name
        product_name = product_part[: qty_match.start()] + product_part[qty_match.end() :]
        product_name = re.sub(r"\s+", " ", product_name).strip()

    # Remove leftover noise like double spaces, trailing punctuation
    product_name = re.sub(r"[^\w\s]", "", product_name).strip()

    return {
        "raw": line,
        "product": product_name,
        "quantity": quantity,
        "unit": unit,
    }


# ----------------------------------------------------------------------
# Nutritional Lookup using Open Food Facts
# ----------------------------------------------------------------------
def search_openfoodfacts(query: str, country: str = OPENFOODFACTS_COUNTRY) -> Optional[Dict]:
    """
    Search Open Food Facts by product name and return the first result
    that contains nutritional information.
    Works with openfoodfacts>=2.0,<3.0 (v5 in your case).
    """
    try:
        api = openfoodfacts.API(user_agent="MyCapstone/1.0")
        # Correct method for v2.x+ is product.text_search(query, ...)
        # but in some versions it's product.search(query, ...)
        # Let's try the correct parameter names for v5.x:
        # The text_search method in v5 accepts: query, page, page_size, sort_by,
        # and also "countries" (not "cc") as a filter. Actually, the parameter might be "countries".
        # I'll catch TypeError to fallback.
        try:
            results = api.product.text_search(
                query,
                page_size=SEARCH_PAGE_SIZE,
                countries=country,          # <-- "countries" not "cc"
            )
        except TypeError:
            # maybe older version, try without country filter
            results = api.product.text_search(query, page_size=SEARCH_PAGE_SIZE)
    except Exception as e:
        logging.error(f"Open Food Facts API error: {e}")
        return None

    if not results or "products" not in results:
        return None

    for product in results["products"]:
        if product.get("nutriments"):
            return {
                "product_name": product.get("product_name", query),
                "nutriments": product["nutriments"],
            }
    return None


def fuzzy_match_product(query: str, candidates: List[Dict]) -> Optional[Dict]:
    """
    Given a list of product dicts (from API), pick the one whose
    product_name best matches the query using token_sort_ratio.
    Returns the best match if score >= MIN_FUZZY_SCORE, else None.
    """
    if not candidates:
        return None

    names = [p.get("product_name", "") for p in candidates]
    # Remove empty names
    valid = [(p, n) for p, n in zip(candidates, names) if n]
    if not valid:
        return None

    best_match = process.extractOne(
        query,
        [n for _, n in valid],
        scorer=fuzz.token_sort_ratio,
        score_cutoff=MIN_FUZZY_SCORE,
    )
    if best_match:
        matched_name, score, idx = best_match
        return valid[idx][0]
    return None


def get_nutrition_openfoodfacts(product_name: str) -> Optional[Dict]:
    # First try a direct search
    result = search_openfoodfacts(product_name)
    if result:
        return result["nutriments"]

    # If no direct hit, broader fuzzy match
    api = openfoodfacts.API(user_agent="MyCapstone/1.0")
    try:
        try:
            broad_results = api.product.text_search(
                product_name,
                page_size=SEARCH_PAGE_SIZE * 2,
                countries="world",           # <-- "countries"
            )
        except TypeError:
            broad_results = api.product.text_search(product_name, page_size=SEARCH_PAGE_SIZE * 2)
    except Exception as e:
        logging.error(f"Open Food Facts broad search error: {e}")
        return None
    # … rest of fuzzy matching remains the same


# ----------------------------------------------------------------------
# Fallback via OpenAI
# ----------------------------------------------------------------------
OPENAI_SYSTEM_PROMPT = (
    "You are a nutritional database. For a given food product, return a JSON object "
    "with typical nutritional values per 100 grams (or per 100 ml for liquids). "
    "Include only these keys: energy-kcal_100g, fat_100g, saturated-fat_100g, "
    "carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g. "
    "Use standard units. If the product is unknown, return an empty JSON object {}."
)


def get_nutrition_openai(product_name: str) -> Optional[Dict]:
    """
    Use OpenAI's ChatCompletion to generate nutritional data per 100g/ml.
    Requires OPENAI_API_KEY to be set.
    """
    if not OPENAI_API_KEY:
        logging.warning("OpenAI API key not set – skipping fallback.")
        return None

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": f"Product: {product_name}"},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Parse the JSON; sometimes it’s inside a markdown code block
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            # Rename keys to match Open Food Facts style (for consistency)
            return {
                "energy-kcal_100g": data.get("energy-kcal_100g"),
                "fat_100g": data.get("fat_100g"),
                "saturated-fat_100g": data.get("saturated-fat_100g"),
                "carbohydrates_100g": data.get("carbohydrates_100g"),
                "sugars_100g": data.get("sugars_100g"),
                "fiber_100g": data.get("fiber_100g"),
                "proteins_100g": data.get("proteins_100g"),
                "salt_100g": data.get("salt_100g"),
            }
        else:
            logging.warning(f"OpenAI response not valid JSON: {content}")
    except Exception as e:
        logging.error(f"OpenAI fallback error: {e}")
    return None


# ----------------------------------------------------------------------
# Main Nutrition Retrieval for a Single Product
# ----------------------------------------------------------------------
def get_product_nutrition(product_name: str) -> Optional[Dict]:
    """
    Combined pipeline:
    1. Try Open Food Facts (primary).
    2. If not found, fallback to OpenAI.
    Returns a dictionary of nutrients per 100g/ml, or None.
    """
    nutrients = get_nutrition_openfoodfacts(product_name)
    if nutrients:
        logging.info(f"Found in Open Food Facts: {product_name}")
        return nutrients

    logging.info(f"Falling back to OpenAI for: {product_name}")
    return get_nutrition_openai(product_name)


# ----------------------------------------------------------------------
# Aggregation Across a Full Receipt
# ----------------------------------------------------------------------
def scale_nutrients(nutrients: Dict, quantity: float, unit: str) -> Dict[str, float]:
    """
    Convert nutrient values (per 100g or per 100ml) to actual amounts
    based on the quantity and unit.

    For solid items: assumes 'g', 'kg' → scale to grams.
    For liquids: assumes 'ml', 'l' → scale to millilitres.
    For 'piece': we assume an average weight of 100g (can be customised).
    """
    # Conversion factors to grams or millilitres
    if unit in ("g", "ml"):
        factor = quantity / 100.0
    elif unit in ("kg",):
        factor = quantity * 1000 / 100.0
    elif unit in ("l",):
        factor = quantity * 1000 / 100.0
    elif unit in ("oz",):
        factor = quantity * 28.3495 / 100.0
    elif unit in ("lb",):
        factor = quantity * 453.592 / 100.0
    elif unit == "piece":
        factor = quantity  # assume 100g per piece
    else:
        factor = quantity / 100.0  # fallback

    scaled = {}
    for k, v in nutrients.items():
        if isinstance(v, (int, float)):
            scaled[k] = v * factor
        else:
            scaled[k] = v  # keep non-numeric fields unchanged
    return scaled


def calculate_total_nutrition(
    receipt_lines: List[str],
    abbreviations: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Main entry point: given a list of raw receipt lines,
    return:
        - items: list of per-item details (name, quantity, unit, nutrients, scaled_nutrients)
        - totals: total nutritional values across all items
        - warnings: list of items that could not be found
    """
    if abbreviations is None:
        abbreviations = load_abbreviations()

    total_nutrients = {
        "energy-kcal": 0,
        "fat_g": 0,
        "saturated-fat_g": 0,
        "carbohydrates_g": 0,
        "sugars_g": 0,
        "fiber_g": 0,
        "proteins_g": 0,
        "salt_g": 0,
    }

    items = []
    warnings = []

    for line in receipt_lines:
        parsed = parse_receipt_line(line)
        product_raw = parsed["product"]
        if not product_raw:
            continue

        # Expand abbreviations
        product_expanded = expand_abbreviations(product_raw, abbreviations)
        logging.info(f"Processing: '{product_expanded}' (from '{product_raw}')")

        # Get per-100g nutrients
        nutrients_per100 = get_product_nutrition(product_expanded)

        scaled = None
        if nutrients_per100 and parsed["quantity"] is not None and parsed["unit"] is not None:
            scaled = scale_nutrients(nutrients_per100, parsed["quantity"], parsed["unit"])
            # Accumulate totals (map keys to standard names)
            total_nutrients["energy-kcal"] += scaled.get("energy-kcal_100g", 0)
            total_nutrients["fat_g"] += scaled.get("fat_100g", 0)
            total_nutrients["saturated-fat_g"] += scaled.get("saturated-fat_100g", 0)
            total_nutrients["carbohydrates_g"] += scaled.get("carbohydrates_100g", 0)
            total_nutrients["sugars_g"] += scaled.get("sugars_100g", 0)
            total_nutrients["fiber_g"] += scaled.get("fiber_100g", 0)
            total_nutrients["proteins_g"] += scaled.get("proteins_100g", 0)
            total_nutrients["salt_g"] += scaled.get("salt_100g", 0)
        elif nutrients_per100:
            # No quantity found; just record per-100g values, no totals
            scaled = nutrients_per100
            warnings.append(
                f"No quantity for '{product_expanded}' – using per 100g/ml values without scaling."
            )
        else:
            warnings.append(f"No nutritional data found for '{product_expanded}'.")

        items.append(
            {
                "raw": parsed["raw"],
                "product_original": product_raw,
                "product_expanded": product_expanded,
                "quantity": parsed["quantity"],
                "unit": parsed["unit"],
                "nutrients_per100": nutrients_per100,
                "scaled_nutrients": scaled,
            }
        )

    return {"items": items, "totals": total_nutrients, "warnings": warnings}


# ----------------------------------------------------------------------
# Example usage (can be tested standalone)
# ----------------------------------------------------------------------
if __name__ == "__main__":

    path = "data/abbreviations.json"
    print("Looking for:", os.path.abspath(path))
    print("Exists:", os.path.exists(path))
    if os.path.exists(path):
        with open(path, "rb") as f:
            raw = f.read()[:100]
            print("First 100 bytes (repr):", repr(raw))
            
    # Sample receipt text (extracted by OCR)
    sample_lines = [
        "CHK BRST 1.5KG @ 45.00",
        "WHL WHT BREAD 1PC @ 5.50",
        "TOM 0.5KG @ 3.00",
    ]

    result = calculate_total_nutrition(sample_lines)

    print(json.dumps(result["totals"], indent=2))
    for warn in result["warnings"]:
        print(f"WARNING: {warn}")