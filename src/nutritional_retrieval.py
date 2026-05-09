"""
nutrition_retrieval.py
----------------------
Multi‑source nutritional retrieval pipeline for receipt items.

Free sources used (in order):
    1. USDA FoodData Central
    2. Open Food Facts
    3. API Ninjas (nutrition endpoint)
    4. Google Gemini (free LLM)
    5. Ollama (optional, local, unlimited)
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import requests
import openai  # kept for compatibility but not used unless OPENAI_API_KEY is set
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
import openfoodfacts

load_dotenv()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MIN_FUZZY_SCORE = 75
SEARCH_PAGE_SIZE = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# API keys from environment
USDA_API_KEY = os.getenv("USDA_API_KEY", "")
NINJAS_API_KEY = os.getenv("NINJAS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Optional: Ollama is local, no key needed

# ----------------------------------------------------------------------
# Abbreviation handling
# ----------------------------------------------------------------------
def load_abbreviations(filepath: str = "data/abbreviations.json") -> Dict[str, str]:
    if not os.path.exists(filepath):
        logging.warning(f"Abbreviations file not found at {filepath}. Using empty mapping.")
        return {}
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "abbr_map" in data:
            return data["abbr_map"]
        elif isinstance(data, dict):
            return data
        else:
            logging.warning("Unexpected format – using empty mapping")
            return {}
    except Exception as e:
        logging.warning(f"Error loading abbreviations: {e}")
        return {}

def expand_abbreviations(text: str, mapping: Dict[str, str]) -> str:
    lowered = text.lower().strip()
    if lowered in mapping:
        return mapping[lowered]
    words = lowered.split()
    expanded = [mapping.get(w, w) for w in words]
    return " ".join(expanded)

# ----------------------------------------------------------------------
# Receipt line parsing
# ----------------------------------------------------------------------
QTY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|litre|liter|oz|lb|piece|pcs|pc)\b",
    re.IGNORECASE,
)

UNIT_NORMALIZE = {
    "kg": ("kg", 1000),
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
    line = line.strip()
    product_part = line
    if " @ " in product_part:
        product_part = product_part.split(" @ ")[0]
    product_part = re.sub(r"\s+\d+\.?\d*\s*$", "", product_part)

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
            unit = unit_info[0]
        else:
            unit = unit_info
        product_name = product_part[: qty_match.start()] + product_part[qty_match.end() :]
        product_name = re.sub(r"\s+", " ", product_name).strip()

    product_name = re.sub(r"[^\w\s]", "", product_name).strip()
    return {
        "raw": line,
        "product": product_name,
        "quantity": quantity,
        "unit": unit,
    }

# ----------------------------------------------------------------------
# Helper: add context to ambiguous food queries
# ----------------------------------------------------------------------
def add_food_context(product_name: str) -> str:
    """Append words like 'raw' or 'fresh' to improve API accuracy for generic items."""
    # Simple heuristics – can be expanded
    ambiguous_words = {
        "tom": "fresh tomato",
        "potato": "raw potato",
        "chicken breast": "chicken breast raw",
        "rice": "white rice raw",
        "bread": "whole wheat bread",
    }
    lowered = product_name.lower().strip()
    if lowered in ambiguous_words:
        return ambiguous_words[lowered]
    # If the name is very short, add " raw" or " generic"
    if len(lowered.split()) == 1 and len(lowered) <= 4:
        return f"{lowered} raw"
    return product_name

# ----------------------------------------------------------------------
# Source 1: USDA FoodData Central
# ----------------------------------------------------------------------
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

def search_usda_food(query: str, page_size: int = 5) -> Optional[List[Dict]]:
    if not USDA_API_KEY:
        logging.info("USDA API key not set – skipping.")
        return None

    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "api_key": USDA_API_KEY,
        "query": query,
        "pageSize": page_size,
        # USDA expects a comma‑separated string, NOT a list
        "dataType": "Foundation,SR Legacy",   # removed Survey (FNDDS) to avoid special‑char issues
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("foods", [])
    except Exception as e:
        logging.error(f"USDA search error: {e}")
        return None

def get_usda_nutrition_per_100g(food_item: Dict) -> Optional[Dict]:
    nutrients = {}
    for fn in food_item.get("foodNutrients", []):
        nid = fn.get("nutrientId")
        value = fn.get("value")
        if value is None:
            continue
        if nid == 1008:
            nutrients["energy-kcal_100g"] = value
        elif nid == 1003:
            nutrients["proteins_100g"] = value
        elif nid == 1004:
            nutrients["fat_100g"] = value
        elif nid == 1258:
            nutrients["saturated-fat_100g"] = value
        elif nid == 1005:
            nutrients["carbohydrates_100g"] = value
        elif nid == 2000:
            nutrients["sugars_100g"] = value
        elif nid == 1079:
            nutrients["fiber_100g"] = value
        elif nid == 1093:
            salt_per_100g = (value * 2.5) / 1000
            nutrients["salt_100g"] = round(salt_per_100g, 2)

    if "energy-kcal_100g" in nutrients:
        return nutrients
    return None

def get_nutrition_usda(product_name: str) -> Optional[Dict]:
    # Enrich the query to get better matches for fresh foods
    enriched = add_food_context(product_name)
    foods = search_usda_food(enriched)
    if not foods:
        return None

    # Fuzzy match on description
    best, best_score = None, 0
    for f in foods:
        score = fuzz.token_sort_ratio(product_name.lower(), f.get("description", "").lower())
        if score > best_score:
            best_score = score
            best = f
    if best and best_score >= MIN_FUZZY_SCORE:
        nutrients = get_usda_nutrition_per_100g(best)
        if nutrients:
            logging.info(f"✅ USDA: {best.get('description')} (score: {best_score})")
            return nutrients
    return None

# ----------------------------------------------------------------------
# Source 2: Open Food Facts
# ----------------------------------------------------------------------
def search_openfoodfacts(query: str) -> Optional[Dict]:
    try:
        api = openfoodfacts.API(user_agent="MyCapstone/1.0")
        # Try with country filter first
        results = api.product.text_search(
            query,
            page_size=SEARCH_PAGE_SIZE,
            cc="qatar",
        )
    except TypeError:
        # Fallback without country filter (older or different OFF version)
        try:
            results = api.product.text_search(query, page_size=SEARCH_PAGE_SIZE)
        except Exception as e:
            logging.error(f"OFF search error (fallback): {e}")
            return None
    except Exception as e:
        logging.error(f"OFF search error: {e}")
        return None

    if not results or "products" not in results:
        return None
    for p in results["products"]:
        if p.get("nutriments"):
            return {
                "product_name": p.get("product_name", query),
                "nutriments": p["nutriments"],
            }
    return None

def get_nutrition_openfoodfacts(product_name: str) -> Optional[Dict]:
    result = search_openfoodfacts(product_name)
    if result:
        logging.info(f"✅ OFF: {result['product_name']}")
        return result["nutriments"]

    # Broader fuzzy search
    api = openfoodfacts.API(user_agent="MyCapstone/1.0")
    try:
        broad = api.product.text_search(product_name, page_size=SEARCH_PAGE_SIZE*2, cc="world")
    except TypeError:
        broad = api.product.text_search(product_name, page_size=SEARCH_PAGE_SIZE*2)
    except Exception as e:
        logging.error(f"OFF broad search error: {e}")
        return None

    if not broad or "products" not in broad:
        return None

    best_sc, best_p = 0, None
    for p in broad["products"]:
        name = p.get("product_name", "")
        sc = fuzz.token_sort_ratio(product_name.lower(), name.lower())
        if sc > best_sc:
            best_sc = sc
            best_p = p
    if best_p and best_sc >= MIN_FUZZY_SCORE and best_p.get("nutriments"):
        logging.info(f"✅ OFF fuzzy: {best_p['product_name']} (score: {best_sc})")
        return best_p["nutriments"]
    return None

# ----------------------------------------------------------------------
# Source 3: API Ninjas (replaces CalorieNinjas)
# ----------------------------------------------------------------------
def get_nutrition_ninjas(product_name: str, quantity: Optional[float] = None, unit: Optional[str] = None) -> Optional[Dict]:
    if not NINJAS_API_KEY:
        logging.info("API Ninjas key not set – skipping.")
        return None

    query = product_name
    if quantity and unit:
        query = f"{quantity} {unit} {product_name}"

    url = "https://api.api-ninjas.com/v1/nutrition"
    headers = {"X-Api-Key": NINJAS_API_KEY}
    try:
        resp = requests.get(url, params={"query": query}, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.error(f"API Ninjas error: {e}")
        return None

    items = data if isinstance(data, list) else data.get("items", [])
    if not items:
        return None

    item = items[0]
    logging.info(f"✅ API Ninjas: {item.get('name', product_name)}")

    # Determine total weight for scaling
    total_weight_g = item.get("serving_size_g", 100)
    if not isinstance(total_weight_g, (int, float)) or total_weight_g <= 0:
        total_weight_g = 100
    factor = 100.0 / total_weight_g

    def safe_get(key: str, default: float = 0.0) -> float:
        value = item.get(key, default)
        if isinstance(value, str):
            # Premium field – try to estimate
            logging.debug(f"'{key}' is premium placeholder: '{value}'")
            return default
        return float(value) if value is not None else default

    fat = safe_get("fat_total_g")
    saturated_fat = safe_get("fat_saturated_g")
    carbs = safe_get("carbohydrates_total_g")
    fiber = safe_get("fiber_g")
    sugar = safe_get("sugar_g")
    sodium_mg = safe_get("sodium_mg")

    calories = safe_get("calories")
    protein = safe_get("protein_g")

    # Estimate missing values from available macronutrients
    if calories == 0.0 and (fat > 0 or carbs > 0 or protein > 0):
        estimated_protein = 0.15 * (fat * 9 + carbs * 4) / 4 if protein == 0 else protein
        calories = 9 * fat + 4 * carbs + 4 * estimated_protein
        logging.debug(f"Calories estimated: {calories:.1f} kcal")
    if protein == 0.0 and calories > 0:
        # Assume 15% of calories from protein
        protein = (calories * 0.15) / 4
        logging.debug(f"Protein estimated: {protein:.1f} g")

    nutrients = {
        "energy-kcal_100g": calories * factor,
        "fat_100g": fat * factor,
        "saturated-fat_100g": saturated_fat * factor,
        "carbohydrates_100g": carbs * factor,
        "sugars_100g": sugar * factor,
        "fiber_100g": fiber * factor,
        "proteins_100g": protein * factor,
        "salt_100g": (sodium_mg * 2.5 / 1000) * factor,
    }
    return nutrients

# ----------------------------------------------------------------------
# Source 4: Google Gemini (free LLM)
# ----------------------------------------------------------------------
def get_nutrition_gemini(product_name: str) -> Optional[Dict]:
    if not GEMINI_API_KEY:
        logging.info("Gemini API key not set – skipping.")
        return None

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    prompt = (
        "You are a nutritional database. For the following food product, return a JSON object "
        "with typical nutritional values per 100 grams (or per 100 ml for liquids). "
        "Include only these keys (all numeric): energy-kcal_100g, fat_100g, saturated-fat_100g, "
        "carbohydrates_100g, sugars_100g, fiber_100g, proteins_100g, salt_100g. "
        "If the product is unknown, return an empty JSON object {}."
    )
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"{prompt}\n\nProduct: {product_name}")
        content = response.text.strip()
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "energy-kcal_100g": data.get("energy-kcal_100g", 0),
                "fat_100g": data.get("fat_100g", 0),
                "saturated-fat_100g": data.get("saturated-fat_100g", 0),
                "carbohydrates_100g": data.get("carbohydrates_100g", 0),
                "sugars_100g": data.get("sugars_100g", 0),
                "fiber_100g": data.get("fiber_100g", 0),
                "proteins_100g": data.get("proteins_100g", 0),
                "salt_100g": data.get("salt_100g", 0),
            }
        else:
            logging.warning(f"Gemini response not valid JSON: {content}")
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
    return None

# ----------------------------------------------------------------------
# Source 5: Ollama (local, optional)
# ----------------------------------------------------------------------
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

def get_nutrition_ollama(product_name: str, model: str = "llama3.2") -> Optional[Dict]:
    if not OLLAMA_AVAILABLE:
        return None
    prompt = (
        "You are a nutritional database. Return a JSON object with per 100g values: "
        "energy-kcal_100g, fat_100g, saturated-fat_100g, carbohydrates_100g, "
        "sugars_100g, fiber_100g, proteins_100g, salt_100g.\n"
        f"Product: {product_name}"
    )
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        content = response["message"]["content"]
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "energy-kcal_100g": data.get("energy-kcal_100g", 0),
                "fat_100g": data.get("fat_100g", 0),
                "saturated-fat_100g": data.get("saturated-fat_100g", 0),
                "carbohydrates_100g": data.get("carbohydrates_100g", 0),
                "sugars_100g": data.get("sugars_100g", 0),
                "fiber_100g": data.get("fiber_100g", 0),
                "proteins_100g": data.get("proteins_100g", 0),
                "salt_100g": data.get("salt_100g", 0),
            }
    except Exception as e:
        logging.error(f"Ollama error: {e}")
    return None

# ----------------------------------------------------------------------
# Unified pipeline
# ----------------------------------------------------------------------
def get_product_nutrition(product_name: str) -> Optional[Dict]:
    # 1. USDA
    nutrients = get_nutrition_usda(product_name)
    if nutrients:
        return nutrients

    # 2. Open Food Facts
    nutrients = get_nutrition_openfoodfacts(product_name)
    if nutrients:
        return nutrients

    # 3. API Ninjas
    nutrients = get_nutrition_ninjas(product_name)
    if nutrients:
        return nutrients

    # 4. Google Gemini
    nutrients = get_nutrition_gemini(product_name)
    if nutrients:
        return nutrients

    # 5. Local Ollama
    nutrients = get_nutrition_ollama(product_name)
    if nutrients:
        return nutrients

    logging.warning(f"No source found data for '{product_name}'.")
    return None

# ----------------------------------------------------------------------
# Scaling and Totals
# ----------------------------------------------------------------------
def scale_nutrients(nutrients: Dict, quantity: float, unit: str) -> Dict[str, float]:
    if unit in ("g", "ml"):
        factor = quantity / 100.0
    elif unit == "kg":
        factor = quantity * 1000 / 100.0
    elif unit == "l":
        factor = quantity * 1000 / 100.0
    elif unit == "oz":
        factor = quantity * 28.3495 / 100.0
    elif unit == "lb":
        factor = quantity * 453.592 / 100.0
    elif unit == "piece":
        factor = quantity  # assume 100g per piece
    else:
        factor = quantity / 100.0

    scaled = {}
    for k, v in nutrients.items():
        if isinstance(v, (int, float)):
            scaled[k] = v * factor
        else:
            scaled[k] = v
    return scaled

def calculate_total_nutrition(
    receipt_lines: List[str],
    abbreviations: Optional[Dict[str, str]] = None,
) -> Dict:
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

        product_expanded = expand_abbreviations(product_raw, abbreviations)
        logging.info(f"Processing: '{product_expanded}' (from '{product_raw}')")

        nutrients_per100 = get_product_nutrition(product_expanded)

        scaled = None
        if nutrients_per100 and parsed["quantity"] is not None and parsed["unit"] is not None:
            scaled = scale_nutrients(nutrients_per100, parsed["quantity"], parsed["unit"])
            total_nutrients["energy-kcal"] += scaled.get("energy-kcal_100g", 0)
            total_nutrients["fat_g"] += scaled.get("fat_100g", 0)
            total_nutrients["saturated-fat_g"] += scaled.get("saturated-fat_100g", 0)
            total_nutrients["carbohydrates_g"] += scaled.get("carbohydrates_100g", 0)
            total_nutrients["sugars_g"] += scaled.get("sugars_100g", 0)
            total_nutrients["fiber_g"] += scaled.get("fiber_100g", 0)
            total_nutrients["proteins_g"] += scaled.get("proteins_100g", 0)
            total_nutrients["salt_g"] += scaled.get("salt_100g", 0)
        elif nutrients_per100:
            scaled = nutrients_per100
            warnings.append(
                f"No quantity for '{product_expanded}' – using per 100g/ml values without scaling."
            )
        else:
            warnings.append(f"No nutritional data found for '{product_expanded}'.")

        items.append({
            "raw": parsed["raw"],
            "product_original": product_raw,
            "product_expanded": product_expanded,
            "quantity": parsed["quantity"],
            "unit": parsed["unit"],
            "nutrients_per100": nutrients_per100,
            "scaled_nutrients": scaled,
        })

    return {"items": items, "totals": total_nutrients, "warnings": warnings}

# ----------------------------------------------------------------------
# Example run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_lines = [
        "CHK BRST 1.5KG @ 45.00",
        "WHL WHT BREAD 1PC @ 5.50",
        "TOM 0.5KG @ 3.00",
    ]

    result = calculate_total_nutrition(sample_lines)
    print("\n===== NUTRITION TOTALS =====")
    for k, v in result["totals"].items():
        print(f"{k}: {v:.2f}")
    print("\nWarnings:")
    for w in result["warnings"]:
        print(f" - {w}")