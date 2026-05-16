import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class NutritionAPIClient(ABC):
    @abstractmethod
    def search(self, product: str, quantity: float,
               unit: str) -> Optional[Dict]: ...

class USDAClient(NutritionAPIClient):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"

    def search(self, product: str, quantity: float,
               unit: str) -> Optional[Dict]:
        try:
            resp = requests.get(
                f"{self.base_url}/foods/search",
                params={"query": product, "api_key": self.api_key,
                        "pageSize": 1},
                timeout=5
            )
            data = resp.json()
            if not data.get("foods"):
                return None
            food = data["foods"][0]
            nutrients = {n["nutrientName"]: n["value"]
                         for n in food.get("foodNutrients", [])}
            return {
                "energy_kcal_100g": nutrients.get("Energy"),
                "proteins_100g": nutrients.get("Protein"),
                "fat_100g": nutrients.get("Total lipid (fat)"),
                "carbohydrates_100g": nutrients.get(
                    "Carbohydrate, by difference"),
            }
        except Exception as e:
            logger.warning(f"USDA failed for {product}: {e}")
            return None

class OpenFoodFactsClient(NutritionAPIClient):
    def search(self, product: str, quantity: float,
               unit: str) -> Optional[Dict]:
        try:
            resp = requests.get(
                "https://world.openfoodfacts.org/cgi/search.pl",
                params={"search_terms": product, "json": 1,
                        "page_size": 1},
                timeout=5
            )
            products = resp.json().get("products", [])
            if not products:
                return None
            n = products[0].get("nutriments", {})
            return {
                "energy_kcal_100g": n.get("energy-kcal_100g"),
                "proteins_100g": n.get("proteins_100g"),
                "fat_100g": n.get("fat_100g"),
                "carbohydrates_100g": n.get("carbohydrates_100g"),
            }
        except Exception as e:
            logger.warning(f"OpenFoodFacts failed for {product}: {e}")
            return None

class APIClientFactory:
    @staticmethod
    def create_clients(config) -> List[NutritionAPIClient]:
        clients = []
        for source in config.sources:
            if source == "usda" and config.usda_api_key:
                clients.append(USDAClient(config.usda_api_key))
            elif source == "openfoodfacts":
                clients.append(OpenFoodFactsClient())
        return clients