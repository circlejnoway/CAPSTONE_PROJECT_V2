from pydantic_settings import BaseSettings
from pydantic import Field

class NutritionConfig(BaseSettings):
    usda_api_key: str = Field("", env="USDA_API_KEY")
    ninjas_api_key: str = Field("", env="NINJAS_API_KEY")
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    min_fuzzy_score: int = Field(75, ge=0, le=100)
    search_page_size: int = Field(5, ge=1)
    sources: list = ["usda", "openfoodfacts", "ninjas"]
    enable_ai_abbreviation_fallback: bool = True
    enable_portion_estimation: bool = True
    ollama_model: str = "llama3.2"  # ← add this

    class Config:
        env_file = ".env"