from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ml-fraud-service")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)

    model_registry_path: Path = Field(default=Path("./model_registry"))
    data_path: Path = Field(default=Path("./data"))

    inference_host: str = Field(default="0.0.0.0")
    inference_port: int = Field(default=8000)

    optimal_threshold: float = Field(default=0.251)

    min_precision: float = Field(default=0.80)
    min_recall: float = Field(default=0.70)
    min_f1: float = Field(default=0.75)

    lgbm_learning_rate: float = Field(default=0.05)
    lgbm_max_depth: int = Field(default=5)
    lgbm_min_child_samples: int = Field(default=20)
    lgbm_n_estimators: int = Field(default=200)
    lgbm_num_leaves: int = Field(default=31)
    lgbm_random_state: int = Field(default=42)

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    def get_lgbm_params(self) -> dict:
        return {
            "learning_rate": self.lgbm_learning_rate,
            "max_depth": self.lgbm_max_depth,
            "min_child_samples": self.lgbm_min_child_samples,
            "n_estimators": self.lgbm_n_estimators,
            "num_leaves": self.lgbm_num_leaves,
            "random_state": self.lgbm_random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    def get_quality_gates(self) -> dict:
        return {
            "precision": self.min_precision,
            "recall": self.min_recall,
            "f1": self.min_f1,
        }


settings = Settings()
