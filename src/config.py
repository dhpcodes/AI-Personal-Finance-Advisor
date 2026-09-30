import os
import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"

class Config:
    """Configuration loader and manager for AI Personal Finance Advisor."""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
            
    @property
    def project_name(self) -> str:
        return self._config.get("project", {}).get("name", "AI Personal Finance Advisor")
        
    @property
    def random_seed(self) -> int:
        return self._config.get("project", {}).get("random_seed", 42)
        
    @property
    def raw_data_path(self) -> Path:
        base_dir = self.config_path.parent.parent
        return base_dir / self._config.get("paths", {}).get("raw_data", "data/raw/0000.parquet")
        
    @property
    def reports_dir(self) -> Path:
        base_dir = self.config_path.parent.parent
        p = base_dir / self._config.get("paths", {}).get("reports_dir", "reports")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_dir(self) -> Path:
        base_dir = self.config_path.parent.parent
        p = base_dir / self._config.get("paths", {}).get("models_dir", "models")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def final_model_path(self) -> Path:
        base_dir = self.config_path.parent.parent
        return base_dir / self._config.get("paths", {}).get("final_model_path", "models/final_model.joblib")

    @property
    def split_metadata_path(self) -> Path:
        base_dir = self.config_path.parent.parent
        return base_dir / self._config.get("paths", {}).get("split_metadata_path", "reports/split_metadata.json")

    @property
    def data_split_params(self) -> Dict[str, Any]:
        return self._config.get("data_split", {})

    @property
    def tfidf_params(self) -> Dict[str, Any]:
        params = self._config.get("tfidf", {})
        if "ngram_range" in params and isinstance(params["ngram_range"], list):
            params["ngram_range"] = tuple(params["ngram_range"])
        return params

    @property
    def model_params(self) -> Dict[str, Any]:
        return self._config.get("models", {})

    @property
    def analytics_params(self) -> Dict[str, Any]:
        return self._config.get("analytics", {})

    @property
    def app_params(self) -> Dict[str, Any]:
        return self._config.get("app", {})

# Global instance
config = Config()
