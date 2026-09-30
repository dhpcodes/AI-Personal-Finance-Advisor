import pandas as pd
from pathlib import Path
from typing import Optional, Union, List
from src.utils import logger

class DataLoader:
    """Memory-efficient transaction data loader for Parquet and CSV files."""

    @staticmethod
    def load_parquet(
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load Parquet dataset safely using PyArrow engine."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found at {path}")
            
        logger.info(f"Loading Parquet dataset from {path}...")
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df

    @staticmethod
    def load_csv(
        file_path: Union[str, Path],
        required_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load CSV dataset safely with schema validation."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at {path}")
            
        logger.info(f"Loading CSV dataset from {path}...")
        df = pd.read_csv(path)
        logger.info(f"Loaded CSV dataset with shape {df.shape}")
        
        if required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")
                
        return df
