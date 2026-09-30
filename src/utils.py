import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def setup_logger(name: str = "ai_finance_advisor", level: int = logging.INFO) -> logging.Logger:
    """Set up structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()

def save_csv_report(df: pd.DataFrame, file_path: Path, index: bool = False) -> None:
    """Save dataframe as CSV report ensuring directory exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    logger.info(f"Saved report to {file_path}")
