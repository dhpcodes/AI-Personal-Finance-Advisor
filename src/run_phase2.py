import pandas as pd
from pathlib import Path
from src.config import config
from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.preprocessing import TextPreprocessor
from src.utils import logger

def main():
    logger.info("--- Starting Phase 2: Dataset Inspection & Preprocessing ---")
    
    # 1. Load Parquet Data
    raw_path = config.raw_data_path
    df = DataLoader.load_parquet(raw_path)
    
    # 2. Validate Schema
    is_valid, missing = DataValidator.validate_raw_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed! Missing columns: {missing}")
    logger.info("Raw dataset schema validation passed.")
    
    # 3. Analyze dataset and save all 7 inspection CSV reports + markdown report
    DataValidator.analyze_dataset(df, config.reports_dir)
    
    # 4. Generate Preprocessing Examples Report
    preprocessor = TextPreprocessor()
    preprocessor.generate_examples_report(df, config.reports_dir, n_examples=50)
    
    logger.info("--- Phase 2 Completed Successfully! ---")

if __name__ == "__main__":
    main()
