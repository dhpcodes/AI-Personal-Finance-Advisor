from src.config import config
from src.data_loader import DataLoader
from src.split_data import DataSplitter
from src.utils import logger

def main():
    logger.info("--- Starting Phase 4: Data Leakage Audit & Train/Val/Test Split ---")
    
    # 1. Load Data
    raw_path = config.raw_data_path
    df = DataLoader.load_parquet(raw_path)
    
    splitter = DataSplitter(seed=config.random_seed)
    
    # 2. Run leakage audit
    splitter.run_leakage_audit(df)
    
    # 3. Create reproducible train, validation, and test splits
    # Using sample_size_for_training from config (200,000 unique records) for high efficiency & exact model benchmarking
    sample_size = config.data_split_params.get("sample_size_for_training", 200000)
    train_df, val_df, test_df, metadata = splitter.create_splits(
        df, sample_size=sample_size, deduplicate=True
    )
    
    logger.info(f"Splits Created Successfully!")
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    logger.info("--- Phase 4 Completed Successfully! ---")

if __name__ == "__main__":
    main()
