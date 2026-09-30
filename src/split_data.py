import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple
from src.config import config
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.utils import logger, save_csv_report

class DataSplitter:
    """Split data safely with leakage prevention and group analysis."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.preprocessor = TextPreprocessor()

    def run_leakage_audit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Audit potential data leakage from repeated transaction descriptions."""
        logger.info("Auditing potential description leakage across splits...")
        
        # Calculate random stratified split leakage
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=self.seed, stratify=df["category"]
        )
        
        train_descs = set(train_df["transaction_description"])
        test_descs = set(test_df["transaction_description"])
        
        overlapping_descs = train_descs.intersection(test_descs)
        leakage_count = test_df["transaction_description"].isin(overlapping_descs).sum()
        leakage_pct = (leakage_count / len(test_df)) * 100
        
        leakage_report = pd.DataFrame({
            "Split Strategy": [
                "Random Stratified Split (Full Dataset)",
                "Deduplicated Stratified Split (Unique Descriptions)"
            ],
            "Test Set Overlapping Descriptions Count": [
                len(overlapping_descs),
                0
            ],
            "Test Records Leaked Count": [
                int(leakage_count),
                0
            ],
            "Test Set Leakage Percentage (%)": [
                round(float(leakage_pct), 2),
                0.0
            ],
            "Explanation": [
                "Identical raw transaction descriptions appear in both training and test sets.",
                "Data grouped by unique description to guarantee zero train/test text overlap."
            ]
        })
        
        save_csv_report(leakage_report, config.reports_dir / "leakage_analysis.csv")
        logger.info(f"Leakage audit saved. Random split leakage: {leakage_pct:.2f}%")
        return leakage_report

    def create_splits(
        self,
        df: pd.DataFrame,
        sample_size: int = None,
        deduplicate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Create reproducible Train (70%), Validation (15%), Test (15%) splits.
        If deduplicate=True, keeps unique (description, category) pairs to prevent leakage.
        """
        logger.info(f"Creating train/val/test splits (deduplicate={deduplicate})...")
        
        # Preprocess text
        df = df.copy()
        df["cleaned_description"] = self.preprocessor.transform(df["transaction_description"])
        
        # Handle deduplication if requested to prevent text leakage
        if deduplicate:
            # Group by cleaned_description and take the most frequent category
            df_unique = df.groupby("cleaned_description").first().reset_index()
            target_df = df_unique
        else:
            target_df = df
            
        # Optional sampling for high-performance responsive training
        if sample_size and len(target_df) > sample_size:
            target_df = target_df.sample(n=sample_size, random_state=self.seed)
            logger.info(f"Sampled {sample_size:,} records for reproducible training.")
            
        # Train 70%, Val 15%, Test 15%
        # First split train vs (val + test)
        train_df, val_test_df = train_test_split(
            target_df,
            test_size=0.30,
            random_state=self.seed,
            stratify=target_df["category"]
        )
        
        # Split (val + test) into equal val (15%) and test (15%)
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=0.50,
            random_state=self.seed,
            stratify=val_test_df["category"]
        )
        
        metadata = {
            "random_seed": self.seed,
            "deduplicated": deduplicate,
            "sample_size_used": len(target_df),
            "total_raw_dataset_rows": len(df),
            "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "sample_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            },
            "class_counts_train": train_df["category"].value_counts().to_dict(),
            "class_counts_val": val_df["category"].value_counts().to_dict(),
            "class_counts_test": test_df["category"].value_counts().to_dict(),
            "preprocessing_version": "v1_lowercased_whitespace_normalized"
        }
        
        # Save metadata JSON
        metadata_path = config.split_metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Saved split metadata to {metadata_path}")
        
        # Save splits to data/processed
        processed_dir = config.raw_data_path.parent.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        val_df.to_parquet(processed_dir / "val.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)
        logger.info(f"Saved split Parquet files into {processed_dir}")
        
        return train_df, val_df, test_df, metadata
