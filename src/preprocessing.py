import re
import pandas as pd
from typing import List, Union
from pathlib import Path
from src.utils import logger, save_csv_report
from src.config import config

class TextPreprocessor:
    """Text preprocessing pipeline for transaction descriptions."""

    def __init__(self, lowercase: bool = True, normalize_whitespace: bool = True):
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        
    def clean_text(self, text: str) -> str:
        """Clean single transaction description string safely."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
            
        # 1. Lowercase if configured
        if self.lowercase:
            text = text.lower()
            
        # 2. Normalize URLs / web domains (e.g. www.amazon.com -> amazon)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # 3. Clean excessive non-alphanumeric noise while preserving meaningful terms (e.g., #, &, -, numbers)
        # We preserve hashes (#7731), dashes (amazon-au), and store numbers
        text = re.sub(r'[^\w\s\-\#\&]', ' ', text)
        
        # 4. Normalize multiple spaces/tabs/newlines to single space
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text

    def transform(self, texts: Union[pd.Series, List[str]]) -> pd.Series:
        """Transform series or list of transaction descriptions."""
        if isinstance(texts, list):
            texts = pd.Series(texts)
        return texts.astype(str).apply(self.clean_text)

    def generate_examples_report(self, df: pd.DataFrame, reports_dir: Path = None, n_examples: int = 30) -> pd.DataFrame:
        """Generate before/after preprocessing examples report."""
        if reports_dir is None:
            reports_dir = config.reports_dir
            
        sample = df.sample(n=min(n_examples, len(df)), random_state=config.random_seed) if hasattr(config, 'random_seed') else df.head(n_examples)
        
        raw_texts = sample["transaction_description"].tolist()
        cleaned_texts = [self.clean_text(t) for t in raw_texts]
        
        examples_df = pd.DataFrame({
            "Original Description": raw_texts,
            "Preprocessed Description": cleaned_texts,
            "Category": sample["category"].tolist() if "category" in sample.columns else ["N/A"] * len(raw_texts)
        })
        
        save_csv_report(examples_df, reports_dir / "preprocessing_examples.csv")
        logger.info("Generated preprocessing_examples.csv report.")
        return examples_df
