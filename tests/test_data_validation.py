import pytest
import pandas as pd
from src.data_validation import DataValidator

def test_raw_schema_validation_valid():
    df = pd.DataFrame(columns=["transaction_description", "category", "country", "currency"])
    is_valid, missing = DataValidator.validate_raw_schema(df)
    assert is_valid is True
    assert len(missing) == 0

def test_raw_schema_validation_missing():
    df = pd.DataFrame(columns=["transaction_description", "category"])
    is_valid, missing = DataValidator.validate_raw_schema(df)
    assert is_valid is False
    assert "country" in missing
    assert "currency" in missing

def test_user_upload_validation_valid():
    df = pd.DataFrame({
        "description": ["Groceries at Walmart", "Salary Payment"],
        "amount": ["45.50", "3500.00"],
        "date": ["2026-01-15", "2026-01-01"]
    })
    is_valid, val_info, clean_df = DataValidator.validate_user_upload(df)
    assert is_valid is True
    assert val_info["valid_rows"] == 2
    assert "transaction_description" in clean_df.columns
    assert "amount" in clean_df.columns

def test_user_upload_validation_missing_desc():
    df = pd.DataFrame({
        "amount": [10.0, 20.0],
        "date": ["2026-01-01", "2026-01-02"]
    })
    is_valid, val_info, clean_df = DataValidator.validate_user_upload(df)
    assert is_valid is False
    assert len(val_info["errors"]) > 0
