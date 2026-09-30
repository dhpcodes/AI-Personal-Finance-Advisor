# Dataset Analysis & Inspection Report

## Status: VERIFIED FROM RAW DATA

- **Total Rows**: 4,501,043
- **Total Columns**: 4
- **Columns**: `['transaction_description', 'category', 'country', 'currency']`
- **Missing Values**: `0` across all fields
- **Unique Categories**: `10`
- **Exact Duplicate Rows**: `2,940,825` (65.34%)
- **Unique Descriptions**: `1,387,044`
- **Descriptions Mapped to >1 Category**: `2,485`

## Status: DOCUMENTATION ONLY

- Dataset repository: `mitulshah/transaction-categorization` on Hugging Face.
- Main file: `default/train/0000.parquet` stored locally at `data/raw/0000.parquet`.

## Status: NOT VERIFIED

- Transaction dates and transaction monetary amounts are **NOT** present in the Hugging Face dataset.
- User uploads with transaction amounts and dates are required for cash flow financial analytics.
