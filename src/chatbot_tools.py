
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from src.analytics import FinanceAnalytics
from src.predict import TransactionPredictor
from src.explainability import ModelExplainer
from src.anomaly_detection import SpendingAnomalyDetector
from src.utils import logger


DEFAULT_CURRENCY = "INR"


class ChatbotFinancialTools:
    """
    Deterministic financial tools layer for the AI Personal Finance Chatbot.
    Financial calculations are performed in Python using active transaction data.
    """

    @staticmethod
    def _normalize_and_validate_df(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize transaction schema and classify income and expenses."""

        info = {
            "is_valid": True,
            "error": None,
            "warnings": [],
            "currencies": [DEFAULT_CURRENCY],
        }

        if df is None or len(df) == 0:
            info["is_valid"] = False
            info["error"] = (
                "No transaction data is available in the current session."
            )
            return pd.DataFrame(), info

        df_clean = df.copy()
        if "_row_id" not in df_clean.columns:
            df_clean["_row_id"] = list(range(len(df_clean)))
        df_clean.columns = [str(col).strip() for col in df_clean.columns]

        # 1. Identify transaction description column.
        desc_col = next(
            (
                col
                for col in [
                    "transaction_description",
                    "description",
                    "text",
                    "narration",
                ]
                if col in df_clean.columns
            ),
            None,
        )

        if desc_col is None:
            info["is_valid"] = False
            info["error"] = (
                "Missing transaction description column in dataset."
            )
            return pd.DataFrame(), info

        df_clean["transaction_description"] = (
            df_clean[desc_col].fillna("").astype(str).str.strip()
        )

        # 2. Prefer final_category, then original category,
        # and use predicted_category only when needed.
        cat_col = next(
            (
                col
                for col in [
                    "final_category",
                    "category",
                    "predicted_category",
                ]
                if col in df_clean.columns
            ),
            None,
        )

        if cat_col is None:
            df_clean["category"] = "Uncategorized"
            info["warnings"].append(
                "No category column found; defaulting to 'Uncategorized'."
            )
        else:
            df_clean["category"] = (
                df_clean[cat_col]
                .fillna("Uncategorized")
                .astype(str)
                .str.strip()
            )
            df_clean["category"] = df_clean["category"].replace(
                "", "Uncategorized"
            )

        # 3. Detect currency.
        if "currency" in df_clean.columns:
            currencies = [
                str(value).strip().upper()
                for value in df_clean["currency"].dropna().unique()
                if str(value).strip()
            ]

            if len(currencies) > 1:
                info["warnings"].append(
                    "Multiple currencies detected. Amounts are not converted "
                    "between currencies; results use the first detected "
                    "currency label."
                )

            info["currencies"] = currencies or [DEFAULT_CURRENCY]
            df_clean["currency"] = (
                df_clean["currency"]
                .fillna(info["currencies"][0])
                .astype(str)
                .str.strip()
                .str.upper()
            )
        else:
            df_clean["currency"] = DEFAULT_CURRENCY
            info["currencies"] = [DEFAULT_CURRENCY]

        # 4. Parse transaction amounts.
        if "amount" in df_clean.columns:
            # Handles common formats such as "$1,200.00" and "₹1,200.00".
            amount_values = (
                df_clean["amount"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("INR", "", regex=False)
                .str.replace("USD", "", regex=False)
                .str.strip()
            )
            df_clean["amount"] = pd.to_numeric(
                amount_values, errors="coerce"
            )
            invalid_amounts = int(df_clean["amount"].isna().sum())

            if invalid_amounts:
                info["warnings"].append(
                    f"{invalid_amounts} transaction amount(s) could not be "
                    "parsed and were treated as zero."
                )

            df_clean["amount"] = df_clean["amount"].fillna(0.0)
        else:
            df_clean["amount"] = 0.0
            info["warnings"].append(
                "Missing 'amount' column in transaction file."
            )

        # 5. Classify income and expenses.
        type_col = next(
            (
                col
                for col in ["transaction_type", "type"]
                if col in df_clean.columns
            ),
            None,
        )

        category_lower = (
            df_clean["category"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if type_col is not None:
            transaction_type = (
                df_clean[type_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            income_labels = {
                "income",
                "credit",
                "deposit",
                "salary",
                "earning",
                "earnings",
            }
            expense_labels = {
                "expense",
                "expenses",
                "debit",
                "withdrawal",
                "payment",
                "spending",
            }

            is_income = pd.Series(
                np.where(
                    transaction_type.isin(income_labels),
                    True,
                    np.where(
                        transaction_type.isin(expense_labels),
                        False,
                        category_lower.eq("income"),
                    ),
                ),
                index=df_clean.index,
            )
        else:
            is_income = category_lower.eq("income")
            info["warnings"].append(
                "No transaction type column found. Income is identified "
                "using the category label 'Income'; verify the dataset."
            )

        df_clean["is_income"] = is_income.astype(bool)

        # 6. Parse transaction dates.
        if "date" in df_clean.columns:
            df_clean["parsed_date"] = pd.to_datetime(
                df_clean["date"], errors="coerce"
            )
            invalid_dates = int(df_clean["parsed_date"].isna().sum())

            if invalid_dates:
                info["warnings"].append(
                    f"{invalid_dates} transaction date(s) could not be parsed."
                )
        else:
            df_clean["parsed_date"] = pd.NaT
            info["warnings"].append(
                "No date column found; date-based analysis may be unavailable."
            )

        return df_clean, info

    @staticmethod
    def get_spending_by_category(
        df: pd.DataFrame,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """Return spending totals, category shares, and top category."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        if date_range:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            df_clean = df_clean[
                df_clean["parsed_date"].between(start_dt, end_dt)
            ]

        expense_df = df_clean[~df_clean["is_income"]].copy()

        if expense_df.empty:
            return {
                "status": "insufficient_data",
                "message": "No expense transactions found in the active dataset.",
                "total_expense": 0.0,
                "highest_spending_category": None,
                "highest_spending_amount": 0.0,
                "category_breakdown": [],
                "currency": info["currencies"][0],
                "warnings": info["warnings"],
            }

        expense_df["abs_amount"] = expense_df["amount"].abs()
        total_expense = float(expense_df["abs_amount"].sum())

        cat_grouped = (
            expense_df.groupby("category", dropna=False)
            .agg(
                total_amount=("abs_amount", "sum"),
                transaction_count=("abs_amount", "count"),
            )
            .reset_index()
        )

        cat_grouped["percentage"] = np.where(
            total_expense > 0,
            np.round(
                cat_grouped["total_amount"] / total_expense * 100, 2
            ),
            0.0,
        )

        cat_grouped = cat_grouped.sort_values(
            by="total_amount", ascending=False
        ).reset_index(drop=True)

        top_cat = cat_grouped.iloc[0] if not cat_grouped.empty else None

        breakdown = [
            {
                "category": str(row["category"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "percentage": float(row["percentage"]),
                "transaction_count": int(row["transaction_count"]),
            }
            for _, row in cat_grouped.iterrows()
        ]

        return {
            "status": "success",
            "total_expense": round(total_expense, 2),
            "currency": info["currencies"][0],
            "highest_spending_category": (
                str(top_cat["category"]) if top_cat is not None else "N/A"
            ),
            "highest_spending_amount": (
                round(float(top_cat["total_amount"]), 2)
                if top_cat is not None
                else 0.0
            ),
            "category_breakdown": breakdown,
            "warnings": info["warnings"],
        }

    @staticmethod
    def get_category_spending_for_period(
        df: pd.DataFrame,
        category: str,
        period: str = "this_month",
        date_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """Get spending for a category and compare with the prior month."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        target_cat = category.strip().lower()
        category_lower = (
            df_clean["category"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        category_aliases = {
            "food & dining": [
                "food & dining",
                "dining out",
                "food",
                "restaurants",
            ],
            "dining out": [
                "dining out",
                "food & dining",
                "food",
                "restaurants",
            ],
            "food": [
                "food",
                "food & dining",
                "dining out",
                "restaurants",
            ],
            "restaurants": [
                "restaurants",
                "food & dining",
                "dining out",
                "food",
            ],
        }

        accepted_categories = category_aliases.get(
            target_cat, [target_cat]
        )

        matched_df = df_clean[
            category_lower.isin(accepted_categories)
            & (~df_clean["is_income"])
        ].copy()

        if matched_df.empty:
            matched_df = df_clean[
                category_lower.str.contains(
                    target_cat, regex=False, na=False
                )
                & (~df_clean["is_income"])
            ].copy()

        if matched_df.empty:
            return {
                "status": "not_found",
                "message": (
                    f"No expense transactions found for category '{category}'."
                ),
                "available_categories": sorted(
                    df_clean.loc[~df_clean["is_income"], "category"]
                    .dropna()
                    .unique()
                    .tolist()
                ),
            }

        matched_df["abs_amount"] = matched_df["amount"].abs()

        if date_range:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            period_df = matched_df[
                matched_df["parsed_date"].between(start_dt, end_dt)
            ]

            return {
                "status": "success",
                "category": category,
                "period": f"{start_dt.date()} to {end_dt.date()}",
                "current_spending": round(
                    float(period_df["abs_amount"].sum()), 2
                ),
                "transaction_count": int(len(period_df)),
                "currency": info["currencies"][0],
                "warnings": info["warnings"],
            }

        valid_dates = matched_df.dropna(
            subset=["parsed_date"]
        ).sort_values(by="parsed_date")

        if not valid_dates.empty:
            latest_date = valid_dates["parsed_date"].max()
            current_month_str = latest_date.strftime("%Y-%m")

            curr_period_df = valid_dates[
                valid_dates["parsed_date"].dt.strftime("%Y-%m")
                == current_month_str
            ]

            curr_spending = float(curr_period_df["abs_amount"].sum())
            curr_count = len(curr_period_df)

            prev_month_dt = latest_date.replace(day=1) - pd.Timedelta(days=1)
            prev_month_str = prev_month_dt.strftime("%Y-%m")

            prev_period_df = valid_dates[
                valid_dates["parsed_date"].dt.strftime("%Y-%m")
                == prev_month_str
            ]

            has_prev = not prev_period_df.empty
            prev_spending = (
                float(prev_period_df["abs_amount"].sum())
                if has_prev
                else 0.0
            )

            pct_change = None
            if has_prev and prev_spending > 0:
                pct_change = round(
                    (curr_spending - prev_spending) / prev_spending * 100,
                    1,
                )

            return {
                "status": "success",
                "category": category,
                "current_period": current_month_str,
                "current_spending": round(curr_spending, 2),
                "transaction_count": int(curr_count),
                "has_previous_period": has_prev,
                "previous_period": prev_month_str if has_prev else None,
                "previous_spending": (
                    round(prev_spending, 2) if has_prev else None
                ),
                "percentage_change": pct_change,
                "total_historical_category_spending": round(
                    float(matched_df["abs_amount"].sum()), 2
                ),
                "currency": info["currencies"][0],
                "warnings": info["warnings"],
            }

        return {
            "status": "success",
            "category": category,
            "period": "All Time (dates unavailable)",
            "current_spending": round(
                float(matched_df["abs_amount"].sum()), 2
            ),
            "transaction_count": int(len(matched_df)),
            "has_previous_period": False,
            "currency": info["currencies"][0],
            "warnings": info["warnings"],
        }

    @staticmethod
    def get_recurring_expenses(
        df: pd.DataFrame, min_count: int = 2
    ) -> Dict[str, Any]:
        """Detect recurring expenses and subscription patterns."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        expense_df = df_clean[~df_clean["is_income"]].copy()

        if expense_df.empty:
            return {
                "status": "insufficient_data",
                "message": "No expense transactions available for recurring detection.",
            }

        recurring_df = FinanceAnalytics.detect_recurring_transactions(
            expense_df, min_count=min_count
        )

        items = []
        for _, row in recurring_df.iterrows():
            description = str(row["transaction_description"])
            frequency = int(row["Frequency"])
            average_amount = float(row.get("Avg_Amount", 0.0))

            items.append(
                {
                    "merchant": description,
                    "frequency": frequency,
                    "average_amount": round(average_amount, 2),
                    "classification": (
                        "Confirmed Recurring"
                        if frequency >= 3
                        else "Possible Recurring (Repeated)"
                    ),
                }
            )

        return {
            "status": "success",
            "recurring_expenses_count": len(items),
            "recurring_expenses": items,
            "currency": info["currencies"][0],
            "warnings": info["warnings"],
        }

    @staticmethod
    def get_highest_recurring_expense(
        previous_result: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Compute the recurring expense pattern with the highest average amount from previous tool results."""
        rec_result = previous_result
        if not rec_result or not isinstance(rec_result, dict) or rec_result.get("status") != "success":
            return {
                "status": "unavailable",
                "message": "Previous recurring expense results are unavailable. Please ask 'What are my recurring expenses?' first or clarify your question.",
            }

        rec_list = rec_result.get("recurring_expenses", [])
        if not rec_list:
            return {
                "status": "not_found",
                "message": "No recurring expenses were found in the previous calculation to determine the highest average amount.",
            }

        highest_item = max(
            rec_list,
            key=lambda item: float(item.get("average_amount", 0.0)),
        )

        return {
            "status": "success",
            "highest_recurring_expense": highest_item,
            "merchant": str(highest_item.get("merchant", "N/A")),
            "average_amount": round(float(highest_item.get("average_amount", 0.0)), 2),
            "frequency": int(highest_item.get("frequency", 0)),
            "classification": str(highest_item.get("classification", "")),
            "currency": rec_result.get("currency", DEFAULT_CURRENCY),
            "total_recurring_count": len(rec_list),
            "warnings": rec_result.get("warnings", []),
        }

    @staticmethod
    def explain_transaction_classification(
        transaction_description: str,
        predictor: Optional[TransactionPredictor] = None,
        explainer: Optional[ModelExplainer] = None,
    ) -> Dict[str, Any]:
        """Explain a transaction's predicted category."""

        if not transaction_description or not transaction_description.strip():
            return {
                "status": "error",
                "message": "Please provide a valid transaction description string.",
            }

        try:
            if explainer is None:
                predictor_inst = predictor or TransactionPredictor()
                explainer_inst = ModelExplainer(predictor_inst)
            else:
                explainer_inst = explainer

            result = explainer_inst.explain_prediction(
                transaction_description
            )

            return {
                "status": "success",
                "transaction_description": transaction_description,
                "predicted_category": result.get(
                    "predicted_category", "Unknown"
                ),
                "confidence_score": round(
                    float(result.get("confidence", 0.0)) * 100, 1
                ),
                "explanation_summary": result.get(
                    "explanation_summary", ""
                ),
                "top_k_probabilities": result.get("top_k", []),
                "important_features": result.get(
                    "important_features", []
                ),
            }

        except Exception as exc:
            logger.error(f"Classification explanation failed: {exc}")
            return {
                "status": "error",
                "message": f"Unable to explain classification: {str(exc)}",
                "transaction_description": transaction_description,
            }

    @staticmethod
    def forecast_monthly_expenses(
        df: pd.DataFrame,
        forecast_months: int = 1,
        min_months: int = 2,
    ) -> Dict[str, Any]:
        """Forecast expenses using a weighted average and linear trend."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        expense_df = df_clean[~df_clean["is_income"]].copy()
        valid_dates = expense_df.dropna(subset=["parsed_date"]).copy()

        if valid_dates.empty:
            return {
                "status": "insufficient_data",
                "message": (
                    "No valid date-stamped expense transactions found "
                    "to generate a forecast."
                ),
            }

        valid_dates["YearMonth"] = (
            valid_dates["parsed_date"].dt.to_period("M").astype(str)
        )
        valid_dates["abs_amount"] = valid_dates["amount"].abs()
        monthly_exp = valid_dates.groupby("YearMonth")["abs_amount"].sum()

        if len(monthly_exp) < min_months:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Not enough historical data. Found {len(monthly_exp)} "
                    f"month(s); at least {min_months} are required."
                ),
                "available_months_count": len(monthly_exp),
                "historical_monthly_spending": {
                    month: round(float(value), 2)
                    for month, value in monthly_exp.items()
                },
            }

        y_vals = monthly_exp.values
        x_vals = np.arange(len(y_vals))
        weights = np.linspace(1, 2, len(y_vals))
        weighted_average = np.average(y_vals, weights=weights)

        if len(y_vals) >= 3:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            trend_prediction = max(
                0.0, slope * len(y_vals) + intercept
            )
            forecast_value = (
                0.6 * weighted_average + 0.4 * trend_prediction
            )
        else:
            forecast_value = weighted_average

        std_dev = (
            float(np.std(y_vals))
            if len(y_vals) >= 3
            else 0.15 * forecast_value
        )

        return {
            "status": "success",
            "forecast_method": (
                "Weighted Moving Average + Linear Trend Baseline"
            ),
            "forecast_period": "Next Month",
            "predicted_amount": round(float(forecast_value), 2),
            "lower_bound": round(
                max(0.0, forecast_value - std_dev), 2
            ),
            "upper_bound": round(forecast_value + std_dev, 2),
            "historical_months_count": len(monthly_exp),
            "recent_3_months_avg": round(
                float(np.mean(y_vals[-3:])), 2
            ),
            "currency": info["currencies"][0],
            "historical_monthly_spending": {
                month: round(float(value), 2)
                for month, value in monthly_exp.items()
            },
        }

    @staticmethod
    def get_monthly_income_expense(
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute monthly income, expenses, and net cash flow."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        trends_df = FinanceAnalytics.get_monthly_trends(df_clean)

        if trends_df.empty:
            return {
                "status": "insufficient_data",
                "message": (
                    "Unable to calculate monthly cash flows due to "
                    "missing or invalid date and amount data."
                ),
            }

        return {
            "status": "success",
            "monthly_cash_flow": trends_df.to_dict(orient="records"),
            "currency": info["currencies"][0],
            "warnings": info["warnings"],
        }

    @staticmethod
    def get_transaction_details(
        df: pd.DataFrame,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Retrieve transactions matching a query or category."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        filtered = df_clean.copy()

        if category:
            requested_category = category.strip().lower()
            aliases = {
                "food & dining": [
                    "food & dining",
                    "dining out",
                    "food",
                    "restaurants",
                ],
                "dining out": [
                    "dining out",
                    "food & dining",
                    "food",
                    "restaurants",
                ],
                "food": [
                    "food",
                    "food & dining",
                    "dining out",
                    "restaurants",
                ],
                "restaurants": [
                    "restaurants",
                    "food & dining",
                    "dining out",
                    "food",
                ],
            }

            accepted = aliases.get(
                requested_category, [requested_category]
            )
            filtered = filtered[
                filtered["category"]
                .fillna("")
                .str.strip()
                .str.lower()
                .isin(accepted)
            ]

        if query:
            query_clean = query.strip().lower()
            filtered = filtered[
                filtered["transaction_description"]
                .str.lower()
                .str.contains(query_clean, regex=False, na=False)
            ]

        if filtered.empty:
            return {
                "status": "not_found",
                "message": (
                    f"No transactions matching query "
                    f"'{query or category}' found."
                ),
            }

        columns = [
            col
            for col in [
                "date",
                "transaction_description",
                "amount",
                "category",
                "currency",
                "is_income",
            ]
            if col in filtered.columns
        ]

        result_df = filtered.head(max(1, limit))[columns]

        return {
            "status": "success",
            "matched_count": int(len(filtered)),
            "returned_count": int(len(result_df)),
            "transactions": result_df.to_dict(orient="records"),
            "warnings": info["warnings"],
        }

    @staticmethod
    def detect_unusual_spending(
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Detect unusual transactions."""

        df_clean, info = ChatbotFinancialTools._normalize_and_validate_df(df)

        if not info["is_valid"]:
            return {"status": "error", "message": info["error"]}

        expense_df = df_clean[~df_clean["is_income"]].copy()

        if expense_df.empty:
            return {
                "status": "insufficient_data",
                "message": "No expense transactions available for anomaly detection.",
                "unusual_transactions_count": 0,
                "unusual_transactions": [],
                "currency": info["currencies"][0],
            }

        detector = SpendingAnomalyDetector()
        anomaly_df = detector.detect_anomalies(expense_df)

        if "is_unusual" not in anomaly_df.columns:
            return {
                "status": "error",
                "message": (
                    "The anomaly detector did not return an 'is_unusual' column."
                ),
            }

        unusual_df = anomaly_df[anomaly_df["is_unusual"] == True]

        results = []
        seen_row_ids = set()
        for idx, row in unusual_df.iterrows():
            row_id = row.get("transaction_id", row.get("_row_id", idx))
            if row_id in seen_row_ids:
                continue
            seen_row_ids.add(row_id)
            results.append(
                {
                    "transaction_id": str(row_id),
                    "description": str(
                        row.get("transaction_description", "")
                    ),
                    "amount": round(float(row.get("amount", 0.0)), 2),
                    "category": str(
                        row.get(
                            "category",
                            row.get("predicted_category", "Unknown"),
                        )
                    ),
                    "reason": str(
                        row.get("anomaly_reason", "Outlier detected")
                    ),
                }
            )

        return {
            "status": "success",
            "unusual_transactions_count": len(results),
            "unusual_transactions": results,
            "currency": info["currencies"][0],
            "warnings": info["warnings"],
        }