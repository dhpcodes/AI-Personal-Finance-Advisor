import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
from src.config import config
from src.preprocessing import TextPreprocessor
from src.utils import logger

class TransactionPredictor:
    """Production inference pipeline for transaction categorization."""

    def __init__(self, model_path: Path = None):
        self.model_path = model_path or config.final_model_path
        self._load_artifact()
        self.preprocessor = TextPreprocessor()

    def _load_artifact(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Final model artifact not found at {self.model_path}")
        logger.info(f"Loading prediction model artifact from {self.model_path}...")
        artifact = joblib.load(self.model_path)
        self.vectorizer = artifact["vectorizer"]
        self.label_encoder = artifact["label_encoder"]
        self.model = artifact["model"]
        self.classes = artifact["classes"]
        self.model_name = artifact.get("model_name", "Classification Model")

    def predict_single(self, transaction_description: str, top_k: int = 3) -> Dict[str, Any]:
        """Predict category, confidence, and top-k probabilities for a single transaction string."""
        cleaned = self.preprocessor.clean_text(transaction_description)
        if not cleaned:
            return {
                "transaction_description": transaction_description,
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "top_k": []
            }

        X_vec = self.vectorizer.transform([cleaned])
        
        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
        elif hasattr(self.model, "decision_function"):
            df_scores = self.model.decision_function(X_vec)[0]
            # Softmax calibration fallback
            exp_scores = np.exp(df_scores - np.max(df_scores))
            probs = exp_scores / np.sum(exp_scores)
        else:
            pred_idx = self.model.predict(X_vec)[0]
            probs = np.zeros(len(self.classes))
            probs[pred_idx] = 1.0

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [
            {"category": self.classes[i], "probability": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        best_idx = top_indices[0]
        return {
            "transaction_description": transaction_description,
            "cleaned_description": cleaned,
            "predicted_category": self.classes[best_idx],
            "confidence": round(float(probs[best_idx]), 4),
            "top_k": top_k_list,
            "model_used": self.model_name
        }

    def predict_batch(self, df: pd.DataFrame, desc_col: str = "transaction_description") -> pd.DataFrame:
        """Batch predict categories and confidence for a DataFrame."""
        df_out = df.copy()
        raw_texts = df_out[desc_col].fillna("").astype(str)
        cleaned_texts = [self.preprocessor.clean_text(t) for t in raw_texts]
        
        X_vec = self.vectorizer.transform(cleaned_texts)
        
        if hasattr(self.model, "predict_proba"):
            probs_matrix = self.model.predict_proba(X_vec)
        else:
            preds = self.model.predict(X_vec)
            probs_matrix = np.zeros((len(preds), len(self.classes)))
            for row_idx, p in enumerate(preds):
                probs_matrix[row_idx, p] = 1.0

        best_indices = np.argmax(probs_matrix, axis=1)
        confidences = np.max(probs_matrix, axis=1)
        
        df_out["predicted_category"] = [self.classes[i] for i in best_indices]
        df_out["confidence"] = np.round(confidences, 4)
        return df_out
