import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.neighbors import KNeighborsClassifier
from src.predict import TransactionPredictor
from src.preprocessing import TextPreprocessor
from src.database import db_manager
from src.utils import logger

class AdaptiveNLPClassifier:
    """
    Multi-level Personalization Layer over Global Classifier.
    Level 1: Global Linear SVM (Fallback)
    Level 2: User Correction Memory (Exact / Keyword mapping)
    Level 3: Lightweight User k-NN / Prototype Model (when >= 5 corrections exist)
    """

    def __init__(self, global_predictor: TransactionPredictor = None):
        self.global_predictor = global_predictor or TransactionPredictor()
        self.preprocessor = TextPreprocessor()

    def record_user_correction(
        self,
        user_id: str,
        description: str,
        original_prediction: str,
        corrected_category: str,
        transaction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record user correction feedback and update personalization memory."""
        cleaned = self.preprocessor.clean_text(description)
        fb_id = db_manager.record_feedback(
            user_id=user_id,
            transaction_id=transaction_id,
            description=cleaned,
            original_prediction=original_prediction,
            corrected_category=corrected_category
        )
        logger.info(f"Adaptive NLP learned correction for user '{user_id}': '{cleaned}' -> '{corrected_category}'")
        return {
            "feedback_id": fb_id,
            "status": "success",
            "message": f"Learned personalized rule: '{cleaned}' will map to '{corrected_category}'."
        }

    def predict_personalized(
        self, user_id: str, transaction_description: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict category using 3-Level Adaptive Architecture.
        """
        cleaned = self.preprocessor.clean_text(transaction_description)
        if not cleaned:
            return self.global_predictor.predict_single(transaction_description, top_k=top_k)

        # 1. Level 1: Obtain Global Baseline Prediction
        global_res = self.global_predictor.predict_single(transaction_description, top_k=top_k)
        
        # Fetch user feedback history
        feedback_df = db_manager.get_user_feedback(user_id)
        if len(feedback_df) == 0:
            global_res["is_personalized"] = False
            global_res["personalization_source"] = "Global Linear SVM Baseline"
            global_res["personalization_explanation"] = "No user feedback available yet; using global model baseline."
            return global_res

        # 2. Level 2: Exact or Direct Correction Memory Match
        feedback_df["cleaned_desc"] = [
            self.preprocessor.clean_text(d) for d in feedback_df["description"]
        ]
        exact_matches = feedback_df[feedback_df["cleaned_desc"] == cleaned]
        if len(exact_matches) > 0:
            most_recent_correction = exact_matches.iloc[0]["corrected_category"]
            return {
                "transaction_description": transaction_description,
                "cleaned_description": cleaned,
                "predicted_category": most_recent_correction,
                "confidence": 1.0,
                "is_personalized": True,
                "personalization_source": "User Exact Correction Memory (Level 2)",
                "personalization_explanation": f"Directly matched your previous correction for '{cleaned}'.",
                "top_k": [{"category": most_recent_correction, "probability": 1.0}],
                "global_baseline_prediction": global_res["predicted_category"]
            }

        # 3. Level 3: Lightweight User k-NN Model if sufficient feedback (>= 5 corrections)
        if len(feedback_df) >= 5:
            try:
                fb_texts = feedback_df["cleaned_desc"].tolist()
                fb_labels = feedback_df["corrected_category"].tolist()
                
                # Transform feedback texts using vectorizer
                X_fb = self.global_predictor.vectorizer.transform(fb_texts)
                X_curr = self.global_predictor.vectorizer.transform([cleaned])
                
                # Fit 3-NN classifier on user feedback vectors
                knn = KNeighborsClassifier(n_neighbors=min(3, len(fb_texts)), metric="cosine")
                knn.fit(X_fb, fb_labels)
                
                knn_pred = knn.predict(X_curr)[0]
                knn_probs = knn.predict_proba(X_curr)[0]
                max_prob = float(np.max(knn_probs))
                
                # Only override global model if k-NN cosine similarity confidence > 0.6
                if max_prob >= 0.6 and knn_pred != global_res["predicted_category"]:
                    return {
                        "transaction_description": transaction_description,
                        "cleaned_description": cleaned,
                        "predicted_category": knn_pred,
                        "confidence": round(max_prob, 4),
                        "is_personalized": True,
                        "personalization_source": "User Adaptive k-NN Model (Level 3)",
                        "personalization_explanation": f"Personalized k-NN learned from {len(feedback_df)} corrections.",
                        "top_k": [{"category": knn_pred, "probability": round(max_prob, 4)}],
                        "global_baseline_prediction": global_res["predicted_category"]
                    }
            except Exception as ex:
                logger.warning(f"Level 3 k-NN evaluation skipped: {ex}")

        # Default Level 1 Fallback
        global_res["is_personalized"] = False
        global_res["personalization_source"] = "Global Linear SVM Baseline"
        global_res["personalization_explanation"] = "Global baseline prediction retained (insufficient user override confidence)."
        return global_res
