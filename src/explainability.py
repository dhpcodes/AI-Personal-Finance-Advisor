import numpy as np
from typing import Dict, Any, List
from src.predict import TransactionPredictor

class ModelExplainer:
    """Provides model feature importance and term weight explanations for classification decisions."""

    def __init__(self, predictor: TransactionPredictor = None):
        self.predictor = predictor or TransactionPredictor()
        
    def explain_prediction(self, transaction_description: str) -> Dict[str, Any]:
        """Extract top positive n-grams contributing to predicted category."""
        pred_res = self.predictor.predict_single(transaction_description)
        cleaned = pred_res.get("cleaned_description", "")
        predicted_cat = pred_res["predicted_category"]
        
        if not cleaned or predicted_cat == "Unknown":
            return {
                **pred_res,
                "important_features": [],
                "explanation_summary": "Transaction description was empty or invalid."
            }

        # Transform single text to TF-IDF vector
        X_vec = self.predictor.vectorizer.transform([cleaned])
        feature_names = np.array(self.predictor.vectorizer.get_feature_names_out())
        
        # Get non-zero feature indices for this transaction
        nonzero_indices = X_vec.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return {
                **pred_res,
                "important_features": [],
                "explanation_summary": "No known vocabulary n-grams found in transaction text."
            }

        # Determine class index
        class_idx = list(self.predictor.classes).index(predicted_cat)
        
        # Retrieve base estimator weights if CalibratedClassifierCV
        model = self.predictor.model
        if hasattr(model, "calibrated_classifiers_"):
            # Average coefficients across calibrated classifiers
            coefs = np.mean(
                [clf.estimator.coef_[class_idx] for clf in model.calibrated_classifiers_],
                axis=0
            )
        elif hasattr(model, "coef_"):
            coefs = model.coef_[class_idx]
        else:
            coefs = np.ones(len(feature_names))

        # Calculate feature contributions (TF-IDF value * weight)
        tfidf_vals = X_vec[0, nonzero_indices].toarray()[0]
        feature_weights = coefs[nonzero_indices] * tfidf_vals
        
        # Sort features by highest positive contribution
        sorted_indices = np.argsort(feature_weights)[::-1]
        
        important_features = []
        for idx in sorted_indices:
            feat_name = feature_names[nonzero_indices[idx]]
            weight = round(float(feature_weights[idx]), 4)
            important_features.append({"feature": feat_name, "contribution": weight})
            
        top_words = [f["feature"] for f in important_features[:3]]
        explanation_summary = (
            f"Prediction '{predicted_cat}' was primarily driven by terms: "
            f"{', '.join(top_words)} (Model-generated feature attribution)."
        )
        
        return {
            **pred_res,
            "important_features": important_features,
            "explanation_summary": explanation_summary
        }
