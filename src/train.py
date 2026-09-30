import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from src.config import config
from src.utils import logger

class ModelTrainer:
    """Trains text classification candidate models using strictly fitted TF-IDF pipeline."""

    def __init__(self, tfidf_params: Dict[str, Any] = None):
        self.tfidf_params = tfidf_params or config.tfidf_params
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.label_encoder = LabelEncoder()
        
    def prepare_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit TF-IDF and LabelEncoder strictly on training set.
        Transform validation and test sets without fitting.
        """
        logger.info("Fitting TF-IDF vectorizer strictly on TRAINING split...")
        X_train = self.vectorizer.fit_transform(train_df["cleaned_description"])
        
        logger.info("Transforming Validation and Test splits using fitted vectorizer...")
        X_val = self.vectorizer.transform(val_df["cleaned_description"])
        X_test = self.vectorizer.transform(test_df["cleaned_description"])
        
        logger.info("Fitting LabelEncoder strictly on TRAINING categories...")
        y_train = self.label_encoder.fit_transform(train_df["category"])
        y_val = self.label_encoder.transform(val_df["category"])
        y_test = self.label_encoder.transform(test_df["category"])
        
        logger.info(f"Feature matrix shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_candidate_models(self) -> Dict[str, Any]:
        """Return candidate classifiers."""
        seed = config.random_seed
        return {
            "Logistic Regression": LogisticRegression(
                C=1.0, max_iter=1000, random_state=seed, solver="lbfgs"
            ),
            "Linear SVM": CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=seed),
                method="sigmoid"
            ),
            "Naive Bayes": MultinomialNB(alpha=0.1)
        }

    def train_and_benchmark(
        self, X_train: Any, y_train: np.ndarray, X_val: Any, y_val: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Train candidate models and return fitted model objects and training times."""
        models = self.get_candidate_models()
        results = {}
        
        for name, clf in models.items():
            logger.info(f"Training candidate model: {name}...")
            start_time = time.time()
            clf.fit(X_train, y_train)
            elapsed_time = round(time.time() - start_time, 3)
            
            results[name] = {
                "model": clf,
                "training_time": elapsed_time
            }
            logger.info(f"Finished training {name} in {elapsed_time:.3f}s")
            
        return results
