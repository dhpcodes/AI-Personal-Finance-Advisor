import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)
from src.config import config
from src.utils import logger, save_csv_report

class ModelEvaluator:
    """Evaluates candidate models, plots confusion matrix, and exports empirical benchmark metrics."""

    def __init__(self, label_encoder: Any):
        self.label_encoder = label_encoder
        self.class_names = list(label_encoder.classes_)

    def evaluate_model(self, model: Any, X: Any, y_true: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics on given split."""
        y_pred = model.predict(X)
        
        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        
        # Per class report
        report_dict = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
        )
        
        return {
            "model_name": model_name,
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class_report": report_dict,
            "y_pred": y_pred
        }

    def generate_benchmark_table(
        self, benchmark_results: Dict[str, Dict[str, Any]], reports_dir: Path = None
    ) -> pd.DataFrame:
        """Create and save comparison table of evaluated models."""
        if reports_dir is None:
            reports_dir = config.reports_dir
            
        rows = []
        for model_name, res in benchmark_results.items():
            eval_metrics = res["metrics"]
            rows.append({
                "model": model_name,
                "accuracy": eval_metrics["accuracy"],
                "macro_precision": eval_metrics["macro_precision"],
                "macro_recall": eval_metrics["macro_recall"],
                "macro_f1": eval_metrics["macro_f1"],
                "weighted_f1": eval_metrics["weighted_f1"],
                "training_time_sec": res["training_time"]
            })
            
        df_bench = pd.DataFrame(rows)
        save_csv_report(df_bench, reports_dir / "model_evaluation.csv")
        logger.info("Saved model evaluation benchmark table to reports/model_evaluation.csv")
        return df_bench

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, reports_dir: Path = None
    ) -> None:
        """Plot and save confusion matrix visualization."""
        if reports_dir is None:
            reports_dir = config.reports_dir
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f"Confusion Matrix — {model_name}")
        plt.xlabel("Predicted Category")
        plt.ylabel("True Category")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = reports_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {save_path}")

    @staticmethod
    def save_final_model_artifact(
        vectorizer: Any,
        label_encoder: Any,
        model: Any,
        model_name: str,
        metrics: Dict[str, Any],
        save_path: Path = None
    ) -> Path:
        """Save vectorizer, label encoder, classifier, and metadata bundle."""
        if save_path is None:
            save_path = config.final_model_path
            
        artifact = {
            "version": "1.0.0",
            "model_name": model_name,
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
            "model": model,
            "classes": list(label_encoder.classes_),
            "metrics": metrics
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, save_path)
        logger.info(f"Saved winning final model artifact ({model_name}) to {save_path}")
        return save_path
