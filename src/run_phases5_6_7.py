
import pandas as pd
from pathlib import Path
from src.config import config
from src.data_loader import DataLoader
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import logger


def main():
    logger.info(
        "--- Starting Phases 5, 6 & 7: Model Baseline, Benchmark & Selection ---"
    )

    # 1. Load Processed Splits
    processed_dir = config.raw_data_path.parent.parent / "processed"

    train_df = DataLoader.load_parquet(processed_dir / "train.parquet")
    val_df = DataLoader.load_parquet(processed_dir / "val.parquet")
    test_df = DataLoader.load_parquet(processed_dir / "test.parquet")

    # 2. Fit TF-IDF and LabelEncoder ONLY on Training Split
    trainer = ModelTrainer()

    X_train, X_val, X_test, y_train, y_val, y_test = (
        trainer.prepare_features(train_df, val_df, test_df)
    )

    # 3. Train Candidate Models Using Training Data
    benchmark_data = trainer.train_and_benchmark(
        X_train, y_train, X_val, y_val
    )

    # 4. Evaluate Candidates on Validation Data
    # Use validation Macro F1 to select the winning model.
    evaluator = ModelEvaluator(trainer.label_encoder)

    final_benchmark = {}
    best_model_name = None
    best_f1 = -1.0

    for name, data in benchmark_data.items():
        clf = data["model"]

        metrics = evaluator.evaluate_model(
            clf, X_val, y_val, model_name=name
        )

        final_benchmark[name] = {
            "metrics": metrics,
            "training_time": data["training_time"],
            "model": clf,
        }

        logger.info(
            f"Validation [{name}] -> "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Weighted F1: {metrics['weighted_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_model_name = name

    if best_model_name is None:
        raise RuntimeError(
            "No candidate model was available for selection."
        )

    # 5. Export Candidate Benchmark CSV
    # This benchmark contains validation metrics, not test metrics.
    evaluator.generate_benchmark_table(final_benchmark)

    # 6. Evaluate the Selected Model on the Test Set ONCE
    best_model_obj = final_benchmark[best_model_name]["model"]

    test_metrics = evaluator.evaluate_model(
        best_model_obj,
        X_test,
        y_test,
        model_name=best_model_name,
    )

    logger.info(
        f"Final Test [{best_model_name}] -> "
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | "
        f"Weighted F1: {test_metrics['weighted_f1']:.4f}"
    )

    # 7. Plot Confusion Matrix Using Test Predictions
    evaluator.plot_confusion_matrix(
        y_test,
        test_metrics["y_pred"],
        best_model_name,
    )

    # 8. Save Winning Model Artifact With Test Metrics
    evaluator.save_final_model_artifact(
        trainer.vectorizer,
        trainer.label_encoder,
        best_model_obj,
        best_model_name,
        test_metrics,
    )

    logger.info(
        f"--- Selected Model: {best_model_name} "
        f"(Validation Macro F1 = {best_f1:.4f}) ---"
    )
    logger.info("--- Phases 5, 6 & 7 Completed Successfully! ---")


if __name__ == "__main__":
    main()

