import pandas as pd
from pathlib import Path
from src.config import config
from src.data_loader import DataLoader
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import logger

def main():
    logger.info("--- Starting Phases 5, 6 & 7: Model Baseline, Benchmark & Selection ---")
    
    # 1. Load Processed Splits
    processed_dir = config.raw_data_path.parent.parent / "processed"
    train_df = DataLoader.load_parquet(processed_dir / "train.parquet")
    val_df = DataLoader.load_parquet(processed_dir / "val.parquet")
    test_df = DataLoader.load_parquet(processed_dir / "test.parquet")
    
    # 2. Fit TF-IDF & LabelEncoder ONLY on Training split
    trainer = ModelTrainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )
    
    # 3. Train & Benchmark Candidate Models on Train set
    benchmark_data = trainer.train_and_benchmark(X_train, y_train, X_val, y_val)
    
    # 4. Evaluate Candidates on Test set
    evaluator = ModelEvaluator(trainer.label_encoder)
    
    final_benchmark = {}
    best_model_name = None
    best_f1 = -1.0
    
    for name, data in benchmark_data.items():
        clf = data["model"]
        metrics = evaluator.evaluate_model(clf, X_test, y_test, model_name=name)
        
        final_benchmark[name] = {
            "metrics": metrics,
            "training_time": data["training_time"],
            "model": clf
        }
        
        logger.info(
            f"Candidate [{name}] -> Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}"
        )
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_model_name = name
            
    # 5. Export Benchmark CSV
    evaluator.generate_benchmark_table(final_benchmark)
    
    # 6. Plot Confusion Matrix for Best Model
    best_model_obj = final_benchmark[best_model_name]["model"]
    best_y_pred = final_benchmark[best_model_name]["metrics"]["y_pred"]
    evaluator.plot_confusion_matrix(y_test, best_y_pred, best_model_name)
    
    # 7. Save Winning Model Artifact
    best_metrics = final_benchmark[best_model_name]["metrics"]
    evaluator.save_final_model_artifact(
        trainer.vectorizer,
        trainer.label_encoder,
        best_model_obj,
        best_model_name,
        best_metrics
    )
    
    logger.info(f"--- Final Selected Model: {best_model_name} (Macro F1 = {best_f1:.4f}) ---")
    logger.info("--- Phases 5, 6 & 7 Completed Successfully! ---")

if __name__ == "__main__":
    main()
