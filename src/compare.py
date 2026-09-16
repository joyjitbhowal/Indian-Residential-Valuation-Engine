import argparse
import json
import joblib
from pathlib import Path
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.scoring import compute_scores, multi_modal_ensemble
from src.utils import logger, timestamp


def run_comparison(data_path: str = None, output_dir: str = None):
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON — GPU-Accelerated Multi-Modal Ensemble")
    logger.info("=" * 60)

    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load()
    df = preprocessor.clean()
    preprocessor.fit_encode_categories(df)
    df = preprocessor.encode_categoricals(df)

    fe = FeatureEngineer()
    df = fe.fit_transform(df)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split(df)
    scaled = preprocessor.scale(X_train, X_val, X_test)

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "models")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(df.drop(columns=[preprocessor.TARGET]).columns)
    results = {}
    trainers = {}
    ts = timestamp()

    for model_name in ModelTrainer.MODELS:
        logger.info(f"\n--- Training {model_name} ---")
        trainer = ModelTrainer(model_name)
        trainer.train(scaled["X_train"], y_train, scaled["X_val"], y_val)

        train_m = trainer.evaluate(scaled["X_train"], y_train, "train")
        val_m = trainer.evaluate(scaled["X_val"], y_val, "val")
        test_m = trainer.evaluate(scaled["X_test"], y_test, "test")

        model_path = str(out_dir / f"{model_name}_{ts}.joblib")
        trainer.save(model_path)
        trainers[model_name] = trainer
        results[model_name] = {"train": train_m, "val": val_m, "test": test_m}
        print(f"  {model_name:20s}  Test R²: {test_m['r2']:.4f}  MAE: {test_m['mae']:,.2f}L  MAPE: {test_m['mape']:.1f}%")

    test_metrics = {name: results[name]["test"] for name in results}
    scores = compute_scores(test_metrics)

    ensemble_preds = {}
    for name in trainers:
        ensemble_preds[name] = float(trainers[name].predict(scaled["X_test"][:1])[0])
    ensemble_result = multi_modal_ensemble(ensemble_preds, scores)

    comparison = {
        "timestamp": ts,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "onehot_categories": preprocessor._onehot_categories,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "models": results,
        "model_scores": scores,
        "ensemble_sample": ensemble_result,
    }

    comparison_path = str(out_dir / "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison saved to {comparison_path}")

    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(preprocessor.scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    best = max(scores, key=lambda m: scores[m])
    print(f"\n{'='*60}")
    print(f"  BEST MODEL: {best} (Score: {scores[best]:.4f})")
    print(f"  ENSEMBLE: {ensemble_result['weighted_ensemble']}L | RANGE: [{ensemble_result['price_range']['lower']}L - {ensemble_result['price_range']['upper']}L] ({ensemble_result['price_range']['confidence']})")
    print(f"{'='*60}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models with multi-modal scoring")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    run_comparison(args.data, args.output)
