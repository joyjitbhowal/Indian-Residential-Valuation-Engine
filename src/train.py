import argparse
import joblib
from pathlib import Path
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.utils import logger, save_metrics, timestamp


def train_pipeline(
    model_name: str = "random_forest",
    data_path: str = None,
    output_dir: str = None,
):
    logger.info(f"Training pipeline: model={model_name}")

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

    trainer = ModelTrainer(model_name)
    trainer.train(scaled["X_train"], y_train, scaled["X_val"], y_val)

    train_m = trainer.evaluate(scaled["X_train"], y_train, "train")
    val_m = trainer.evaluate(scaled["X_val"], y_val, "val")
    test_m = trainer.evaluate(scaled["X_test"], y_test, "test")

    ts = timestamp()
    model_path = str(out_dir / f"{model_name}_{ts}.pkl")
    trainer.save(model_path)

    feature_cols = list(df.drop(columns=[preprocessor.TARGET]).columns)

    artifact = {
        "timestamp": ts,
        "model": model_name,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "onehot_categories": preprocessor._onehot_categories,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "metrics": {"train": train_m, "val": val_m, "test": test_m},
    }

    metrics_path = str(out_dir / f"{model_name}_metrics.json")
    save_metrics(artifact, metrics_path)

    scaler_path = out_dir / "scaler.joblib"
    joblib.dump(preprocessor.scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    print(f"\n{'='*50}")
    print(f"  Model:  {model_name}")
    print(f"  Test R²: {test_m['r2']:.4f}")
    print(f"  Test MAE: {test_m['mae']:,.2f}L")
    print(f"  Test MAPE: {test_m['mape']:.1f}%")
    print(f"  Saved: {model_path}")
    print(f"{'='*50}")

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["xgboost", "lightgbm", "random_forest"],
                        help="Model to train")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    train_pipeline(args.model, args.data, args.output)
