import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

_LGB_GPU_CACHED = None


def _check_gpu_lightgbm():
    global _LGB_GPU_CACHED
    if _LGB_GPU_CACHED is not None:
        return _LGB_GPU_CACHED
    try:
        X_t = np.random.rand(10, 3)
        y_t = np.random.rand(10)
        model = lgb.LGBMRegressor(device='gpu', n_estimators=2, verbose=-1)
        model.fit(X_t, y_t)
        _LGB_GPU_CACHED = True
    except Exception:
        _LGB_GPU_CACHED = False
    return _LGB_GPU_CACHED


LGB_GPU = False


class ModelTrainer:
    MODELS = {}

    if XGB_AVAILABLE:
        MODELS["xgboost"] = xgb.XGBRegressor(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=3, gamma=0.1,
            early_stopping_rounds=50,
            tree_method="hist", random_state=42,
        )

    if LGB_AVAILABLE:
        lgb_params = dict(
            n_estimators=2000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=10, num_leaves=63,
            verbose=-1, random_state=42,
        )
        MODELS["lightgbm"] = lgb.LGBMRegressor(**lgb_params)

    MODELS["random_forest"] = RandomForestRegressor(
        n_estimators=300, max_depth=25, min_samples_split=5,
        min_samples_leaf=2, max_features=1.0,
        n_jobs=-1, random_state=42,
    )

    def __init__(self, model_name: str = "random_forest"):
        if model_name not in self.MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from {list(self.MODELS.keys())}"
            )
        self.model_name = model_name
        self.model = self.MODELS[model_name]
        self.metrics = {}
        self._best_iteration = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        global LGB_GPU
        logger.info(f"Training {self.model_name}...")

        if not LGB_GPU and self.model_name == "lightgbm":
            LGB_GPU = _check_gpu_lightgbm()
            if LGB_GPU:
                logger.info("LightGBM GPU enabled")
                self.model.set_params(device='gpu', gpu_platform_id=0, gpu_device_id=0)

        if X_val is not None and y_val is not None:
            if self.model_name in ("xgboost",):
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                if hasattr(self.model, 'best_iteration') and self.model.best_iteration:
                    self._best_iteration = self.model.best_iteration
                    logger.info(f"Early stopping: {self._best_iteration} trees")

            elif self.model_name == "lightgbm":
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50)],
                )
                if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_:
                    self._best_iteration = self.model.best_iteration_
                    logger.info(f"Early stopping: {self._best_iteration} trees")
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        logger.info("Training complete")
        return self

    def evaluate(self, X, y, dataset_name: str = "val"):
        predictions = self.model.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-8))) * 100

        metrics = {
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
        }
        self.metrics[dataset_name] = metrics
        logger.info(
            f"{dataset_name} — MAE: {mae:,.2f}L, RMSE: {rmse:,.2f}L, R²: {r2:.4f}, MAPE: {mape:.1f}%"
        )
        return metrics

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str = None):
        if path is None:
            path = str(
                Path(__file__).parent.parent
                / "models"
                / f"{self.model_name}.joblib"
            )
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return self
