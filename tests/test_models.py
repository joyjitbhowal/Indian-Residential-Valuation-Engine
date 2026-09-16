import pytest
import numpy as np
from src.models import ModelTrainer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = X[:, 0] * 100000 + X[:, 1] * 50000 + np.random.randn(100) * 10000
    return X, y


@pytest.fixture
def larger_data():
    np.random.seed(42)
    X = np.random.rand(500, 10)
    y = X[:, 0] * 100000 + X[:, 1] * 50000 + X[:, 2] * 30000 + np.random.randn(500) * 10000
    return X, y


class TestModelTrainer:
    def test_train_and_predict(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer("random_forest")
        trainer.train(X, y)
        preds = trainer.predict(X)
        assert len(preds) == len(y)
        assert all(preds > 0)

    def test_evaluate_returns_metrics(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer("random_forest")
        trainer.train(X, y)
        metrics = trainer.evaluate(X, y, "test")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mse" in metrics
        assert "mape" in metrics

    def test_evaluate_r2_range(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer("random_forest")
        trainer.train(X, y)
        metrics = trainer.evaluate(X, y, "test")
        assert -1 <= metrics["r2"] <= 1

    def test_evaluate_metrics_positive(self, sample_data):
        X, y = sample_data
        trainer = ModelTrainer("random_forest")
        trainer.train(X, y)
        metrics = trainer.evaluate(X, y, "test")
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mse"] >= 0

    def test_invalid_model_name(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelTrainer("invalid_model")

    def test_invalid_model_name_suggests_options(self):
        with pytest.raises(ValueError, match="Choose from"):
            ModelTrainer("nonexistent")

    def test_all_models_available(self):
        names = list(ModelTrainer.MODELS.keys())
        assert "random_forest" in names
        assert len(names) <= 3

    def test_train_with_validation(self, larger_data):
        X, y = larger_data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        trainer = ModelTrainer("random_forest")
        trainer.train(X_train, y_train, X_val, y_val)
        preds = trainer.predict(X_val)
        assert len(preds) == len(X_val)

    def test_model_can_be_saved_and_loaded(self, sample_data, tmp_path):
        X, y = sample_data
        trainer = ModelTrainer("random_forest")
        trainer.train(X, y)
        save_path = str(tmp_path / "test_model.pkl")
        trainer.save(save_path)

        trainer2 = ModelTrainer("random_forest")
        trainer2.load(save_path)
        preds1 = trainer.predict(X[:5])
        preds2 = trainer2.predict(X[:5])
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_multiple_datasets_evaluation(self, larger_data):
        X, y = larger_data
        split_train = int(len(X) * 0.6)
        split_val = int(len(X) * 0.8)
        X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
        y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

        trainer = ModelTrainer("random_forest")
        trainer.train(X_train, y_train, X_val, y_val)

        train_m = trainer.evaluate(X_train, y_train, "train")
        val_m = trainer.evaluate(X_val, y_val, "val")
        test_m = trainer.evaluate(X_test, y_test, "test")

        assert train_m["r2"] >= val_m["r2"] or abs(train_m["r2"] - val_m["r2"]) < 0.1
        assert "train" in trainer.metrics
        assert "val" in trainer.metrics
        assert "test" in trainer.metrics
