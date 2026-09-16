import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from src.scoring import compute_scores, multi_modal_ensemble


class TestComputeScores:
    def test_perfect_model_scores_high(self):
        metrics = {"model_a": {"r2": 1.0, "mae": 1.0, "mape": 1.0}}
        scores = compute_scores(metrics)
        assert scores["model_a"] > 0.9

    def test_poor_model_scores_low(self):
        metrics = {"model_a": {"r2": 0.1, "mae": 1000, "mape": 100}}
        scores = compute_scores(metrics)
        assert scores["model_a"] < 0.3

    def test_multiple_models(self):
        metrics = {
            "xgboost": {"r2": 0.95, "mae": 10, "mape": 5},
            "lightgbm": {"r2": 0.93, "mae": 15, "mape": 6},
        }
        scores = compute_scores(metrics)
        assert len(scores) == 2
        assert scores["xgboost"] > scores["lightgbm"]

    def test_missing_metrics_handled(self):
        metrics = {"model_a": {}}
        scores = compute_scores(metrics)
        assert "model_a" in scores

    def test_negative_r2_clamped_to_zero(self):
        metrics = {"model_a": {"r2": -0.5, "mae": 10, "mape": 5}}
        scores = compute_scores(metrics)
        assert scores["model_a"] > 0

    def test_zero_mae_very_high_score(self):
        metrics = {"model_a": {"r2": 1.0, "mae": 0, "mape": 0}}
        scores = compute_scores(metrics)
        assert scores["model_a"] > 0.9


class TestMultiModalEnsemble:
    def test_weighted_ensemble_calculation(self):
        preds = {"model_a": 100.0, "model_b": 120.0}
        scores = {"model_a": 0.8, "model_b": 0.6}
        result = multi_modal_ensemble(preds, scores)
        assert "weighted_ensemble" in result
        assert "price_range" in result
        assert result["price_range"]["lower"] < result["weighted_ensemble"]
        assert result["price_range"]["upper"] > result["weighted_ensemble"]

    def test_confidence_levels(self):
        preds = {"a": 100.0, "b": 100.0, "c": 100.0}
        scores = {"a": 0.9, "b": 0.9, "c": 0.9}
        result = multi_modal_ensemble(preds, scores)
        assert result["price_range"]["confidence"] == "High"

    def test_medium_confidence(self):
        preds = {"a": 100.0, "b": 150.0, "c": 180.0}
        scores = {"a": 0.9, "b": 0.8, "c": 0.7}
        result = multi_modal_ensemble(preds, scores)
        assert result["price_range"]["confidence"] in ("Medium", "Low")

    def test_model_weights_sum_to_one(self):
        preds = {"a": 100.0, "b": 120.0, "c": 110.0}
        scores = {"a": 0.9, "b": 0.8, "c": 0.7}
        result = multi_modal_ensemble(preds, scores)
        total_weight = sum(result["model_weights"].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_best_model_is_highest_weighted(self):
        preds = {"good": 100.0, "bad": 200.0}
        scores = {"good": 0.95, "bad": 0.3}
        result = multi_modal_ensemble(preds, scores)
        assert result["best_model"]["name"] == "good"

    def test_single_model(self):
        preds = {"only": 50.0}
        scores = {"only": 0.9}
        result = multi_modal_ensemble(preds, scores)
        assert result["weighted_ensemble"] == 50.0

    def test_empty_scores_use_defaults(self):
        preds = {"a": 100.0, "b": 110.0}
        scores = {}
        result = multi_modal_ensemble(preds, scores)
        assert "weighted_ensemble" in result

    def test_mean_median_trimmed_mean_present(self):
        preds = {"a": 100.0, "b": 110.0, "c": 120.0}
        scores = {"a": 0.9, "b": 0.8, "c": 0.7}
        result = multi_modal_ensemble(preds, scores)
        assert "mean" in result
        assert "median" in result
        assert "trimmed_mean" in result
