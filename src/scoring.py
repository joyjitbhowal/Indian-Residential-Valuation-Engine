import numpy as np
from src.utils import logger


def compute_scores(metrics: dict) -> dict:
    scores = {}
    for name, m in metrics.items():
        r2 = max(m.get("r2", 0), 0)
        mae = m.get("mae", 1e6)
        mape = m.get("mape", 100)
        mae_score = 1 / (1 + mae / 100)
        mape_score = 1 / (1 + mape / 100)
        composite = (r2 + mae_score + mape_score) / 3
        scores[name] = round(composite, 4)
    return scores


def multi_modal_ensemble(predictions: dict, model_scores: dict) -> dict:
    names = list(predictions.keys())
    prices = np.array([predictions[n] for n in names])
    weights = np.array([max(model_scores.get(n, 0), 0.01) for n in names])
    weights = weights / weights.sum()

    weighted_price = float(np.average(prices, weights=weights))
    mean_price = float(np.mean(prices))
    median_price = float(np.median(prices))

    sorted_prices = np.sort(prices)
    n = len(sorted_prices)
    if n >= 3:
        trimmed = sorted_prices[1:-1]
        trimmed_mean = float(np.mean(trimmed))
    else:
        trimmed_mean = mean_price

    std_price = float(np.std(prices))
    lower = round(max(weighted_price - 1.5 * std_price, 0), 2)
    upper = round(weighted_price + 1.5 * std_price, 2)

    best_idx = int(np.argmax(weights))
    best_model = names[best_idx]

    return {
        "weighted_ensemble": round(weighted_price, 2),
        "mean": round(mean_price, 2),
        "median": round(median_price, 2),
        "trimmed_mean": round(trimmed_mean, 2),
        "best_model": {"name": best_model, "price": round(prices[best_idx], 2)},
        "price_range": {
            "lower": lower,
            "upper": upper,
            "spread": round(upper - lower, 2),
            "confidence": "High" if std_price < 20 else "Medium" if std_price < 50 else "Low",
        },
        "model_weights": {n: round(float(w), 3) for n, w in zip(names, weights)},
    }
