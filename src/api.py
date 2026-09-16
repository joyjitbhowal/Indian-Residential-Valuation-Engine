import os
import time
import threading
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.predict import HousePricePredictor
from src.city_rates import get_rate
from src.utils import logger

load_dotenv()

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "templates"),
    static_folder=str(Path(__file__).parent.parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 1_048_576))

CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*").split(",")}})

predictor = None
_states_cache = None
_states_cache_time = 0
cities_cache = {}
cities_cache_time = {}

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "india_housing_prices.csv"


class RateLimiter:
    def __init__(self, requests: int = 60, window: int = 60):
        self.requests = requests
        self.window = window
        self._clients = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            history = self._clients.get(key, [])
            history = [t for t in history if now - t < self.window]
            if len(history) >= self.requests:
                return False
            history.append(now)
            self._clients[key] = history
            return True


_limiter = RateLimiter(requests=int(os.getenv("RATE_LIMIT_REQUESTS", "60")),
                       window=int(os.getenv("RATE_LIMIT_WINDOW", "60")))


def rate_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        key = request.remote_addr or "unknown"
        if not _limiter.is_allowed(key):
            return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
        return f(*args, **kwargs)
    return wrapper


def load_predictor():
    global predictor
    predictor = HousePricePredictor()
    return predictor.load()


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


@app.route("/health", methods=["GET"])
def health():
    status = {
        "status": "ok",
        "models_loaded": list(predictor.models.keys()) if predictor else [],
        "model_count": len(predictor.models) if predictor else 0,
        "feature_count": len(predictor.feature_columns) if predictor else 0,
        "gpu_accelerated": "lightgbm" in (predictor.models if predictor else {}),
    }
    return jsonify(status)


@app.route("/predict", methods=["POST"])
@rate_limit
def predict_api():
    start_time = time.time()

    if not predictor or not predictor.models:
        return jsonify({"error": "No models loaded. Run compare.py first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON. Send Content-Type: application/json"}), 400

    features = data.get("features", data)
    if not isinstance(features, dict):
        return jsonify({"error": "Features must be a JSON object"}), 400

    errors = predictor.validate_input(features)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    try:
        result = predictor.predict(features)
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "An unexpected error occurred during prediction."}), 500

    elapsed = round((time.time() - start_time) * 1000, 1)
    result["latency_ms"] = elapsed
    return jsonify(result)


@app.route("/api/comparison", methods=["GET"])
def get_comparison():
    if not predictor:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify(predictor.comparison_data)


@app.route("/api/scores", methods=["GET"])
def get_scores():
    if not predictor:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify(predictor.model_scores)


@app.route("/api/rate", methods=["GET"])
def get_rate_api():
    state = request.args.get("state", "").strip()
    city = request.args.get("city", "").strip()
    if not state or not city:
        return jsonify({"error": "Both 'state' and 'city' parameters are required"}), 400
    rate = get_rate(state, city, predictor.city_rates if predictor else None)
    return jsonify({"Price_per_SqFt": rate, "state": state, "city": city})


@app.route("/api/states", methods=["GET"])
def get_states():
    global _states_cache, _states_cache_time
    now = time.time()
    if _states_cache is None or now - _states_cache_time > 300:
        try:
            df = _load_data()
            _states_cache = sorted(df["State"].unique().tolist())
            _states_cache_time = now
        except Exception:
            logger.exception("Failed to load states")
            return jsonify({"error": "Failed to load states"}), 500
    return jsonify({"states": _states_cache, "count": len(_states_cache)})


@app.route("/api/cities", methods=["GET"])
def get_cities():
    state = request.args.get("state", "").strip()
    if not state:
        return jsonify({"error": "State parameter is required"}), 400

    now = time.time()
    cache_key = state
    if cache_key in cities_cache and now - cities_cache_time.get(cache_key, 0) < 300:
        return jsonify({"cities": cities_cache[cache_key], "state": state})

    try:
        df = _load_data()
        state_df = df[df["State"] == state]
        if state_df.empty:
            return jsonify({"error": f"No data found for state: {state}"}), 404
        cities = sorted(state_df["City"].unique().tolist())
        cities_cache[cache_key] = cities
        cities_cache_time[cache_key] = now
        return jsonify({"cities": cities, "state": state, "count": len(cities)})
    except Exception:
        logger.exception("Failed to load cities")
        return jsonify({"error": "Failed to load cities"}), 500


@app.route("/models", methods=["GET"])
def list_models():
    if not predictor:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify({
        "models": list(predictor.models.keys()),
        "count": len(predictor.models),
        "gpu": "lightgbm" in predictor.models,
        "feature_count": len(predictor.feature_columns),
    })


def _load_data():
    if DATA_PATH.exists():
        import pandas as pd
        return pd.read_csv(DATA_PATH)
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")


try:
    load_predictor()
except Exception as e:
    logger.warning(f"Could not load models at startup: {e}")

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting API on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
