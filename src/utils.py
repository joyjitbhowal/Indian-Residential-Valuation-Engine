import os
import json
import logging
from datetime import datetime

log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "app.log")),
    ],
)

logger = logging.getLogger("house_price_prediction")


def save_metrics(metrics: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {filepath}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
