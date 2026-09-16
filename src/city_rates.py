import pandas as pd
import json
from pathlib import Path


def build_city_rates(data_path: str = None, output_path: str = None):
    if data_path is None:
        data_path = str(Path(__file__).parent.parent / "india_housing_prices.csv")
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "models" / "city_rates.json")

    df = pd.read_csv(data_path)
    rates = (
        df.groupby(["State", "City"])["Price_per_SqFt"]
        .agg(["mean", "median"])
        .reset_index()
    )
    rates["mean"] = rates["mean"].round(4)
    rates["median"] = rates["median"].round(4)

    lookup = {}
    for _, row in rates.iterrows():
        key = f"{row['State']}|{row['City']}"
        lookup[key] = {"mean": row["mean"], "median": row["median"]}

    overall_median = float(df["Price_per_SqFt"].median())
    lookup["__default__"] = {"mean": round(overall_median, 4), "median": round(overall_median, 4)}

    with open(output_path, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"City rates saved to {output_path} ({len(lookup)-1} cities)")
    return lookup


def get_rate(state: str, city: str, rates: dict = None) -> float:
    if rates is None:
        rates_path = Path(__file__).parent.parent / "models" / "city_rates.json"
        if rates_path.exists():
            with open(rates_path) as f:
                rates = json.load(f)
        else:
            return 0.10

    key = f"{state}|{city}"
    entry = rates.get(key, rates.get("__default__", {"median": 0.10}))
    return entry["median"]


if __name__ == "__main__":
    build_city_rates()
