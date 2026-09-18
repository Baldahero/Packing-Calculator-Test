"""ML compatibility score used by the constrained packing optimiser."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "models" / "packing_model.joblib"


def _number(row, column):
    value = row.get(column, 0)
    return 0.0 if pd.isna(value) else float(value)


def _type(row):
    return str(row.get("Type", "")).strip().lower()


def _weight(row):
    frame = _number(row, "Frame weight (kg)")
    glass = _number(row, "Glass weight (kg)")
    return frame + glass if str(row.get("Glass mode", "Glazed")).strip().lower() == "glazed" else frame


def pair_features(first, second):
    first_type = _type(first)
    second_type = _type(second)
    first_weight = _weight(first)
    second_weight = _weight(second)

    return {
        "same_type": int(first_type == second_type),
        "both_doors": int(first_type == second_type == "door"),
        "both_windows": int(first_type == second_type == "window"),
        "width_difference": abs(_number(first, "Width (mm)") - _number(second, "Width (mm)")),
        "height_difference": abs(_number(first, "Height (mm)") - _number(second, "Height (mm)")),
        "weight_difference": abs(first_weight - second_weight),
        "combined_weight": first_weight + second_weight,
        "maximum_width": max(_number(first, "Width (mm)"), _number(second, "Width (mm)")),
        "maximum_height": max(_number(first, "Height (mm)"), _number(second, "Height (mm)")),
        "same_glass_mode": int(
            str(first.get("Glass mode", "Glazed")).strip().lower()
            == str(second.get("Glass mode", "Glazed")).strip().lower()
        ),
    }


@lru_cache(maxsize=1)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def model_status():
    bundle = load_model_bundle()
    if bundle is None:
        return {"loaded": False, "projects": 0, "training_pairs": 0}
    return {
        "loaded": True,
        "projects": int(bundle.get("projects", 0)),
        "training_pairs": int(bundle.get("training_pairs", 0)),
    }


def pairing_cost(first, second):
    """Return a low cost for a pair the model prefers on the same pallet."""
    bundle = load_model_bundle()
    if bundle is None:
        return (
            abs(_number(first, "Width (mm)") - _number(second, "Width (mm)"))
            + abs(_number(first, "Height (mm)") - _number(second, "Height (mm)"))
        ) / 1000.0

    features = bundle["features"]
    row = pd.DataFrame([pair_features(first, second)], columns=features)
    model = bundle["model"]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    same_pallet_probability = float(probabilities[classes.index(1)])
    return 1.0 - same_pallet_probability
