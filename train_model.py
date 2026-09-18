from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_DIR = Path("training_data")
MODEL_PATH = Path("models/packing_model.joblib")

FEATURES = [
    "same_type",
    "both_doors",
    "both_windows",
    "width_difference",
    "height_difference",
    "weight_difference",
    "combined_weight",
    "maximum_width",
    "maximum_height",
    "same_glass_mode",
]


def normalise_type(value):
    return str(value).strip().lower()


def load_units(file_path):
    constructions = pd.read_excel(file_path, sheet_name="Constructions")
    actual = pd.read_excel(file_path, sheet_name="Actual Packing")

    units = []
    for _, row in constructions.iterrows():
        project = str(row["Project"]).strip()
        item = str(row["Item"]).strip()
        qty = int(row["Qty"])

        packing_rows = actual[
            (actual["Project"].astype(str).str.strip() == project)
            & (actual["Item"].astype(str).str.strip() == item)
        ]

        pallets = []
        for _, packing_row in packing_rows.iterrows():
            pallets.extend(
                [str(packing_row["Pallet"]).strip()]
                * int(packing_row["Qty on pallet"])
            )

        if len(pallets) != qty:
            raise ValueError(
                f"{file_path.name}: {project}/{item} has Qty={qty}, "
                f"but Actual Packing contains {len(pallets)} units"
            )

        unit_weight = float(row["Frame weight (kg)"]) + float(
            row["Glass weight (kg)"]
        )

        for unit_number, pallet in enumerate(pallets, start=1):
            units.append(
                {
                    "project": project,
                    "unit_id": f"{item}-{unit_number}",
                    "type": normalise_type(row["Type"]),
                    "width": float(row["Width (mm)"]),
                    "height": float(row["Height (mm)"]),
                    "weight": unit_weight,
                    "glass_mode": str(row["Glass mode"]).strip().lower(),
                    "pallet": pallet,
                }
            )

    return units


def pair_features(first, second):
    return {
        "same_type": int(first["type"] == second["type"]),
        "both_doors": int(first["type"] == second["type"] == "door"),
        "both_windows": int(first["type"] == second["type"] == "window"),
        "width_difference": abs(first["width"] - second["width"]),
        "height_difference": abs(first["height"] - second["height"]),
        "weight_difference": abs(first["weight"] - second["weight"]),
        "combined_weight": first["weight"] + second["weight"],
        "maximum_width": max(first["width"], second["width"]),
        "maximum_height": max(first["height"], second["height"]),
        "same_glass_mode": int(first["glass_mode"] == second["glass_mode"]),
    }


def build_training_data(files):
    rows = []
    labels = []
    projects = set()

    for file_path in files:
        units = load_units(file_path)
        projects.update(unit["project"] for unit in units)

        for first, second in combinations(units, 2):
            if first["project"] != second["project"]:
                continue
            rows.append(pair_features(first, second))
            labels.append(int(first["pallet"] == second["pallet"]))

    return pd.DataFrame(rows, columns=FEATURES), pd.Series(labels), projects


def main():
    files = sorted(DATA_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No .xlsx files found in training_data")

    x, y, projects = build_training_data(files)
    if x.empty or y.nunique() < 2:
        raise ValueError("Training data must contain both same-pallet and different-pallet pairs")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "projects": len(projects),
            "training_pairs": len(x),
        },
        MODEL_PATH,
    )

    training_accuracy = accuracy_score(y, model.predict(x))
    print(f"Projects: {len(projects)}")
    print(f"Construction pairs: {len(x)}")
    print(f"Training accuracy: {training_accuracy:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    if len(projects) < 3:
        print("Warning: this is only a pipeline demonstration; more projects are needed for validation.")


if __name__ == "__main__":
    main()
