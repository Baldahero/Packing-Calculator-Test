import unittest

import pandas as pd

from ml_packing import (
    calculate_rule_result,
    generate_training_data,
    normalize_input,
    predict_batch,
    train_models,
    validate_input,
)


class MLPackingTests(unittest.TestCase):
    def test_simple_window_rule_result(self):
        row = {
            "Item": "W-01",
            "Type": "Window",
            "Configuration": "Openable",
            "Width (mm)": 1400,
            "Height (mm)": 1800,
            "Qty": 12,
            "Frame weight (kg)": 70,
            "Glass weight (kg)": 55,
            "Packing thickness (mm)": 75,
            "Handles installed": "Yes",
            "Glass mode": "Glazed",
        }
        result = calculate_rule_result(row)
        self.assertEqual(result["Rule units/pallet"], 6)
        self.assertEqual(result["Rule pallets"], 2)
        self.assertEqual(result["Rule pallet length (mm)"], 1500)
        self.assertEqual(result["Rule total LDM"], 1.5)
        self.assertEqual(result["Rule manual review"], "No")

    def test_door_handles_use_500_mm_total_allowance(self):
        row = {
            "Item": "D-01", "Type": "Door", "Configuration": "Single",
            "Width (mm)": 1000, "Height (mm)": 2300, "Qty": 6,
            "Frame weight (kg)": 95, "Glass weight (kg)": 65,
            "Packing thickness (mm)": 75, "Handles installed": "Yes",
            "Glass mode": "Glazed",
        }
        result = calculate_rule_result(row)
        self.assertEqual(result["Rule units/pallet"], 6)
        self.assertEqual(result["Rule pallet length (mm)"], 1500)

    def test_hard_limit_requires_manual_review(self):
        row = {
            "Item": "Tall", "Type": "Door", "Configuration": "Single",
            "Width (mm)": 1100, "Height (mm)": 2800, "Qty": 2,
            "Frame weight (kg)": 130, "Glass weight (kg)": 90,
            "Packing thickness (mm)": 75, "Handles installed": "Yes",
            "Glass mode": "Glazed",
        }
        result = calculate_rule_result(row)
        self.assertEqual(result["Rule manual review"], "Yes")
        self.assertIn("2700", result["Rule reason"])

    def test_system_names_are_not_training_features(self):
        data = generate_training_data(40)
        self.assertNotIn("System", data.columns)
        self.assertEqual(set(data["Type"].unique()), {"Window", "Door"})
        self.assertTrue((data["Data origin"] == "Synthetic").all())

    def test_old_template_is_normalized(self):
        source = pd.DataFrame([{
            "Item": "W-01", "Type": "Fixed Window", "Width (mm)": 1200,
            "Height (mm)": 1600, "Qty": 3, "Unit weight (kg)": 55,
            "Glass weight (kg)": 45, "Glass mode": "Glazed", "Rotated": "NO",
        }])
        normalized = normalize_input(source)
        self.assertEqual(normalized.iloc[0]["Type"], "Window")
        self.assertEqual(normalized.iloc[0]["Configuration"], "Fixed")
        self.assertEqual(normalized.iloc[0]["Packing thickness (mm)"], 75)
        self.assertEqual(validate_input(normalized), [])

    def test_models_return_comparison_and_pallets(self):
        bundle = train_models()
        source = normalize_input(pd.DataFrame([{
            "Item": "W-01", "Type": "Window", "Configuration": "Openable",
            "Width (mm)": 1400, "Height (mm)": 1800, "Qty": 12,
            "Frame weight (kg)": 70, "Glass weight (kg)": 55,
            "Packing thickness (mm)": 75, "Handles installed": "Yes",
            "Glass mode": "Glazed",
        }]))
        comparison, pallets = predict_batch(source, bundle)
        self.assertEqual(len(comparison), 1)
        self.assertEqual(comparison.iloc[0]["Rule pallets"], 2)
        self.assertEqual(len(pallets), 2)


if __name__ == "__main__":
    unittest.main()
