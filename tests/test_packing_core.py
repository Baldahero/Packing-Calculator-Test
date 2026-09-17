import unittest

import pandas as pd

from packing_core import (
    Construction,
    build_pallet_outputs,
    calculate_construction,
    calculate_glass_boxes,
)


class PackingCoreTests(unittest.TestCase):
    def test_tall_construction_uses_height_for_sideways_pallet_length(self):
        result = calculate_construction(Construction(
            "Tall door", "Door", 1600, 2800, 1, 150, "Unglazed", 50
        ))
        self.assertEqual(result["Packed sideways"], "YES")
        self.assertEqual(result["Pallet width (mm)"], 3000)
        summary, _, _, ldm = build_pallet_outputs(pd.DataFrame([result]))
        self.assertEqual(summary.iloc[0]["Price band width (mm)"], 3500)
        self.assertEqual(ldm, 1.5)

    def test_glass_weight_is_not_multiplied_by_number_of_parts(self):
        result = calculate_construction(Construction(
            "Triple slider", "Triple Sliding Door", 4500, 2400, 1, 300,
            "Unglazed", 600,
        ))
        boxes, weight, _, _, _ = calculate_glass_boxes(pd.DataFrame([result]))
        self.assertEqual(result["Glass parts"], 3)
        self.assertEqual(weight, 600)
        self.assertEqual(boxes, 1)

    def test_glass_is_separated_when_combined_weight_exceeds_limit(self):
        result = calculate_construction(Construction(
            "Heavy glazed door", "Door", 2000, 2400, 1, 800, "Glazed", 300
        ))
        self.assertEqual(result["Packed as"], "UNGLAZED")
        self.assertEqual(result["Glass separate"], "YES")
        summary, _, _, _ = build_pallet_outputs(pd.DataFrame([result]))
        self.assertEqual(summary.iloc[0]["Pallet weight (kg)"], 800)

    def test_facades_have_weight_limit_but_no_six_item_limit(self):
        result = calculate_construction(Construction(
            "Facade bar", "Facade", 3000, 1, 10, 80, "Without glass", 0
        ))
        summary, _, _, _ = build_pallet_outputs(pd.DataFrame([result]))
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["Units count"], 10)
        self.assertEqual(summary.iloc[0]["Pallet weight (kg)"], 800)

    def test_facades_split_only_when_weight_exceeds_1000_kg(self):
        result = calculate_construction(Construction(
            "Facade bar", "Facade", 3000, 1, 13, 80, "Without glass", 0
        ))
        summary, _, _, _ = build_pallet_outputs(pd.DataFrame([result]))
        self.assertEqual(len(summary), 2)
        self.assertEqual(sorted(summary["Pallet weight (kg)"].tolist()), [80, 960])

    def test_required_pallet_length_over_6800_is_rejected(self):
        result = calculate_construction(Construction(
            "Too long", "Door", 6700, 2000, 1, 100, "Without glass", 0
        ))
        self.assertEqual(result["Packed as"], "NOT POSSIBLE")
        self.assertIn("6800", result["Notes"])

    def test_frame_over_1000_kg_is_rejected(self):
        result = calculate_construction(Construction(
            "Too heavy", "Door", 2000, 2000, 1, 1001, "Without glass", 0
        ))
        self.assertEqual(result["Packed as"], "NOT POSSIBLE")

if __name__ == "__main__":
    unittest.main()
