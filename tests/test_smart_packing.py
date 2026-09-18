import unittest

from smart_packing import (
    aggregate_assignments,
    demo_project,
    required_pallet_length,
    suggest_packing,
    summarize_allocation,
    validate_input,
)


class SmartPackingTests(unittest.TestCase):
    def test_project_one_matches_confirmed_example(self):
        allocation, summary = suggest_packing(demo_project())
        actual = aggregate_assignments(allocation)
        composition = {
            int(pallet): dict(zip(group["Item"], group["Qty on pallet"]))
            for pallet, group in actual.groupby("Pallet no")
        }
        self.assertEqual(
            composition,
            {
                1: {"D01": 3, "W02": 2},
                2: {"D02": 2, "D03": 3},
                3: {"D03": 2, "W03": 4},
                4: {"W01": 4},
            },
        )
        self.assertEqual(summary["Weight (kg)"].tolist(), [720.0, 920.0, 1000.0, 280.0])
        self.assertEqual(summary["Pallet length (mm)"].tolist(), [1300, 2500, 1700, 700])
        self.assertEqual(summary["Status"].tolist(), ["Within limits"] * 4)

    def test_length_allowance_boundary(self):
        self.assertEqual(required_pallet_length(3000), 3100)
        self.assertEqual(required_pallet_length(3001), 3201)

    def test_total_units_and_weight_are_preserved(self):
        source = demo_project()
        allocation, summary = suggest_packing(source)
        self.assertEqual(len(allocation), 20)
        expected_weight = sum(
            (row["Frame weight (kg)"] + row["Glass weight (kg)"]) * row["Qty"]
            for _, row in source.iterrows()
        )
        self.assertEqual(allocation["Unit weight (kg)"].sum(), expected_weight)
        self.assertEqual(summary["Weight (kg)"].sum(), expected_weight)

    def test_manual_overload_is_visible(self):
        allocation, _ = suggest_packing(demo_project())
        allocation["Pallet no"] = 1
        summary = summarize_allocation(allocation)
        self.assertIn("20 units > 6", summary.iloc[0]["Status"])
        self.assertIn("2920 kg > 1000 kg", summary.iloc[0]["Status"])

    def test_unsupported_type_is_rejected(self):
        source = demo_project()
        source.loc[0, "Type"] = "Sliding Door"
        self.assertTrue(any("Door or Window" in error for error in validate_input(source)))


if __name__ == "__main__":
    unittest.main()
