"""Smart mixed-pallet demonstration for simple windows and doors.

The prototype uses deterministic constrained optimisation.  Hard limits stay
in code; future production examples can be used to learn the compatibility
score that decides which different constructions should share a pallet.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from ml_scoring import pairing_cost


PALLET_DEPTH_MM = 1200
MAX_ITEMS_PER_PALLET = 6
MAX_PALLET_WEIGHT_KG = 1000.0
MAX_PALLET_LENGTH_MM = 6800
MAX_DEMO_HEIGHT_MM = 2700

INPUT_COLUMNS = [
    "Project",
    "Item",
    "Type",
    "Width (mm)",
    "Height (mm)",
    "Qty",
    "Frame weight (kg)",
    "Glass weight (kg)",
    "Glass mode",
]


def _number(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace("\xa0", "").replace(" ", "").replace(",", "."))
    except (TypeError, ValueError):
        return default


def _normalise_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if "window" in text or "lang" in text:
        return "Window"
    if "door" in text or "dur" in text:
        return "Door"
    return str(value or "").strip()


def _normalise_glass_mode(value: object) -> str:
    text = str(value or "Glazed").strip().lower()
    if text in {"without glass", "no glass", "be stiklo"}:
        return "Without glass"
    if text in {"unglazed", "separate", "separate glass", "neįstiklinta", "neistiklinta"}:
        return "Unglazed"
    return "Glazed"


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the demo template or the existing calculator import to one schema."""
    aliases = {
        "project": "Project",
        "project id": "Project",
        "item": "Item",
        "item name": "Item",
        "construction": "Item",
        "type": "Type",
        "width (mm)": "Width (mm)",
        "width mm": "Width (mm)",
        "height (mm)": "Height (mm)",
        "height mm": "Height (mm)",
        "qty": "Qty",
        "quantity": "Qty",
        "frame weight (kg)": "Frame weight (kg)",
        "frame kg": "Frame weight (kg)",
        "unit weight (kg)": "Frame weight (kg)",
        "glass weight (kg)": "Glass weight (kg)",
        "glass kg": "Glass weight (kg)",
        "glass mode": "Glass mode",
    }
    source = df.copy()
    source = source.rename(
        columns={column: aliases.get(str(column).strip().lower(), str(column).strip()) for column in source.columns}
    )

    for column in INPUT_COLUMNS:
        if column not in source.columns:
            source[column] = None

    rows: List[Dict[str, object]] = []
    for index, row in source.iterrows():
        if all(pd.isna(row.get(column)) for column in source.columns):
            continue
        item = str(row.get("Item") or "").strip()
        if not item or item.lower() == "nan":
            item = f"Item {index + 1}"
        project = str(row.get("Project") or "Project 1").strip()
        if not project or project.lower() == "nan":
            project = "Project 1"
        rows.append({
            "Project": project,
            "Item": item,
            "Type": _normalise_type(row.get("Type")),
            "Width (mm)": _number(row.get("Width (mm)")),
            "Height (mm)": _number(row.get("Height (mm)")),
            "Qty": int(max(0, round(_number(row.get("Qty"))))),
            "Frame weight (kg)": _number(row.get("Frame weight (kg)")),
            "Glass weight (kg)": _number(row.get("Glass weight (kg)")),
            "Glass mode": _normalise_glass_mode(row.get("Glass mode")),
        })
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def demo_project() -> pd.DataFrame:
    return normalize_input(pd.DataFrame([
        {"Project": "Project 1", "Item": "D01", "Type": "Door", "Width (mm)": 1200, "Height (mm)": 2400, "Qty": 3, "Frame weight (kg)": 60, "Glass weight (kg)": 80, "Glass mode": "Glazed"},
        {"Project": "Project 1", "Item": "D02", "Type": "Door", "Width (mm)": 2400, "Height (mm)": 2600, "Qty": 2, "Frame weight (kg)": 100, "Glass weight (kg)": 120, "Glass mode": "Glazed"},
        {"Project": "Project 1", "Item": "D03", "Type": "Door", "Width (mm)": 1500, "Height (mm)": 2500, "Qty": 5, "Frame weight (kg)": 70, "Glass weight (kg)": 90, "Glass mode": "Glazed"},
        {"Project": "Project 1", "Item": "W01", "Type": "Window", "Width (mm)": 600, "Height (mm)": 1200, "Qty": 4, "Frame weight (kg)": 30, "Glass weight (kg)": 40, "Glass mode": "Glazed"},
        {"Project": "Project 1", "Item": "W02", "Type": "Window", "Width (mm)": 1100, "Height (mm)": 1400, "Qty": 2, "Frame weight (kg)": 70, "Glass weight (kg)": 80, "Glass mode": "Glazed"},
        {"Project": "Project 1", "Item": "W03", "Type": "Window", "Width (mm)": 1600, "Height (mm)": 2000, "Qty": 4, "Frame weight (kg)": 80, "Glass weight (kg)": 90, "Glass mode": "Glazed"},
    ]))


def empty_project() -> pd.DataFrame:
    return normalize_input(pd.DataFrame([{
        "Project": "Project 1",
        "Item": "W01",
        "Type": "Window",
        "Width (mm)": 1000,
        "Height (mm)": 1200,
        "Qty": 1,
        "Frame weight (kg)": 40,
        "Glass weight (kg)": 30,
        "Glass mode": "Glazed",
    }]))


def unit_pallet_weight(row: pd.Series | Dict[str, object]) -> float:
    frame = _number(row.get("Frame weight (kg)"))
    glass = _number(row.get("Glass weight (kg)"))
    return frame + glass if row.get("Glass mode") == "Glazed" else frame


def required_pallet_length(width_mm: float) -> float:
    """Up to and including 3000 mm add 100 mm; above 3000 mm add 200 mm."""
    width = float(width_mm)
    return width + (100 if width <= 3000 else 200)


def validate_input(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if df.empty:
        return ["Add at least one construction."]
    projects = [value for value in df["Project"].dropna().unique() if str(value).strip()]
    if len(projects) > 1:
        errors.append("The demonstration calculates one project at a time.")
    duplicate_items = df[df.duplicated(["Project", "Item"], keep=False)]["Item"].unique()
    if len(duplicate_items):
        errors.append(f"Item identifiers must be unique: {', '.join(map(str, duplicate_items))}.")

    for index, row in df.iterrows():
        label = str(row.get("Item") or f"Row {index + 1}")
        if row.get("Type") not in {"Door", "Window"}:
            errors.append(f"{label}: the first demonstration supports only Door or Window.")
        if _number(row.get("Width (mm)")) <= 0:
            errors.append(f"{label}: width must be greater than 0 mm.")
        if _number(row.get("Height (mm)")) <= 0:
            errors.append(f"{label}: height must be greater than 0 mm.")
        if _number(row.get("Height (mm)")) > MAX_DEMO_HEIGHT_MM:
            errors.append(f"{label}: height over 2700 mm requires the later sideways-packing stage.")
        if int(_number(row.get("Qty"))) <= 0:
            errors.append(f"{label}: quantity must be at least 1.")
        if _number(row.get("Frame weight (kg)")) <= 0:
            errors.append(f"{label}: frame weight must be greater than 0 kg.")
        if _number(row.get("Glass weight (kg)")) < 0:
            errors.append(f"{label}: glass weight cannot be negative.")
        weight = unit_pallet_weight(row)
        if weight > MAX_PALLET_WEIGHT_KG:
            errors.append(f"{label}: one construction weighs {weight:.0f} kg and exceeds the 1000 kg limit.")
        length = required_pallet_length(_number(row.get("Width (mm)")))
        if length > MAX_PALLET_LENGTH_MM:
            errors.append(f"{label}: required pallet length {length:.0f} mm exceeds 6800 mm.")
    return errors


def read_uploaded_workbook(source) -> Tuple[pd.DataFrame, str]:
    workbook = pd.ExcelFile(source)
    sheet = "Constructions" if "Constructions" in workbook.sheet_names else workbook.sheet_names[0]
    raw = workbook.parse(sheet_name=sheet)
    return normalize_input(raw), sheet


def _build_problem(rows: pd.DataFrame, bin_count: int):
    """Build a compact position-level MILP for one fixed number of pallets."""
    positions = rows.reset_index(drop=True)
    position_count = len(positions)
    pairs = [(left, right) for left in range(position_count) for right in range(left + 1, position_count)]

    x_start = 0
    u_start = position_count * bin_count
    length_start = u_start + position_count * bin_count
    pair_start = length_start + bin_count
    variable_count = pair_start + len(pairs) * bin_count

    objective = np.zeros(variable_count)
    objective[length_start:pair_start] = 1000.0
    objective[u_start:length_start] = 0.05  # discourage unnecessary position splits
    for pair_index, (left, right) in enumerate(pairs):
        compatibility_cost = pairing_cost(
            positions.loc[left],
            positions.loc[right],
        )
        for pallet in range(bin_count):
            objective[pair_start + pair_index * bin_count + pallet] = compatibility_cost

    integrality = np.zeros(variable_count)
    integrality[x_start:u_start] = 1
    integrality[u_start:length_start] = 1
    integrality[pair_start:] = 1

    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    for position, row in positions.iterrows():
        upper_bounds[position * bin_count:(position + 1) * bin_count] = int(row["Qty"])
    upper_bounds[u_start:length_start] = 1
    upper_bounds[length_start:pair_start] = MAX_PALLET_LENGTH_MM
    upper_bounds[pair_start:] = 1

    matrix_rows: List[Dict[int, float]] = []
    constraint_lower: List[float] = []
    constraint_upper: List[float] = []

    def add_constraint(coefficients: Dict[int, float], lower: float, upper: float) -> None:
        matrix_rows.append(coefficients)
        constraint_lower.append(lower)
        constraint_upper.append(upper)

    for position, row in positions.iterrows():
        add_constraint(
            {position * bin_count + pallet: 1.0 for pallet in range(bin_count)},
            int(row["Qty"]),
            int(row["Qty"]),
        )

    for position, row in positions.iterrows():
        quantity = int(row["Qty"])
        for pallet in range(bin_count):
            x = position * bin_count + pallet
            used = u_start + position * bin_count + pallet
            add_constraint({x: 1.0, used: -float(quantity)}, -np.inf, 0.0)
            add_constraint({used: 1.0, x: -1.0}, -np.inf, 0.0)

    for pallet in range(bin_count):
        add_constraint(
            {position * bin_count + pallet: 1.0 for position in range(position_count)},
            1.0,
            float(MAX_ITEMS_PER_PALLET),
        )
        add_constraint(
            {
                position * bin_count + pallet: unit_pallet_weight(positions.loc[position])
                for position in range(position_count)
            },
            -np.inf,
            MAX_PALLET_WEIGHT_KG,
        )
        for position, row in positions.iterrows():
            used = u_start + position * bin_count + pallet
            length = length_start + pallet
            add_constraint(
                {used: required_pallet_length(row["Width (mm)"]), length: -1.0},
                -np.inf,
                0.0,
            )

    # Reduce equivalent pallet permutations by keeping optimisation lengths descending.
    for pallet in range(bin_count - 1):
        add_constraint({length_start + pallet: 1.0, length_start + pallet + 1: -1.0}, 0.0, np.inf)

    for pair_index, (left, right) in enumerate(pairs):
        for pallet in range(bin_count):
            paired = pair_start + pair_index * bin_count + pallet
            left_used = u_start + left * bin_count + pallet
            right_used = u_start + right * bin_count + pallet
            add_constraint({paired: 1.0, left_used: -1.0}, -np.inf, 0.0)
            add_constraint({paired: 1.0, right_used: -1.0}, -np.inf, 0.0)
            add_constraint({paired: -1.0, left_used: 1.0, right_used: 1.0}, -np.inf, 1.0)

    row_indices: List[int] = []
    column_indices: List[int] = []
    values: List[float] = []
    for matrix_row, coefficients in enumerate(matrix_rows):
        for column, value in coefficients.items():
            row_indices.append(matrix_row)
            column_indices.append(column)
            values.append(value)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(matrix_rows), variable_count),
    ).tocsr()

    return (
        positions,
        objective,
        integrality,
        Bounds(lower_bounds, upper_bounds),
        LinearConstraint(matrix, np.array(constraint_lower), np.array(constraint_upper)),
        x_start,
        u_start,
    )


def suggest_packing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return unit-level pallet assignments and a calculated pallet summary."""
    rows = normalize_input(df)
    errors = validate_input(rows)
    if errors:
        raise ValueError("\n".join(errors))

    total_units = int(rows["Qty"].sum())
    total_weight = float(sum(unit_pallet_weight(row) * int(row["Qty"]) for _, row in rows.iterrows()))
    lower_bound = max(
        1,
        math.ceil(total_units / MAX_ITEMS_PER_PALLET),
        math.ceil(total_weight / MAX_PALLET_WEIGHT_KG),
    )

    solved = None
    for pallet_count in range(lower_bound, total_units + 1):
        problem = _build_problem(rows, pallet_count)
        positions, objective, integrality, bounds, constraints, x_start, u_start = problem
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": 8.0, "mip_rel_gap": 0.0},
        )
        if result.success and result.x is not None:
            solved = (pallet_count, positions, result, x_start, u_start)
            break
        if pallet_count >= lower_bound + 12:
            break
    if solved is None:
        raise RuntimeError("No valid packing plan was found within the demonstration limits.")

    pallet_count, positions, result, x_start, u_start = solved
    quantities = np.rint(result.x[x_start:u_start]).astype(int).reshape(len(positions), pallet_count)

    populated_bins = []
    for pallet in range(pallet_count):
        position_indexes = [position for position in range(len(positions)) if quantities[position, pallet] > 0]
        populated_bins.append((pallet, min(position_indexes)))
    sorted_bins = [pallet for pallet, _ in sorted(populated_bins, key=lambda value: value[1])]
    pallet_number = {old: new for new, old in enumerate(sorted_bins, start=1)}

    allocation_rows: List[Dict[str, object]] = []
    next_unit_number = {position: 1 for position in range(len(positions))}
    for old_pallet in sorted_bins:
        for position, row in positions.iterrows():
            count = int(quantities[position, old_pallet])
            for _ in range(count):
                number = next_unit_number[position]
                next_unit_number[position] += 1
                allocation_rows.append({
                    "Project": row["Project"],
                    "Unit ID": f"{row['Item']}-{number}",
                    "Item": row["Item"],
                    "Type": row["Type"],
                    "Width (mm)": float(row["Width (mm)"]),
                    "Height (mm)": float(row["Height (mm)"]),
                    "Unit weight (kg)": unit_pallet_weight(row),
                    "Pallet no": pallet_number[old_pallet],
                })
    allocation = pd.DataFrame(allocation_rows)
    return allocation, summarize_allocation(allocation)


def validate_allocation(allocation: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if allocation.empty:
        return ["The packing plan is empty."]
    if allocation["Unit ID"].duplicated().any():
        errors.append("Every construction unit must appear only once.")
    pallet_numbers = pd.to_numeric(allocation["Pallet no"], errors="coerce")
    if pallet_numbers.isna().any() or (pallet_numbers < 1).any() or ((pallet_numbers % 1) != 0).any():
        errors.append("Pallet numbers must be positive whole numbers.")
    return errors


def summarize_allocation(allocation: pd.DataFrame) -> pd.DataFrame:
    errors = validate_allocation(allocation)
    if errors:
        raise ValueError("\n".join(errors))
    working = allocation.copy()
    working["Pallet no"] = pd.to_numeric(working["Pallet no"]).astype(int)
    summaries: List[Dict[str, object]] = []
    for pallet, group in working.groupby("Pallet no", sort=True):
        group = group.sort_values(["Width (mm)", "Item"], ascending=[False, True], kind="stable")
        count = int(len(group))
        weight = float(group["Unit weight (kg)"].sum())
        base_length = float(group["Width (mm)"].max())
        length = required_pallet_length(base_length)
        composition_parts = []
        for item in group["Item"].drop_duplicates():
            composition_parts.append(f"{item} × {int((group['Item'] == item).sum())}")
        problems = []
        if count > MAX_ITEMS_PER_PALLET:
            problems.append(f"{count} units > 6")
        if weight > MAX_PALLET_WEIGHT_KG + 1e-6:
            problems.append(f"{weight:.0f} kg > 1000 kg")
        if length > MAX_PALLET_LENGTH_MM:
            problems.append(f"{length:.0f} mm > 6800 mm")
        summaries.append({
            "Pallet no": int(pallet),
            "Composition": ", ".join(composition_parts),
            "Units": count,
            "Weight (kg)": round(weight, 1),
            "Longest construction (mm)": round(base_length),
            "Pallet length (mm)": round(length),
            "Pallet depth (mm)": PALLET_DEPTH_MM,
            "LDM": round(length / 2000.0, 3),
            "Status": "Check: " + "; ".join(problems) if problems else "Within limits",
        })
    return pd.DataFrame(summaries)


def aggregate_assignments(allocation: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        allocation.groupby(["Project", "Pallet no", "Item"], sort=False)
        .agg(**{"Qty on pallet": ("Unit ID", "size"), "_width": ("Width (mm)", "max")})
        .reset_index()
        .sort_values(["Pallet no", "_width", "Item"], ascending=[True, False, True], kind="stable")
        .drop(columns="_width")
        .reset_index(drop=True)
    )
    return grouped


def make_training_workbook(
    input_df: pd.DataFrame,
    allocation: pd.DataFrame,
    summary: pd.DataFrame,
) -> bytes:
    """Export the current confirmed plan in the format needed for future learning."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        normalize_input(input_df).to_excel(writer, sheet_name="Constructions", index=False)
        aggregate_assignments(allocation).to_excel(writer, sheet_name="Actual Packing", index=False)
        summary.to_excel(writer, sheet_name="Pallet Summary", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.sheet_view.showGridLines = False
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")
            for column in worksheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column)
                worksheet.column_dimensions[column[0].column_letter].width = min(max(max_length + 2, 10), 34)
    return output.getvalue()
