"""Pure packing calculation logic used by the Streamlit application.

Confirmed scope for this test version:
- only 1200 mm deep product pallets and glass boxes;
- Ireland pallet/box weight limit: 1000 kg;
- no uPVC rules;
- facades are limited by weight, not by item count;
- facade lengths over 6000 mm are divided into two equal parts;
- facade glass boxes are calculated at 3000 mm length.
"""

import math
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


PALLET_DEPTH_MM = 1200
TRAILER_INTERNAL_WIDTH_MM = 2400
MAX_PACKED_LENGTH_MM = 6800
MAX_PACKED_HEIGHT_MM = 2900
PACKING_ALLOWANCE_HEIGHT_MM = 200
MAX_VERTICAL_PRODUCT_HEIGHT_MM = MAX_PACKED_HEIGHT_MM - PACKING_ALLOWANCE_HEIGHT_MM
MAX_PALLET_WEIGHT_KG = 1000.0
MAX_GLASS_BOX_WEIGHT_KG = 1000.0
GLASS_BOX_PRICE_EUR = 180.0
MAX_FACADE_LENGTH_MM = 6000.0
FACADE_GLASS_BOX_LENGTH_MM = 3000.0

# Conservative capacities for a 1200 mm pallet when the detailed pictured
# configuration is not available in the input data.
CAPACITY_1200 = {
    "door": 6,
    "window": 6,
    "fixed window": 6,
    "door + sidelight": 6,
    "window + sidelight": 6,
    "sliding door": 2,
    "double sliding door": 2,
    "triple sliding door": 2,
    "quad sliding door": 2,
    "folding door": 6,
    "double folding door": 6,
    "triple folding door": 6,
    "quad folding door": 6,
    "5-leaf folding door": 6,
}

SLIDING_PARTS = {
    "sliding door": 1,
    "double sliding door": 2,
    "triple sliding door": 3,
    "quad sliding door": 4,
    "double folding door": 2,
    "triple folding door": 3,
    "quad folding door": 4,
    "5-leaf folding door": 5,
}

HEAVY_GLAZING_TYPES = set(SLIDING_PARTS)
FACADE_TYPES = {"facade"}
MAX_GLAZED_WIDTH_HEAVY_MM = 5000
MAX_ASSEMBLED_SLIDING_WIDTH_MM = 5960


@dataclass
class Construction:
    item_name: str
    item_type: str
    width_mm: float
    height_mm: float
    qty: int
    weight_kg: float
    glass_mode: str = "Glazed"
    glass_weight_kg: float = 0.0
    rotated: bool = False


def round_up_pallet_width(size_mm: float) -> int:
    """Return the commercial pallet-length price band."""
    for limit in (1000, 1500, 2500, 3500, 5000, 6000, 6600):
        if size_mm <= limit:
            return limit
    return int(math.ceil(size_mm / 1000.0) * 1000)


def pallet_price_eur(price_band_mm: float) -> float:
    if price_band_mm <= 1000:
        return 36.0
    if price_band_mm <= 1500:
        return 52.0
    if price_band_mm <= 2500:
        return 80.0
    if price_band_mm <= 3500:
        return 95.0
    if price_band_mm <= 5000:
        return 111.0
    return 145.0


def ldm_from_length(length_mm: float, count: int = 1) -> float:
    """LDM for a 1200 mm deep pallet in a 2400 mm wide trailer."""
    return (
        float(length_mm)
        * PALLET_DEPTH_MM
        * float(count)
        / TRAILER_INTERNAL_WIDTH_MM
        / 1000.0
    )


def real_pallet_width(width_mm: float, height_mm: float) -> float:
    """Compatibility alias: the old UI calls pallet length 'width'."""
    sideways = height_mm > MAX_VERTICAL_PRODUCT_HEIGHT_MM
    packed_length = height_mm if sideways else width_mm
    allowance = 200 if sideways or packed_length >= 3000 else 100
    return packed_length + allowance


def _not_possible(c, reason: str) -> Dict[str, object]:
    return {
        "Item": c.item_name,
        "Type": c.item_type,
        "Width (mm)": float(c.width_mm),
        "Height (mm)": float(c.height_mm),
        "Qty": int(c.qty),
        "Unit weight (kg)": float(c.weight_kg),
        "Glass weight (kg)": float(c.glass_weight_kg),
        "Glass parts": 1,
        "Glass pallet width (mm)": 0.0,
        "Glass mode": c.glass_mode,
        "Rotated": "YES" if c.rotated else "NO",
        "Packed as": "NOT POSSIBLE",
        "Glass separate": "N/A",
        "Packed sideways": "N/A",
        "Max per pallet": 0,
        "Pallet width (mm)": "N/A",
        "Notes": reason,
    }


def calculate_construction(c: Construction) -> Dict[str, object]:
    item_type = str(c.item_type).strip().lower()
    width = float(c.width_mm)
    height = float(c.height_mm)
    frame_weight = float(c.weight_kg)
    glass_weight = 0.0 if c.glass_mode == "Without glass" else float(c.glass_weight_kg)

    is_facade = item_type in FACADE_TYPES

    if frame_weight > MAX_PALLET_WEIGHT_KG and not is_facade:
        return _not_possible(c, "Frame weight exceeds the 1000 kg pallet limit")

    if c.rotated:
        packed_length_base = height
        vertical_product_height = width
        # Rotation is explicitly selected by the user. After the dimensions are
        # swapped, the original width is the vertical packed height.
        packed_sideways = False
    elif height > MAX_VERTICAL_PRODUCT_HEIGHT_MM:
        # Automatic sideways packing: the original height becomes pallet length.
        packed_length_base = height
        vertical_product_height = width
        packed_sideways = True
    else:
        packed_length_base = width
        vertical_product_height = height
        packed_sideways = False

    if vertical_product_height > MAX_VERTICAL_PRODUCT_HEIGHT_MM:
        return _not_possible(
            c,
            "Construction cannot fit vertically or sideways within the 2900 mm packed-height limit",
        )

    facade_length_parts = 1
    if is_facade and packed_length_base > MAX_FACADE_LENGTH_MM:
        facade_length_parts = 2
        packed_length_base /= 2.0
        if packed_length_base > MAX_FACADE_LENGTH_MM:
            return _not_possible(
                c,
                "Facade remains longer than 6000 mm after division into two parts; manual review required",
            )

    allowance = 200 if packed_sideways or packed_length_base >= 3000 else 100
    pallet_length = packed_length_base + allowance
    if pallet_length > MAX_PACKED_LENGTH_MM:
        return _not_possible(
            c,
            f"Required pallet length {pallet_length:.0f} mm exceeds the 6800 mm limit",
        )

    parts = SLIDING_PARTS.get(item_type, 1)
    packed_as = "UNGLAZED"
    glass_separate = "NO"
    notes: List[str] = []

    if c.glass_mode == "Without glass":
        notes.append("Frame only")
    elif c.glass_mode == "Unglazed" or is_facade:
        glass_separate = "YES" if glass_weight > 0 else "NO"
        if is_facade:
            notes.append("Facade: glass packed separately")
        else:
            notes.append("Glass packed separately")
    else:
        must_separate = False
        if packed_sideways:
            must_separate = True
            notes.append("Sideways packing: glass requires separate review/packing")
        if item_type in HEAVY_GLAZING_TYPES and packed_length_base > MAX_GLAZED_WIDTH_HEAVY_MM:
            must_separate = True
            notes.append("Glazed heavy construction exceeds 5000 mm")
        if frame_weight + glass_weight > MAX_PALLET_WEIGHT_KG:
            must_separate = True
            notes.append("Glass separated to keep product pallet within 1000 kg")
        if must_separate:
            packed_as = "UNGLAZED"
            glass_separate = "YES" if glass_weight > 0 else "NO"
        else:
            packed_as = "GLAZED"
            notes.append("Can be packed with glass")

    if (
        item_type in SLIDING_PARTS
        and packed_length_base > MAX_ASSEMBLED_SLIDING_WIDTH_MM
    ):
        packed_as = "SPLIT"
        glass_separate = "YES" if glass_weight > 0 else "NO"
        notes.append(f"Partially assembled in {parts} part(s)")

    if is_facade:
        max_per_pallet = 999999  # weight-only packing rule
        facade_weight_parts = max(1, int(math.ceil(frame_weight / MAX_PALLET_WEIGHT_KG)))
        facade_pallet_parts = max(facade_length_parts, facade_weight_parts)
        if facade_length_parts == 2:
            notes.append(
                f"Facade length divided into 2 equal parts of {width / 2:.0f} mm"
            )
        if facade_weight_parts > 1:
            notes.append(
                f"Facade frame divided across {facade_weight_parts} pallets to keep each at or below 1000 kg"
            )
        notes.append(
            f"Facade requires at least {facade_pallet_parts} product pallet(s); no unit-count limit"
        )
    else:
        max_per_pallet = CAPACITY_1200.get(item_type, 6)
        if item_type in {"window", "fixed window"}:
            notes.append("Conservative 6-unit limit; pictured configuration may allow more")

    # The entered glass weight is the total per construction. Parts describe
    # geometry only and must not multiply that weight.
    if is_facade:
        glass_pallet_length = FACADE_GLASS_BOX_LENGTH_MM
    else:
        glass_part_length = packed_length_base / parts if parts > 1 else packed_length_base
        glass_pallet_length = glass_part_length + (200 if glass_part_length >= 3000 else 100)

    if c.rotated:
        notes.append("Packed with width/height orientation swapped")

    return {
        "Item": c.item_name,
        "Type": c.item_type,
        "Width (mm)": width,
        "Height (mm)": height,
        "Qty": int(c.qty),
        "Unit weight (kg)": frame_weight,
        "Glass weight (kg)": glass_weight,
        "Glass parts": int(parts),
        "Glass pallet width (mm)": float(round(glass_pallet_length)),
        "Glass mode": c.glass_mode,
        "Rotated": "YES" if c.rotated else "NO",
        "Packed as": packed_as,
        "Glass separate": glass_separate,
        "Packed sideways": "YES" if packed_sideways else "NO",
        "Max per pallet": int(max_per_pallet),
        "Pallet width (mm)": int(round(pallet_length)),
        "Notes": "; ".join(notes),
    }


def expand_by_qty(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for unit_idx in range(1, int(row["Qty"]) + 1):
            item_type = str(row.get("Type", "")).strip().lower()
            if item_type in FACADE_TYPES:
                width = float(row.get("Width (mm)", 0) or 0)
                weight = float(row.get("Unit weight (kg)", 0) or 0)
                length_parts = 2 if width > MAX_FACADE_LENGTH_MM else 1
                weight_parts = max(1, int(math.ceil(weight / MAX_PALLET_WEIGHT_KG)))
                split_count = max(length_parts, weight_parts)
            else:
                split_count = 1

            for split_idx in range(1, split_count + 1):
                unit = row.copy()
                unit["Qty"] = 1
                unit["Unit idx"] = (
                    f"{unit_idx}.{split_idx}" if split_count > 1 else unit_idx
                )
                if split_count > 1:
                    unit["Unit weight (kg)"] = float(row["Unit weight (kg)"]) / split_count
                    unit["Max per pallet"] = 1
                rows.append(unit)
    return pd.DataFrame(rows)


def _unit_pallet_weight(row) -> float:
    frame = float(row.get("Unit weight (kg)", 0) or 0)
    glass = float(row.get("Glass weight (kg)", 0) or 0)
    if row.get("Glass separate") == "NO" and row.get("Glass mode") == "Glazed":
        return frame + glass
    return frame


def pack_mixed(units: pd.DataFrame) -> List[Dict[str, object]]:
    if units.empty:
        return []

    prepared = units.copy()
    prepared["_total_weight"] = prepared.apply(_unit_pallet_weight, axis=1)
    prepared = prepared.sort_values("_total_weight", ascending=False).reset_index(drop=True)
    pallets: List[Dict[str, object]] = []

    for _, item in prepared.iterrows():
        weight = float(item["_total_weight"])
        if weight > MAX_PALLET_WEIGHT_KG:
            raise ValueError(f"Single unit exceeds 1000 kg: {item.get('Item', '')}")
        item_max = int(item.get("Max per pallet", 6) or 6)
        placed = False
        for pallet in pallets:
            pallet_max = min(int(pallet["max_per_pallet"]), item_max)
            if (
                float(pallet["weight_kg"]) + weight <= MAX_PALLET_WEIGHT_KG
                and int(pallet["items_count"]) + 1 <= pallet_max
            ):
                pallet["weight_kg"] += weight
                pallet["items_count"] += 1
                pallet["max_per_pallet"] = pallet_max
                pallet["items"].append(item)
                placed = True
                break
        if not placed:
            pallets.append({
                "weight_kg": weight,
                "items_count": 1,
                "max_per_pallet": item_max,
                "items": [item],
            })
    return pallets


def _get_pallet_length(item) -> float:
    return float(item["Pallet width (mm)"])


def build_pallet_outputs(results_df: pd.DataFrame):
    valid = results_df[results_df["Packed as"] != "NOT POSSIBLE"].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0

    pallets = pack_mixed(expand_by_qty(valid))
    summaries = []
    plans = []
    total_cost = 0.0
    total_ldm = 0.0

    for number, pallet in enumerate(pallets, start=1):
        items = pd.DataFrame(pallet["items"])
        actual_length = float(items.apply(_get_pallet_length, axis=1).max())
        price_band = round_up_pallet_width(actual_length)
        price = pallet_price_eur(price_band)
        ldm = ldm_from_length(actual_length)
        total_cost += price
        total_ldm += ldm
        summaries.append({
            "Pallet no": number,
            "Pallet weight (kg)": round(float(pallet["weight_kg"]), 2),
            "Constructions count": int(items["Item"].nunique()),
            "Units count": int(len(items)),
            "Pallet width (mm)": round(actual_length, 1),
            "Price band width (mm)": int(price_band),
            "Pallet price (EUR)": price,
            "Pallet LDM": round(ldm, 3),
        })
        for _, item in items.iterrows():
            plans.append({
                "Pallet no": number,
                "Item": item["Item"],
                "Type": item["Type"],
                "Width (mm)": item["Width (mm)"],
                "Height (mm)": item["Height (mm)"],
                "Unit weight (kg)": item["Unit weight (kg)"],
                "Packed as": item["Packed as"],
                "Glass separate": item["Glass separate"],
                "Packed sideways": item["Packed sideways"],
                "Pallet width (mm)": item["Pallet width (mm)"],
                "Unit idx": item["Unit idx"],
            })
    return pd.DataFrame(summaries), pd.DataFrame(plans), total_cost, total_ldm


def calculate_glass_boxes(results_df: pd.DataFrame):
    if results_df.empty:
        return 0, 0.0, 0.0, 0.0, float(PALLET_DEPTH_MM)
    separate = results_df[results_df["Glass separate"] == "YES"].copy()
    if separate.empty:
        return 0, 0.0, 0.0, 0.0, float(PALLET_DEPTH_MM)

    weights = pd.to_numeric(separate["Glass weight (kg)"], errors="coerce").fillna(0.0)
    quantities = pd.to_numeric(separate["Qty"], errors="coerce").fillna(0.0)
    total_weight = float((weights * quantities).sum())
    if total_weight <= 0:
        return 0, 0.0, 0.0, 0.0, float(PALLET_DEPTH_MM)

    boxes = int(math.ceil(total_weight / MAX_GLASS_BOX_WEIGHT_KG))
    max_length = float(
        pd.to_numeric(separate["Glass pallet width (mm)"], errors="coerce")
        .fillna(PALLET_DEPTH_MM)
        .max()
    )
    cost = boxes * GLASS_BOX_PRICE_EUR
    ldm = ldm_from_length(max_length, boxes)
    return boxes, total_weight, cost, ldm, max_length
