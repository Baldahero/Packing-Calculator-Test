"""Streamlit page for the experimental window/door ML demonstration."""

from __future__ import annotations

import html
from textwrap import dedent

import pandas as pd
import streamlit as st

from ml_packing import (
    INPUT_COLUMNS,
    make_ml_import_template,
    make_result_workbook,
    normalize_input,
    predict_batch,
    read_uploaded_workbook,
    train_models,
    validate_input,
)


if not st.session_state.get("_embedded_ml"):
    st.set_page_config(page_title="Packing ML Demo", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Training experimental model...")
def get_models():
    return train_models()


def demo_rows() -> pd.DataFrame:
    return normalize_input(pd.DataFrame([
        {"Item": "W-01", "Type": "Window", "Configuration": "Fixed", "Width (mm)": 1200, "Height (mm)": 1600, "Qty": 10, "Frame weight (kg)": 55, "Glass weight (kg)": 45, "Packing thickness (mm)": 75, "Handles installed": "No", "Glass mode": "Glazed"},
        {"Item": "W-02", "Type": "Window", "Configuration": "Openable", "Width (mm)": 1400, "Height (mm)": 1800, "Qty": 12, "Frame weight (kg)": 70, "Glass weight (kg)": 55, "Packing thickness (mm)": 75, "Handles installed": "Yes", "Glass mode": "Glazed"},
        {"Item": "D-01", "Type": "Door", "Configuration": "Single", "Width (mm)": 1000, "Height (mm)": 2300, "Qty": 6, "Frame weight (kg)": 95, "Glass weight (kg)": 65, "Packing thickness (mm)": 75, "Handles installed": "Yes", "Glass mode": "Glazed"},
        {"Item": "D-02", "Type": "Door", "Configuration": "Double", "Width (mm)": 2000, "Height (mm)": 2500, "Qty": 9, "Frame weight (kg)": 150, "Glass weight (kg)": 170, "Packing thickness (mm)": 75, "Handles installed": "Yes", "Glass mode": "Glazed"},
    ]))


def empty_rows() -> pd.DataFrame:
    return normalize_input(pd.DataFrame([{
        "Item": "Item 1", "Type": "Window", "Configuration": "Openable",
        "Width (mm)": 1000, "Height (mm)": 1000, "Qty": 1,
        "Frame weight (kg)": 50, "Glass weight (kg)": 40,
        "Packing thickness (mm)": 75, "Handles installed": "Yes", "Glass mode": "Glazed",
    }]))


def pallet_html(row: pd.Series) -> str:
    item = html.escape(str(row["Item"]))
    units_a = int(row["Side A"])
    units_b = int(row["Side B"])
    blocks_a = "".join('<span class="unit unit-a"></span>' for _ in range(units_a)) or '<span class="empty">empty</span>'
    blocks_b = "".join('<span class="unit unit-b"></span>' for _ in range(units_b)) or '<span class="empty">empty</span>'
    review = " review" if row["Manual review"] == "Yes" else ""
    return dedent(f"""
        <div class="pallet-card{review}">
          <div class="pallet-title">{item} · Pallet {int(row['Pallet'])}</div>
          <div class="pallet-meta">{int(row['Units'])} units · {row['Pallet weight (kg)']:.0f} kg · {row['Pallet length (mm)']:.0f} × 1200 mm · {row['Pallet LDM']:.3f} LDM</div>
          <div class="pallet-body">
            <div class="side"><b>Side A</b><div class="units">{blocks_a}</div><small>{units_a} unit(s)</small></div>
            <div class="rack"><span>100 mm rack</span></div>
            <div class="side"><b>Side B</b><div class="units">{blocks_b}</div><small>{units_b} unit(s)</small></div>
          </div>
        </div>
    """).strip()


st.markdown("""
<style>
.pallet-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:14px; margin:8px 0 20px; }
.pallet-card { border:1px solid #b8c7d3; border-radius:12px; padding:12px; background:#f8fbfd; box-shadow:0 2px 8px rgba(24,50,74,.08); }
.pallet-card.review { border:2px solid #d97706; background:#fff8e8; }
.pallet-title { color:#18324a; font-weight:700; font-size:16px; }
.pallet-meta { color:#4b6476; font-size:12px; margin:3px 0 10px; }
.pallet-body { display:grid; grid-template-columns:1fr 34px 1fr; min-height:120px; border:3px solid #6b4f2b; background:#efe2c6; }
.side { padding:8px; text-align:center; }
.rack { display:flex; align-items:center; justify-content:center; background:#805c32; color:white; }
.rack span { writing-mode:vertical-rl; transform:rotate(180deg); font-size:10px; }
.units { display:flex; flex-wrap:wrap; justify-content:center; gap:5px; margin:8px 0; }
.unit { display:inline-block; width:22px; height:56px; border:2px solid #18324a; border-radius:3px; }
.unit-a { background:#75bde0; }.unit-b { background:#92d6a1; }.empty { color:#8a8a8a; font-style:italic; }
</style>
""", unsafe_allow_html=True)

st.title("Packing Calculator · Experimental ML Demo")
st.caption("Simple windows and doors only. No profile-system names are used by the model.")
st.warning("Experimental ML predictions are shown for comparison. The final result always uses the hard packing rules and safety limits.")

if "ml_input" not in st.session_state:
    st.session_state.ml_input = demo_rows()
if "ml_comparison" not in st.session_state:
    st.session_state.ml_comparison = pd.DataFrame()
if "ml_pallets" not in st.session_state:
    st.session_state.ml_pallets = pd.DataFrame()

input_tab, import_tab, model_tab = st.tabs(["Manual input", "Excel import", "Model information"])

with input_tab:
    action1, action2, action3 = st.columns([1, 1, 3])
    if action1.button("Load demo", use_container_width=True):
        st.session_state.ml_input = demo_rows()
        st.session_state.ml_comparison = pd.DataFrame()
        st.session_state.ml_pallets = pd.DataFrame()
        st.rerun()
    if action2.button("Clear", use_container_width=True):
        st.session_state.ml_input = empty_rows()
        st.session_state.ml_comparison = pd.DataFrame()
        st.session_state.ml_pallets = pd.DataFrame()
        st.rerun()
    action3.caption("Add, remove or edit rows directly in the table.")

    edited = st.data_editor(
        st.session_state.ml_input,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(options=["Window", "Door"], required=True),
            "Configuration": st.column_config.SelectboxColumn(options=["Fixed", "Openable", "Mixed", "Single", "Double", "Door + sidelight"], required=True),
            "Handles installed": st.column_config.SelectboxColumn(options=["Yes", "No"], required=True),
            "Glass mode": st.column_config.SelectboxColumn(options=["Glazed", "Unglazed", "Without glass"], required=True),
            "Width (mm)": st.column_config.NumberColumn(min_value=1, step=1),
            "Height (mm)": st.column_config.NumberColumn(min_value=1, step=1),
            "Qty": st.column_config.NumberColumn(min_value=1, step=1),
            "Frame weight (kg)": st.column_config.NumberColumn(min_value=0.01, step=0.1),
            "Glass weight (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "Packing thickness (mm)": st.column_config.NumberColumn(min_value=1, step=1),
        },
        key="ml_editor",
    )
    st.session_state.ml_input = normalize_input(edited)

with import_tab:
    st.download_button(
        "Download simple ML template",
        data=make_ml_import_template(),
        file_name="packing_ml_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption("Accepted formats: the new ML Input sheet, the existing Constructions template, or the experimental ML data file.")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"], key="ml_upload")
    if uploaded is not None:
        try:
            imported, detected, warnings = read_uploaded_workbook(uploaded)
            st.success(f"Detected: {detected}. Found {len(imported)} supported row(s).")
            for warning in warnings:
                st.warning(warning)
            st.dataframe(imported, use_container_width=True, hide_index=True)
            replace_col, add_col = st.columns(2)
            if replace_col.button("Import and replace", use_container_width=True):
                st.session_state.ml_input = imported
                st.session_state.ml_comparison = pd.DataFrame()
                st.session_state.ml_pallets = pd.DataFrame()
                st.success("Input replaced. Open Manual input to review and calculate.")
            if add_col.button("Import and add", use_container_width=True):
                st.session_state.ml_input = pd.concat([st.session_state.ml_input, imported], ignore_index=True)
                st.session_state.ml_comparison = pd.DataFrame()
                st.session_state.ml_pallets = pd.DataFrame()
                st.success("Rows added. Open Manual input to review and calculate.")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

with model_tab:
    bundle = get_models()
    st.write(f"Training examples: **{bundle.training_rows}** · Independent test examples: **{bundle.test_rows}**")
    metric_cols = st.columns(len(bundle.metrics))
    for col, (name, value) in zip(metric_cols, bundle.metrics.items()):
        display = f"{value:.1%}" if "accuracy" in name.lower() or "recall" in name.lower() else f"{value:.3f}"
        col.metric(name, display)
    st.caption("These metrics describe only the synthetic demonstration dataset, not real production accuracy.")

st.divider()
calculate_col, info_col = st.columns([1, 3])
if calculate_col.button("Calculate packing", type="primary", use_container_width=True):
    current = normalize_input(st.session_state.ml_input)
    errors = validate_input(current)
    if errors:
        st.error("Please correct the input before calculation.")
        for error in errors[:20]:
            st.write(f"- {error}")
    else:
        with st.spinner("Calculating rules and ML predictions..."):
            bundle = get_models()
            comparison, pallets = predict_batch(current, bundle)
        st.session_state.ml_comparison = comparison
        st.session_state.ml_pallets = pallets
        st.success("Calculation complete.")
info_col.caption("Rule result is final. ML prediction is displayed to demonstrate learning and comparison.")

comparison = st.session_state.ml_comparison
pallets = st.session_state.ml_pallets
if not comparison.empty:
    st.subheader("Results")
    total_pallets = int(pallets.shape[0])
    total_ldm = float(pallets["Pallet LDM"].sum()) if not pallets.empty else 0.0
    total_weight = float(pallets["Pallet weight (kg)"].sum()) if not pallets.empty else 0.0
    manual_count = int((comparison["Rule manual review"] == "Yes").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product pallets", total_pallets)
    c2.metric("Total LDM", f"{total_ldm:.3f}")
    c3.metric("Packed weight", f"{total_weight:,.0f} kg")
    c4.metric("Manual review", manual_count)

    st.subheader("Rule result compared with ML")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if manual_count:
        st.warning("At least one construction exceeds a hard limit. ML cannot override Manual review.")

    st.subheader("Pallet visualisation")
    cards = "".join(pallet_html(row) for _, row in pallets.iterrows())
    st.markdown(f'<div class="pallet-grid">{cards}</div>', unsafe_allow_html=True)

    st.subheader("Pallet table")
    st.dataframe(pallets, use_container_width=True, hide_index=True)

    bundle = get_models()
    report = make_result_workbook(st.session_state.ml_input, comparison, pallets, bundle.metrics)
    st.download_button(
        "Download calculation result",
        data=report,
        file_name="packing_ml_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
