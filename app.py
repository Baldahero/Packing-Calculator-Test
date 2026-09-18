"""Public Streamlit demonstration for construction packing."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from smart_packing import (
    INPUT_COLUMNS,
    aggregate_assignments,
    demo_project,
    empty_project,
    make_training_workbook,
    normalize_input,
    read_uploaded_workbook,
    summarize_allocation,
    suggest_packing,
    validate_allocation,
    validate_input,
)


st.set_page_config(page_title="Smart Packing Demo", page_icon="📦", layout="wide")

SAMPLE_PATH = Path(__file__).resolve().parent / "examples" / "Packing_Demo_Project_1.xlsx"
COLORS = ["#2F75B5", "#70AD47", "#ED7D31", "#8064A2", "#00A6A6", "#C55A11"]


def reset_result() -> None:
    st.session_state.allocation = pd.DataFrame()
    st.session_state.summary = pd.DataFrame()
    st.session_state.allocation_version = st.session_state.get("allocation_version", 0) + 1


def load_input(rows: pd.DataFrame) -> None:
    st.session_state.project_input = normalize_input(rows)
    st.session_state.input_version = st.session_state.get("input_version", 0) + 1
    reset_result()


def pallet_cards(allocation: pd.DataFrame, summary: pd.DataFrame) -> str:
    item_order = list(allocation["Item"].drop_duplicates())
    color_by_item = {item: COLORS[index % len(COLORS)] for index, item in enumerate(item_order)}
    max_height = max(float(allocation["Height (mm)"].max()), 1.0)
    cards: list[str] = []

    for _, pallet in summary.iterrows():
        number = int(pallet["Pallet no"])
        units = allocation[pd.to_numeric(allocation["Pallet no"]).astype(int) == number]
        shapes: list[str] = []
        for _, unit in units.iterrows():
            item = html.escape(str(unit["Item"]))
            height = max(46, round(116 * float(unit["Height (mm)"]) / max_height))
            width = max(34, min(64, round(34 + float(unit["Width (mm)"]) / 120)))
            color = color_by_item[str(unit["Item"])]
            shapes.append(
                f'<div class="construction" style="height:{height}px;width:{width}px;background:{color}" '
                f'title="{item}: {unit["Width (mm)"]:.0f} × {unit["Height (mm)"]:.0f} mm">{item}</div>'
            )

        status_class = "ok" if pallet["Status"] == "Within limits" else "invalid"
        cards.append(
            f"""
            <div class="pallet-card {status_class}">
              <div class="pallet-title">Pallet {number}</div>
              <div class="pallet-composition">{html.escape(str(pallet['Composition']))}</div>
              <div class="pallet-load">{''.join(shapes)}</div>
              <div class="pallet-base"></div>
              <div class="pallet-metrics">
                <span><b>{int(pallet['Units'])}/6</b> units</span>
                <span><b>{float(pallet['Weight (kg)']):.0f}/1000</b> kg</span>
                <span><b>{float(pallet['Pallet length (mm)']):.0f}</b> × 1200 mm</span>
                <span><b>{float(pallet['LDM']):.3f}</b> LDM</span>
              </div>
              <div class="pallet-status">{html.escape(str(pallet['Status']))}</div>
            </div>
            """
        )
    return '<div class="pallet-grid">' + "".join(cards) + "</div>"


st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 4rem;}
    .demo-steps {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:10px 0 20px}
    .demo-step {border:1px solid #d4dde5;border-radius:10px;padding:12px 14px;background:#f8fafc}
    .demo-step b {color:#18324a}
    .pallet-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin:12px 0 22px}
    .pallet-card {border:1px solid #aab9c5;border-radius:12px;padding:14px;background:white;box-shadow:0 2px 8px rgba(20,45,70,.08)}
    .pallet-card.invalid {border:2px solid #c2410c;background:#fff7ed}
    .pallet-title {font-size:18px;font-weight:700;color:#18324a}
    .pallet-composition {font-size:13px;color:#526575;margin:2px 0 10px}
    .pallet-load {min-height:130px;display:flex;align-items:flex-end;justify-content:center;gap:5px;padding:8px 8px 0;border-left:2px solid #9a6a31;border-right:2px solid #9a6a31;background:#fbf5e9;overflow:hidden}
    .construction {display:flex;align-items:center;justify-content:center;border:2px solid #18324a;color:white;font-size:10px;font-weight:700;border-radius:3px 3px 0 0;text-shadow:0 1px 2px rgba(0,0,0,.45)}
    .pallet-base {height:12px;background:#8b5e2b;border:2px solid #68451f}
    .pallet-metrics {display:grid;grid-template-columns:repeat(2,1fr);gap:5px 12px;font-size:12px;color:#435463;margin-top:10px}
    .pallet-status {font-size:12px;font-weight:700;color:#2e6b3b;margin-top:8px}
    .invalid .pallet-status {color:#9a3412}
    @media (max-width:800px){.demo-steps{grid-template-columns:1fr}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Smart Packing Demo")
st.caption("System-neutral demonstration for simple doors and windows")
st.info(
    "This is a workflow prototype, not a trained ML model yet. It uses constrained optimisation, "
    "then records production corrections as future training data."
)
st.markdown(
    """
    <div class="demo-steps">
      <div class="demo-step"><b>1. Import</b><br>Upload a construction list from Excel or edit it on screen.</div>
      <div class="demo-step"><b>2. Suggest</b><br>The app proposes mixed pallets within the fixed limits.</div>
      <div class="demo-step"><b>3. Confirm</b><br>Production corrects the result and exports a training example.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "project_input" not in st.session_state:
    st.session_state.project_input = demo_project()
if "allocation" not in st.session_state:
    st.session_state.allocation = pd.DataFrame()
if "summary" not in st.session_state:
    st.session_state.summary = pd.DataFrame()
if "allocation_version" not in st.session_state:
    st.session_state.allocation_version = 0
if "input_version" not in st.session_state:
    st.session_state.input_version = 0

project_tab, learning_tab = st.tabs(["Project", "How learning works"])

with project_tab:
    load_col, clear_col, sample_col, note_col = st.columns([1, 1, 1.5, 2.5])
    if load_col.button("Load Project 1", use_container_width=True):
        load_input(demo_project())
        st.rerun()
    if clear_col.button("Clear", use_container_width=True):
        load_input(empty_project())
        st.rerun()
    sample_col.download_button(
        "Download example Excel",
        data=SAMPLE_PATH.read_bytes(),
        file_name="Packing_Demo_Project_1.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    note_col.caption("Project 1 contains 20 constructions across six positions.")

    edited = st.data_editor(
        st.session_state.project_input,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=INPUT_COLUMNS,
        column_config={
            "Type": st.column_config.SelectboxColumn(options=["Door", "Window"], required=True),
            "Glass mode": st.column_config.SelectboxColumn(
                options=["Glazed", "Unglazed", "Without glass"], required=True
            ),
            "Width (mm)": st.column_config.NumberColumn(min_value=1, step=1, format="%.0f"),
            "Height (mm)": st.column_config.NumberColumn(min_value=1, step=1, format="%.0f"),
            "Qty": st.column_config.NumberColumn(min_value=1, step=1, format="%d"),
            "Frame weight (kg)": st.column_config.NumberColumn(min_value=0.1, step=0.1),
            "Glass weight (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
        },
        key=f"project_editor_{st.session_state.input_version}",
    )
    current_input = normalize_input(edited)
    st.session_state.project_input = current_input

    with st.expander("Import an Excel project"):
        uploaded = st.file_uploader("Upload an .xlsx file", type=["xlsx"])
        if uploaded is not None:
            try:
                imported, detected_sheet = read_uploaded_workbook(uploaded)
                st.caption(f"Detected sheet: {detected_sheet}. Rows found: {len(imported)}.")
                st.dataframe(imported, use_container_width=True, hide_index=True)
                if st.button("Use imported project", type="primary"):
                    load_input(imported)
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read the workbook: {exc}")

    calculate_col, limit_col = st.columns([1.4, 3.6])
    if calculate_col.button("Generate packing proposal", type="primary", use_container_width=True):
        errors = validate_input(current_input)
        if errors:
            st.error("Please correct the project input.")
            for error in errors:
                st.write(f"- {error}")
        else:
            try:
                with st.spinner("Finding a valid combination..."):
                    allocation, summary = suggest_packing(current_input)
                st.session_state.allocation = allocation
                st.session_state.summary = summary
                st.session_state.allocation_version += 1
                st.success("Packing proposal generated.")
            except Exception as exc:
                st.error(str(exc))
    limit_col.caption(
        "Demo limits: 6 units and 1000 kg per pallet; depth 1200 mm; "
        "pallet length +100 mm up to 3000 mm and +200 mm above 3000 mm."
    )

with learning_tab:
    st.markdown(
        """
        Safety limits remain fixed in the calculator and cannot be overridden by ML.

        Each completed project produces three connected datasets:

        1. the imported construction list;
        2. the automatically suggested pallet allocation;
        3. the allocation corrected and confirmed by production.

        Once enough real confirmed projects are collected, a model can learn which dimensions and
        construction combinations production usually places together. Excel import and this screen stay the same.
        """
    )

allocation = st.session_state.allocation
summary = st.session_state.summary
if not allocation.empty and not summary.empty:
    st.divider()
    st.subheader("Packing proposal")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Pallets", len(summary))
    metric_2.metric("Constructions", len(allocation))
    metric_3.metric("Packed weight", f"{allocation['Unit weight (kg)'].sum():,.0f} kg")
    metric_4.metric("Total LDM", f"{summary['LDM'].sum():.3f}")

    st.markdown(pallet_cards(allocation, summary), unsafe_allow_html=True)
    st.subheader("Suggested allocation")
    st.dataframe(aggregate_assignments(allocation), use_container_width=True, hide_index=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Adjust and confirm the production result"):
        st.caption("Change only the pallet number; all limits are recalculated after applying the change.")
        editable = allocation.copy()
        adjusted = st.data_editor(
            editable,
            hide_index=True,
            use_container_width=True,
            disabled=[column for column in editable.columns if column != "Pallet no"],
            column_config={"Pallet no": st.column_config.NumberColumn(min_value=1, step=1, format="%d")},
            key=f"allocation_editor_{st.session_state.allocation_version}",
        )
        if st.button("Apply production adjustment"):
            errors = validate_allocation(adjusted)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                adjusted_summary = summarize_allocation(adjusted)
                invalid = adjusted_summary[adjusted_summary["Status"] != "Within limits"]
                if not invalid.empty:
                    st.error("The adjusted plan exceeds a fixed packing limit.")
                    st.dataframe(invalid, use_container_width=True, hide_index=True)
                else:
                    st.session_state.allocation = adjusted
                    st.session_state.summary = adjusted_summary
                    st.session_state.allocation_version += 1
                    st.success("Production adjustment applied.")
                    st.rerun()

        export = make_training_workbook(
            st.session_state.project_input,
            st.session_state.allocation,
            st.session_state.summary,
        )
        st.download_button(
            "Download confirmed training example",
            data=export,
            file_name="confirmed_packing_example.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
