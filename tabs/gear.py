"""
tabs/gear.py — Gear & Shoe Tracker tab render function.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import _d_unit, compute_gear_stats
from auth import fetch_gear_api, get_valid_token
from config import KM_TO_MILES


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]

    activities = data["activities"]
    df_range = data["df_range"]
    gear_details = data["gear_details"]

    st.header("Gear & Shoe Tracker")

    gear_names_map = gear_details if gear_details else {}
    gear_df = compute_gear_stats(activities, gear_names_map)

    if len(gear_df) == 0:
        st.info("No gear data found. Gear info comes from Strava \u2014 make sure you've logged shoes on your activities.")
        return

    # Warning thresholds
    RETIRE_KM = 800
    WARN_KM = 600
    _g_factor = KM_TO_MILES if use_miles else 1.0
    _g_unit = _d_unit(use_miles)
    RETIRE_D = RETIRE_KM * _g_factor
    WARN_D = WARN_KM * _g_factor

    # KPI cards per shoe
    _gear_cols = st.columns(min(len(gear_df), 4))
    for _gi, (_, _grow) in enumerate(gear_df.iterrows()):
        _km = _grow["total_km"]
        _d = _km * _g_factor
        if _km >= RETIRE_KM:
            _delta_str = "Replace now"
            _delta_color = "inverse"
        elif _km >= WARN_KM:
            _delta_str = f"{RETIRE_D - _d:.0f} {_g_unit} left"
            _delta_color = "inverse"
        else:
            _delta_str = f"{RETIRE_D - _d:.0f} {_g_unit} remaining"
            _delta_color = "normal"
        _last_used_str = (
            _grow["last_used"].strftime("%b %d, %Y")
            if pd.notna(_grow["last_used"])
            else "N/A"
        )
        _gear_cols[_gi % 4].metric(
            label=_grow["name"],
            value=f"{_d:,.0f} {_g_unit}",
            delta=_delta_str,
            delta_color=_delta_color,
            help=f"{_grow['runs']} runs \u00b7 Last used: {_last_used_str}",
        )

    st.divider()

    # Bar chart: mileage per shoe with color coding
    gear_df["status"] = gear_df["total_km"].apply(
        lambda x: "Replace now" if x >= RETIRE_KM else ("Getting worn" if x >= WARN_KM else "Good")
    )
    gear_df["total_disp"] = gear_df["total_km"] * _g_factor
    _color_map_gear = {"Good": "#2ca02c", "Getting worn": "#fd8d3c", "Replace now": "#d62728"}
    fig_gear = px.bar(
        gear_df, x="name", y="total_disp", color="status",
        color_discrete_map=_color_map_gear,
        labels={"name": "Shoe / Gear", "total_disp": f"Total {_g_unit}"},
        title=f"Total mileage per shoe ({_g_unit})",
    )
    fig_gear.add_hline(y=WARN_D, line_dash="dot", line_color="#fd8d3c",
                       annotation_text=f"Warn ({WARN_D:.0f} {_g_unit})", annotation_position="top right")
    fig_gear.add_hline(y=RETIRE_D, line_dash="dash", line_color="#d62728",
                       annotation_text=f"Retire ({RETIRE_D:.0f} {_g_unit})", annotation_position="top right")
    fig_gear.update_layout(showlegend=True, xaxis_tickangle=-20)
    st.plotly_chart(fig_gear, use_container_width=True)

    # Monthly mileage per shoe (recent period)
    if "gear_id" in df_range.columns:
        d_gear_range = df_range[pd.notna(df_range["gear_id"]) & (df_range["gear_id"] != "")].copy()
        if len(d_gear_range) > 0:
            d_gear_range["gear_id"] = d_gear_range["gear_id"].astype(str)
            if gear_names_map:
                d_gear_range["shoe"] = d_gear_range["gear_id"].map(lambda g: gear_names_map.get(g, g))
            else:
                d_gear_range["shoe"] = d_gear_range["gear_id"]
            d_gear_range["month"] = d_gear_range["start_dt_local"].dt.to_period("M").astype(str)
            monthly_gear = d_gear_range.groupby(["month", "shoe"])["distance_km"].sum().reset_index()
            monthly_gear["distance_disp"] = monthly_gear["distance_km"] * _g_factor
            fig_monthly_gear = px.bar(
                monthly_gear, x="month", y="distance_disp", color="shoe",
                labels={"month": "Month", "distance_disp": _g_unit, "shoe": "Shoe"},
                title=f"Monthly {_g_unit} per shoe (selected period)",
                barmode="stack",
            )
            st.plotly_chart(fig_monthly_gear, use_container_width=True)

    # Raw table
    with st.expander("Full gear table"):
        _show_cols = ["name", "total_disp", "runs", "first_used", "last_used"]
        st.dataframe(
            gear_df[_show_cols].rename(columns={
                "name": "Shoe", "total_disp": f"Total {_g_unit}", "runs": "Runs",
                "first_used": "First use", "last_used": "Last use",
            }),
            use_container_width=True,
        )
