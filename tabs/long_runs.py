"""
tabs/long_runs.py — Long Runs (durability) tab render function.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    _d_unit,
    _dist_fmt,
    _p_unit,
    _safe_array,
    build_fatigue_table,
    build_within_run_df,
)
from config import KM_TO_MILES
from data_loader import streams_to_df


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    race_km = settings["race_km"]
    long_run_ratio_thresh = settings["long_run_ratio_thresh"]

    df_range = data["df_range"]
    streams_by_id = data["streams_by_id"]

    st.subheader("Long-run durability")

    long_run_min_km = float(long_run_ratio_thresh) * float(race_km)
    st.caption(f"Analysing runs \u2265 {_dist_fmt(long_run_min_km, use_miles)} (your threshold \u00d7 race distance). Adjust in the sidebar.")

    fatigue = build_fatigue_table(df_range, streams_by_id, long_run_min_km=long_run_min_km)

    if len(fatigue) == 0:
        st.info("No long runs with stream data in the selected range. Lower the long-run threshold or widen the date range.")
        return

    med_pf  = np.nanmedian(fatigue["pace_fade_pct"])
    med_hrd = np.nanmedian(fatigue["hr_drift_pct"])
    med_dec = np.nanmedian(fatigue["decoupling"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Long runs analysed", f"{len(fatigue)}")
    k2.metric("Median pace fade", f"{med_pf*100:.1f}%",
              help="<5% = good \u00b7 5\u201310% = moderate \u00b7 >10% = significant fade")
    k3.metric("Median HR drift",  f"{med_hrd*100:.1f}%" if np.isfinite(med_hrd) else "N/A",
              help="How much HR rose in the second half vs first at similar pace. Higher = more cardiovascular stress.")
    k4.metric("Median aerobic efficiency loss", f"{med_dec*100:.1f}%" if np.isfinite(med_dec) else "N/A",
              help="Drop in speed\u00f7HR from first to second half. Negative = efficiency declined. Closer to 0 = better durability.")

    st.divider()

    st.subheader("Fatigue trends over time")
    t1, t2 = st.columns(2)
    with t1:
        fig_pf = px.line(
            fatigue.sort_values("start_dt_local"),
            x="start_dt_local", y="pace_fade_pct",
            markers=True,
            title="Pace fade per long run",
            labels={"start_dt_local": "Date", "pace_fade_pct": "Pace fade"},
        )
        fig_pf.add_hline(y=0.05, line_dash="dash", line_color="orange",
                         annotation_text="5% \u2014 moderate", annotation_position="top left")
        fig_pf.add_hline(y=0.10, line_dash="dash", line_color="red",
                         annotation_text="10% \u2014 significant", annotation_position="top left")
        fig_pf.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pf, use_container_width=True)

    with t2:
        fig_hr = px.line(
            fatigue.sort_values("start_dt_local"),
            x="start_dt_local", y="hr_drift_pct",
            markers=True,
            title="HR drift per long run",
            labels={"start_dt_local": "Date", "hr_drift_pct": "HR drift"},
        )
        fig_hr.add_hline(y=0.05, line_dash="dash", line_color="orange",
                         annotation_text="5% drift", annotation_position="top left")
        fig_hr.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_hr, use_container_width=True)

    st.subheader("Inspect a long run")
    fatigue = fatigue.sort_values("start_dt_local", ascending=False)

    def _label_row(r):
        dt_str = pd.to_datetime(r["start_dt_local"]).strftime("%Y-%m-%d")
        return f"{dt_str} \u2014 {r['distance_km']:.1f} km \u2014 {str(r.get('name',''))[:40]}"

    options = { _label_row(r): int(r["id"]) for _, r in fatigue.iterrows() }
    selected_label = st.selectbox("Select a run", list(options.keys()))
    selected_id = options[selected_label]

    s = streams_by_id.get(int(selected_id), {})
    run_df = build_within_run_df(s)

    if run_df is None or len(run_df) == 0:
        st.warning("Selected run is missing required streams (distance/time/velocity).")
        return

    # ── Grade-adjusted pace ──────────────────────────────────────
    raw_s = s  # streams dict for selected run
    grade_arr = _safe_array(raw_s, "grade_smooth")
    if grade_arr is not None and len(grade_arr) == len(run_df):
        g = grade_arr / 100.0  # percent -> fraction
        # Minetti et al. metabolic cost: C(g) = 280.5g^5 - 58.7g^4 - 76.8g^3 + 51.9g^2 + 19.6g + 2.5
        g = np.clip(g, -0.40, 0.40)
        c_g = 280.5*g**5 - 58.7*g**4 - 76.8*g**3 + 51.9*g**2 + 19.6*g + 2.5
        c_flat = 2.5
        gap_factor = c_flat / np.where(c_g > 0.5, c_g, 0.5)
        gap_pace_raw = run_df["pace_min_km"].values * gap_factor
        gap_pace_raw = np.where(np.isfinite(gap_pace_raw) & (gap_pace_raw < 20), gap_pace_raw, np.nan)
        run_df["gap_pace"] = pd.Series(gap_pace_raw, index=run_df.index).rolling(30, min_periods=1).median()
        has_gap = True
    else:
        has_gap = False

    # ── Pace + HR dual-axis chart ────────────────────────────────
    fig = go.Figure()
    _lr_pace_factor = (1.0 / KM_TO_MILES) if use_miles else 1.0
    _lr_dist_factor = KM_TO_MILES if use_miles else 1.0
    fig.add_trace(go.Scatter(x=run_df["distance_km"] * _lr_dist_factor,
                             y=run_df["pace_smooth"] * _lr_pace_factor,
                             mode="lines", name="Actual pace (smoothed)"))
    if has_gap:
        fig.add_trace(go.Scatter(x=run_df["distance_km"] * _lr_dist_factor,
                                 y=run_df["gap_pace"] * _lr_pace_factor,
                                 mode="lines", name="Grade-adjusted pace (GAP)",
                                 line=dict(dash="dot", color="#fd8d3c")))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=f"Distance ({_d_unit(use_miles)})", yaxis_title=f"Pace ({_p_unit(use_miles)})",
    )
    fig.update_yaxes(autorange="reversed")

    if np.any(np.isfinite(run_df["hr_smooth"])):
        fig.add_trace(go.Scatter(x=run_df["distance_km"] * _lr_dist_factor, y=run_df["hr_smooth"],
                                 mode="lines", name="HR (smoothed)", yaxis="y2"))
        fig.update_layout(
            yaxis2=dict(title="Heart rate (bpm)", overlaying="y", side="right", showgrid=False)
        )

    st.plotly_chart(fig, use_container_width=True)
    if has_gap:
        st.caption("GAP (grade-adjusted pace) normalises for elevation using the Minetti metabolic cost formula \u2014 it shows what your flat equivalent effort was.")

    # ── Route map ───────────────────────────────────────────────
    stream_df = streams_to_df(raw_s)
    if "lat" in stream_df.columns and "lng" in stream_df.columns:
        map_df = stream_df.dropna(subset=["lat", "lng"]).copy()
        if len(map_df) > 10:
            st.subheader("Route map")
            # Colour by distance to show progression
            map_df["distance_km_col"] = pd.to_numeric(map_df.get("distance_km", np.nan), errors="coerce")
            fig_map = px.line_mapbox(
                map_df, lat="lat", lon="lng",
                mapbox_style="open-street-map",
                zoom=12,
            )
            fig_map.update_traces(line=dict(color="#3dba6e", width=3))
            fig_map.update_layout(
                height=420, margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)
