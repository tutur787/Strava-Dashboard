"""
tabs/pace.py — Pace & Efficiency tab render function.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import _d_unit, _format_pace, _p_unit
from config import KM_TO_MILES, RUN_TYPE_COLORS


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    max_hr = settings["max_hr"]
    effort_band = settings["effort_band"]
    _pace_factor = settings["_pace_factor"]

    df_range = data["df_range"]
    cadence_df = data["cadence_df"]
    streams_by_id = data["streams_by_id"]

    st.subheader("Pace, effort & efficiency")

    d2 = df_range.copy()
    d2["hr_intensity"] = d2["avg_hr"] / float(max_hr)
    d2 = d2[pd.notna(d2["avg_hr"]) & pd.notna(d2["pace_min_per_km"])]

    if len(d2) == 0:
        st.info("No runs with both average HR and average speed in the selected range.")
        return

    lo, hi = effort_band
    d2["in_race_effort_band"] = (d2["hr_intensity"] >= lo) & (d2["hr_intensity"] <= hi)
    # Compute display-unit columns once so all charts below stay consistent
    d2["pace_disp"]     = d2["pace_min_per_km"] * _pace_factor
    d2["distance_disp"] = d2["distance_km"] * (KM_TO_MILES if use_miles else 1.0)
    _pace_lbl = f"Pace ({_p_unit(use_miles)})"
    _dist_lbl = f"Distance ({_d_unit(use_miles)})"

    d2["speed_per_hr"] = d2["avg_speed_mps"] / d2["avg_hr"]

    d2 = d2.sort_values("start_dt_local")
    race_eff = d2[d2["in_race_effort_band"]].copy()

    top = st.columns(4)
    top[0].metric("Runs in range", f"{len(d2)}")
    top[1].metric("In race-effort band", f"{len(race_eff)}", help=f"HR band = {lo:.0%}\u2013{hi:.0%} of max HR")
    top[2].metric("Median pace", _format_pace(float(d2["pace_min_per_km"].median()), use_miles))
    top[3].metric("Median HR", f"{int(round(d2['avg_hr'].median()))} bpm")

    if len(race_eff) == 0:
        st.warning(
            f"No runs fall in your race-effort HR band ({lo:.0%}\u2013{hi:.0%} of {max_hr} bpm max HR). "
            "The efficiency trend below will use all runs instead. "
            "Widen the band in the sidebar to capture more efforts."
        )
    else:
        st.caption(f"Race-effort band: {lo:.0%}\u2013{hi:.0%} of max HR. Adjustable in the sidebar.")

    c1, c2 = st.columns([2, 1])

    with c1:
        scatter_color = "run_type" if "run_type" in d2.columns else "distance_km"
        scatter_cmap  = RUN_TYPE_COLORS if scatter_color == "run_type" else None
        fig_scatter = px.scatter(
            d2,
            x="avg_hr",
            y="pace_disp",
            color=scatter_color,
            color_discrete_map=scatter_cmap,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="white",
            hover_data=["name", "start_dt_local", "distance_disp", "duration_min", "total_elevation_gain", "hr_intensity"],
            labels={"avg_hr": "Average HR (bpm)", "pace_disp": _pace_lbl,
                    "run_type": "Type", "distance_disp": _dist_lbl},
            title="Pace vs HR \u2014 coloured by run type",
            category_orders={"run_type": list(RUN_TYPE_COLORS.keys())},
        )
        fig_scatter.update_yaxes(autorange="reversed")
        fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        fig_dist = px.histogram(
            d2,
            x="pace_disp",
            nbins=25,
            title="Pace distribution",
            labels={"pace_disp": _pace_lbl, "count": "Runs"},
        )
        fig_dist.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        fig_dist.update_xaxes(autorange="reversed")
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("X-axis is reversed \u2014 bars further left represent faster paces.")

    st.subheader("Efficiency trend (race-effort runs)")
    trend_df = race_eff.copy() if len(race_eff) >= 3 else d2.copy()
    if len(race_eff) < 3:
        st.info("Not enough race-effort runs for a stable trend yet (need ~3+). Showing all runs instead.")

    trend_df["pace_roll"] = trend_df["pace_min_per_km"].rolling(window=5, min_periods=1).median()
    trend_df["pace_disp"] = trend_df["pace_min_per_km"] * _pace_factor
    trend_df["pace_roll_disp"] = trend_df["pace_roll"] * _pace_factor

    fig_pace = go.Figure()
    fig_pace.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["pace_disp"],
                                  mode="markers", name="Pace",
                                  marker=dict(size=6, opacity=0.6)))
    fig_pace.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["pace_roll_disp"],
                                  mode="lines", name="Rolling median (5 runs)",
                                  line=dict(width=2.5)))
    fig_pace.update_layout(
        height=340,
        title="Pace trend at comparable effort",
        xaxis_title="Date",
        yaxis_title=_pace_lbl,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_pace.update_yaxes(autorange="reversed")
    # Trend direction annotation
    if len(trend_df) >= 5:
        _roll = trend_df["pace_roll_disp"].dropna()
        if len(_roll) >= 2:
            _pace_delta = float(_roll.iloc[-1] - _roll.iloc[0])   # positive = slower
            _delta_sec = abs(_pace_delta) * 60
            _trend_unit = "min/mi" if use_miles else "min/km"
            if abs(_pace_delta) < 0.05:
                fig_pace.add_annotation(
                    text="\u2192 Pace stable over the period", xref="paper", yref="paper",
                    x=0.01, y=0.04, showarrow=False,
                    font=dict(size=11, color="rgba(200,200,200,0.7)"),
                )
            elif _pace_delta < 0:
                fig_pace.add_annotation(
                    text=f"\u2197 {_delta_sec:.0f}s/{_trend_unit} faster over the period",
                    xref="paper", yref="paper", x=0.01, y=0.04, showarrow=False,
                    font=dict(size=11, color="rgba(61,186,110,0.9)"),
                )
            else:
                fig_pace.add_annotation(
                    text=f"\u2198 {_delta_sec:.0f}s/{_trend_unit} slower over the period",
                    xref="paper", yref="paper", x=0.01, y=0.04, showarrow=False,
                    font=dict(size=11, color="rgba(253,141,60,0.9)"),
                )
    st.plotly_chart(fig_pace, use_container_width=True)

    st.divider()

    # ── Cadence analysis ─────────────────────────────────────────
    st.subheader("Running cadence")
    if len(cadence_df) == 0:
        st.info("No cadence data found in streams for the selected range. Cadence requires a GPS watch with step cadence recording.")
    else:
        cad_c1, cad_c2, cad_c3 = st.columns(3)
        cad_c1.metric("Avg cadence", f"{cadence_df['avg_cadence'].mean():.0f} spm",
                      help="Steps per minute (both feet). 170\u2013180 spm is the commonly cited target range.")
        cad_c2.metric("% runs \u2265 170 spm", f"{(cadence_df['avg_cadence'] >= 170).mean()*100:.0f}%")
        cad_c3.metric("% runs \u2265 180 spm", f"{(cadence_df['avg_cadence'] >= 180).mean()*100:.0f}%")

        fig_cad = go.Figure()
        fig_cad.add_hrect(
            y0=170, y1=180, fillcolor="rgba(61,186,110,0.15)", line_width=0,
            annotation_text="\u2713 Optimal range (170\u2013180 spm)",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="rgba(61,186,110,0.85)"),
        )
        fig_cad.add_trace(go.Scatter(
            x=cadence_df["date"], y=cadence_df["avg_cadence"],
            mode="markers+lines", name="Avg cadence",
            marker=dict(size=7, color="#6baed6"),
            line=dict(color="#6baed6", width=2),
        ))
        fig_cad.update_layout(
            height=320, title="Cadence trend",
            xaxis_title="Date", yaxis_title="Steps per min (both feet)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_cad, use_container_width=True)

        # Cadence distribution
        all_cad_vals = []
        for _, row in df_range.iterrows():
            aid = int(row["id"])
            cad_obj = streams_by_id.get(aid, {}).get("cadence", {})
            if isinstance(cad_obj, dict) and isinstance(cad_obj.get("data"), list):
                vals = np.array(cad_obj["data"], dtype=float) * 2
                vals = vals[np.isfinite(vals) & (vals > 100) & (vals < 240)]
                all_cad_vals.extend(vals.tolist())

        if all_cad_vals:
            fig_cad_hist = px.histogram(
                x=all_cad_vals, nbins=40,
                title="Cadence distribution (all runs in range)",
                labels={"x": "Cadence (spm)", "y": "Seconds"},
            )
            fig_cad_hist.add_vline(x=170, line_dash="dash", line_color="orange")
            fig_cad_hist.add_vline(x=180, line_dash="dash", line_color="green")
            fig_cad_hist.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_cad_hist, use_container_width=True)
        st.caption(
            "Higher cadence (shorter, quicker steps) generally reduces injury risk and improves running economy. "
            "If your average is below 165 spm, try increasing by 5% every few weeks."
        )

    st.divider()

    # ── Weather: pace vs temperature ──────────────────────────────────────
    if "temp_c" in df_range.columns and df_range["temp_c"].notna().sum() >= 5:
        st.subheader("Pace vs temperature")
        _wdf = df_range[pd.notna(df_range["temp_c"]) & pd.notna(df_range["pace_min_per_km"])].copy()
        _wdf["pace_disp"] = _wdf["pace_min_per_km"] * _pace_factor
        _wdf["pace_fmt"] = _wdf["pace_disp"].apply(
            lambda v: f"{int(v)}:{int(round((v % 1)*60)):02d}/{_d_unit(use_miles)}" if pd.notna(v) else ""
        )
        _wdf["distance_disp"] = _wdf["distance_km"] * (KM_TO_MILES if use_miles else 1.0)
        fig_weather = px.scatter(
            _wdf, x="temp_c", y="pace_disp",
            color="run_type" if "run_type" in _wdf.columns else None,
            color_discrete_map=RUN_TYPE_COLORS,
            hover_data={"pace_fmt": True, "distance_disp": ":.1f", "temp_c": ":.1f"},
            labels={"temp_c": "Temperature (\u00b0C)", "pace_disp": _pace_lbl},
            title="Pace vs temperature",
            trendline="ols",
        )
        fig_weather.update_yaxes(autorange="reversed")
        fig_weather.update_layout(yaxis_tickformat=".2f")
        st.plotly_chart(fig_weather, use_container_width=True)
        st.caption("Lower pace = faster. A downward trendline suggests you run faster in cooler conditions.")
