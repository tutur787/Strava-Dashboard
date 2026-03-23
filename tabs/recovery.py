"""
tabs/recovery.py — Recovery & Risk tab render function.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    _format_pace,
    compute_compromised_runs,
    compute_risk_table,
)


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    max_hr = settings["max_hr"]
    readiness_window_days = settings["readiness_window_days"]

    daily_all = data["daily_all"]
    weekly_all = data["weekly_all"]
    df_range = data["df_range"]

    st.subheader("Readiness, recovery & injury-risk proxies")

    daily, weekly = daily_all.copy(), weekly_all.copy()
    if len(daily) == 0 or len(weekly) == 0:
        st.info("Not enough data in the selected range to compute readiness.")
        return

    daily_risk, weekly_risk = compute_risk_table(daily, weekly)

    # Focus window
    end_focus = daily_risk["date_ts"].max()
    start_focus = end_focus - pd.Timedelta(days=int(readiness_window_days))
    d_focus = daily_risk[(daily_risk["date_ts"] >= start_focus) & (daily_risk["date_ts"] <= end_focus)].copy()
    w_focus = weekly_risk[weekly_risk["week_start"] >= (start_focus - pd.Timedelta(days=7))].copy()

    latest = d_focus.sort_values("date_ts").iloc[-1]
    risk_score = float(latest["risk_score"])
    acwr_val = latest["acwr"]

    # Simple readiness label
    if risk_score < 25:
        readiness = ("Low risk / likely fresh", "\U0001f7e2")
    elif risk_score < 55:
        readiness = ("Moderate risk", "\U0001f7e0")
    else:
        readiness = ("High risk", "\U0001f534")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overreach/injury risk", f"{risk_score:.0f}/100", f"{readiness[1]} {readiness[0]}",
              help="Composite proxy based on ACWR, recent load spikes, and rest days. Not medical advice.")
    c2.metric("ACWR", f"{acwr_val:.2f}" if pd.notna(acwr_val) else "N/A",
              help="0.8\u20131.3 = balanced. Above 1.5 = elevated overreach risk.")
    c3.metric("Rest days (last 7)", f"{int(latest['rest_days_last7'])}",
              help="Days with near-zero HR load. Aim for at least 1\u20132 per week.")
    c4.metric("Acute load (7d EWMA)", f"{latest['acute_load']:.1f}")

    st.caption("These are *proxies* \u2014 not medical advice. Use them as signals, not verdicts.")

    # Risk score trend
    fig_risk = px.area(
        d_focus,
        x="date_ts",
        y="risk_score",
        title="Composite risk score (recent window)",
        labels={"date_ts": "Date", "risk_score": "Risk score (0\u2013100)"},
    )
    fig_risk.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_risk, use_container_width=True)

    # Stacked flags chart -- shows which flags fired each day
    flag_map = {
        "flag_acwr_high":      "ACWR high (>1.5)",
        "flag_acwr_very_high": "ACWR very high (>1.8)",
        "flag_low_rest":       "Low rest (<2 days/wk)",
        "flag_big_day":        "Outlier load day",
    }
    flag_colors = {
        "ACWR high (>1.5)":      "#fd8d3c",
        "ACWR very high (>1.8)": "#d62728",
        "Low rest (<2 days/wk)": "#6baed6",
        "Outlier load day":      "#9e9ac8",
    }
    flags_long = d_focus[["date_ts"] + list(flag_map.keys())].copy()
    flags_long = flags_long.rename(columns=flag_map)
    flags_melted = flags_long.melt(id_vars="date_ts", var_name="Flag", value_name="Active")
    flags_melted = flags_melted[flags_melted["Active"] == 1]

    if len(flags_melted) == 0:
        st.success("No alert flags in the selected window.")
    else:
        fig_flags = px.bar(
            flags_melted, x="date_ts", y="Active", color="Flag",
            title="Alert flags per day \u2014 hover to see which flags fired",
            labels={"date_ts": "Date", "Active": "Flags"},
            color_discrete_map=flag_colors,
            barmode="stack",
        )
        fig_flags.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig_flags, use_container_width=True)

    # Weekly risk: load spikes & monotony
    st.subheader("Weekly stress patterns")
    r1, r2 = st.columns(2)

    w_focus = w_focus.copy()
    w_focus["strain_size"] = pd.to_numeric(w_focus.get("strain"), errors="coerce").fillna(0.0)
    w_focus.loc[w_focus["strain_size"] < 0, "strain_size"] = 0.0

    with r1:
        fig_spike = px.bar(
            w_focus,
            x="week_start",
            y="load_change_pct",
            title="Weekly load change vs 4-week average",
            labels={"week_start":"Week start", "load_change_pct":"Change (fraction)"},
        )
        fig_spike.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_spike, use_container_width=True)

    with r2:
        fig_sc = px.scatter(
            w_focus,
            x="monotony",
            y="weekly_load_hr",
            size="strain_size",
            hover_data=["week_start", "strain"],
            title="Load vs monotony (size = strain)",
            labels={"monotony":"Monotony", "weekly_load_hr":"Weekly load"},
        )
        fig_sc.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # Low-efficiency run detector
    st.subheader("Low-efficiency run detector")
    st.caption("Flags runs where your speed-per-HR was below your rolling 20th-percentile baseline \u2014 a signal of fatigue, illness, or heat stress.")
    comp = compute_compromised_runs(df_range, max_hr=max_hr)
    if len(comp) == 0 or comp["eff_q20"].isna().all():
        st.info("Not enough HR-bearing runs to detect outliers yet (need ~6+).")
        return

    comp_recent = comp[comp["start_dt_local"] >= start_focus].copy()
    n_flagged = int(comp_recent["compromised"].sum())
    rate = 100.0 * comp_recent["compromised"].mean() if len(comp_recent) else np.nan

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Low-efficiency runs (recent)", f"{n_flagged}/{len(comp_recent)}", f"{rate:.0f}% of runs")
        comp_recent["Status"] = comp_recent["compromised"].map({1: "Low efficiency", 0: "Normal"})
        fig_comp = px.scatter(
            comp_recent,
            x="start_dt_local", y="speed_per_hr",
            color="Status",
            color_discrete_map={"Low efficiency": "#d62728", "Normal": "#74c476"},
            title="Efficiency over time",
            labels={"start_dt_local": "Date", "speed_per_hr": "Speed \u00f7 HR (m/s per bpm)"},
        )
        fig_comp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

    with c2:
        fig_delta = px.histogram(
            comp_recent.dropna(subset=["eff_delta"]),
            x="eff_delta", nbins=25,
            title="Efficiency vs your baseline",
            labels={"eff_delta": "Speed/HR minus 20th-percentile baseline"},
        )
        fig_delta.add_vline(x=0, line_dash="dash", line_color="red",
                            annotation_text="Baseline", annotation_position="top right")
        fig_delta.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_delta, use_container_width=True)

    with st.expander("Show low-efficiency run table"):
        cols = ["start_dt_local","name","distance_km","duration_min","avg_hr","pace_min_per_km","speed_per_hr","eff_q20","eff_delta","compromised"]
        display = comp_recent[cols].copy()
        display["pace_min_per_km"] = display["pace_min_per_km"].apply(lambda x: _format_pace(x, use_miles))
        display["compromised"] = display["compromised"].map({1: "⚠️ Yes", 0: "✓ No"})
        display = display.rename(columns={
            "start_dt_local": "Date", "name": "Run name",
            "distance_km": "Distance (km)", "duration_min": "Duration (min)",
            "avg_hr": "Avg HR", "pace_min_per_km": "Pace",
            "speed_per_hr": "Speed/HR", "eff_q20": "Baseline (q20)",
            "eff_delta": "Δ vs baseline", "compromised": "Low efficiency?",
        })
        st.dataframe(display.sort_values("Date", ascending=False), hide_index=True, use_container_width=True)
