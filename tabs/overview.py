"""
tabs/overview.py — Overview tab render function.
"""
import numpy as np
import pandas as pd
import streamlit as st

from analytics import (
    _d_unit,
    _dist_fmt,
    _format_pace,
    _p_unit,
    _to_display_dist,
    build_calendar_heatmap,
    estimate_training_paces,
)
from config import KM_TO_MILES, RUN_TYPE_COLORS


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    consistency = data["consistency"]
    vo2max_est = data["vo2max_est"]
    vo2max_source = data.get("vo2max_source")
    df_range = data["df_range"]
    activities = data["activities"]
    streams_by_id = data["streams_by_id"]
    bests = data["bests"]

    # ── All-time KPIs ────────────────────────────────────────────────
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Total runs", f"{consistency['total_runs']:,}")
    o2.metric("Total distance", _dist_fmt(consistency['total_km'], use_miles, decimals=0))
    o3.metric("Week streak", f"{consistency['week_streak']} wks",
              help="Consecutive weeks with at least one run.")
    o4.metric("Consistent weeks (last 12)", f"{consistency['pct_consistent_weeks']:.0f}%",
              help="Weeks with \u22653 runs in the last 12 weeks.")
    if vo2max_est is not None:
        _src_label = vo2max_source or "best recorded effort"
        o5.metric("Estimated VO\u2082max", f"{vo2max_est:.1f} ml/kg/min",
                  help=f"VDOT estimate via Jack Daniels' formula from your {_src_label}. Not a lab test.")
    else:
        o5.metric("Estimated VO\u2082max", "\u2014", "Need a 5K\u2013marathon effort")

    st.divider()

    # ── Data completeness indicator ───────────────────────────────────
    st.subheader("Data quality \u2014 selected period")
    _n_total = len(df_range)
    _n_hr    = int(df_range["avg_hr"].notna().sum())
    _n_strm  = int(df_range["id"].astype(int).isin(streams_by_id.keys()).sum())
    _n_gps   = int(df_range["start_latlng"].notna().sum()) if "start_latlng" in df_range.columns else 0
    _n_elev  = int((df_range["total_elevation_gain"].fillna(0) > 0).sum()) if "total_elevation_gain" in df_range.columns else 0

    def _qual_color(pct: float) -> str:
        if pct >= 80: return "#2ca02c"
        if pct >= 50: return "#fd8d3c"
        return "#d62728"

    def _qual_badge(label: str, n: int, total: int, tip: str) -> str:
        pct = n / total * 100 if total > 0 else 0
        c = _qual_color(pct)
        icon = "\u2713" if pct >= 80 else ("\u25b3" if pct >= 50 else "\u2717")
        return (
            f"<div style='flex:1;text-align:center;padding:10px 6px;border-radius:8px;"
            f"border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03)'>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{c}'>{icon} {pct:.0f}%</div>"
            f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.7);margin-top:2px'>{label}</div>"
            f"<div style='font-size:0.7rem;color:rgba(255,255,255,0.4)'>{n}/{total} runs</div>"
            f"<div style='font-size:0.68rem;color:rgba(255,255,255,0.35);margin-top:3px'>{tip}</div>"
            f"</div>"
        )

    _badges = [
        _qual_badge("Heart Rate", _n_hr, _n_total, "Needed for training load, zones & classification"),
        _qual_badge("Streams / GPS", _n_strm, _n_total, "Needed for cadence, GAP & HR zone detail"),
        _qual_badge("GPS Location", _n_gps, _n_total, "Needed for route maps & weather overlay"),
        _qual_badge("Elevation", _n_elev, _n_total, "Needed for grade-adjusted pace (GAP)"),
    ]
    st.markdown(
        f"<div style='display:flex;gap:10px;margin-bottom:4px'>{''.join(_badges)}</div>"
        f"<div style='font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:4px'>"
        f"Based on {_n_total} runs in the selected date range. "
        f"Widen the date range and click <b>Refresh data</b> to fetch missing streams.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Training calendar heatmap ─────────────────────────────────────
    st.subheader("Training calendar \u2014 last 12 months")
    fig_cal = build_calendar_heatmap(activities, use_miles=use_miles)
    st.plotly_chart(fig_cal, use_container_width=True)
    # Colour legend
    legend_html = " &nbsp; ".join(
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;"
        f"background:{c};vertical-align:middle;margin-right:4px'></span>{t}"
        for t, c in RUN_TYPE_COLORS.items()
    )
    st.markdown(f"<small>{legend_html} &nbsp; <span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:#111111;border:1px solid #444;vertical-align:middle;margin-right:4px'></span>Rest</small>", unsafe_allow_html=True)

    st.divider()

    # ── Personal bests ────────────────────────────────────────────────
    st.subheader("Personal bests (all time)")
    pb_labels = {
        "best_5k":       ("Best 5K pace",           "5K"),
        "best_10k":      ("Best 10K pace",           "10K"),
        "best_hm":       ("Best half-marathon pace", "HM"),
        "best_marathon": ("Best marathon pace",      "Marathon"),
        "longest_run":   ("Longest run",             None),
    }
    pb_cols = st.columns(len(pb_labels))
    for col, (key, (label, tag)) in zip(pb_cols, pb_labels.items()):
        if key in bests:
            b = bests[key]
            dt_str = pd.to_datetime(b["date"]).strftime("%b %Y")
            if key == "longest_run":
                col.metric(label, _dist_fmt(b['distance_km'], use_miles), dt_str)
            else:
                col.metric(label, _format_pace(b["pace_min_per_km"], use_miles),
                           f"{_dist_fmt(b['distance_km'], use_miles)} \u00b7 {dt_str}")
        else:
            col.metric(label, "\u2014", "No qualifying run yet")

    st.divider()

    # ── Week-by-week training summary ─────────────────────────────────
    st.subheader("Weekly training log")
    st.caption("One row per week in the selected date range. Quality = Tempo / Workout / Race sessions.")

    _wlog = df_range.copy()
    _wlog["week_start"] = _wlog["start_dt_local"].dt.to_period("W-MON").apply(lambda p: p.start_time)

    _wlog_grp = _wlog.groupby("week_start", as_index=False).agg(
        runs=("id", "count"),
        distance_km=("distance_km", "sum"),
        duration_min=("duration_min", "sum"),
        long_run_km=("distance_km", "max"),
        avg_hr=("avg_hr", "mean"),
        elev_gain=("total_elevation_gain", "sum"),
    )

    # Quality run count per week
    if "run_type" in _wlog.columns:
        _q_wk = (
            _wlog[_wlog["run_type"].isin(["Tempo", "Workout", "Race"])]
            .groupby("week_start").size().reset_index(name="quality")
        )
        _wlog_grp = _wlog_grp.merge(_q_wk, on="week_start", how="left")
        _wlog_grp["quality"] = _wlog_grp["quality"].fillna(0).astype(int)
    else:
        _wlog_grp["quality"] = 0

    # Format for display
    _wlog_grp = _wlog_grp.sort_values("week_start", ascending=False)
    _wlog_disp = pd.DataFrame()
    _wlog_disp["Week"] = _wlog_grp["week_start"].dt.strftime("%b %d, %Y")
    _wlog_disp["Runs"] = _wlog_grp["runs"].astype(int)
    _wlog_disp[f"Distance ({_d_unit(use_miles)})"] = _wlog_grp["distance_km"].apply(
        lambda x: round(_to_display_dist(x, use_miles), 1)
    )
    _wlog_disp["Time"] = _wlog_grp["duration_min"].apply(
        lambda m: f"{int(m // 60)}h {int(m % 60):02d}m" if pd.notna(m) and m > 0 else "\u2014"
    )
    # Avg pace = total time / total distance
    _avg_pace_mkm = (_wlog_grp["duration_min"] / _wlog_grp["distance_km"]).replace([np.inf, -np.inf], np.nan)
    _wlog_disp[f"Avg pace ({_p_unit(use_miles)})"] = _avg_pace_mkm.apply(
        lambda x: _format_pace(x, use_miles) if pd.notna(x) else "\u2014"
    )
    _wlog_disp[f"Long run ({_d_unit(use_miles)})"] = _wlog_grp["long_run_km"].apply(
        lambda x: _dist_fmt(x, use_miles) if pd.notna(x) else "\u2014"
    )
    _wlog_disp["Quality"] = _wlog_grp["quality"].apply(lambda x: f"{int(x)} \U0001f525" if x > 0 else "\u2014")
    _wlog_disp["Avg HR"] = _wlog_grp["avg_hr"].apply(
        lambda x: f"{int(round(x))} bpm" if pd.notna(x) and x > 0 else "\u2014"
    )
    if "total_elevation_gain" in df_range.columns:
        _wlog_disp["Elev gain (m)"] = _wlog_grp["elev_gain"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "\u2014"
        )

    st.dataframe(_wlog_disp, hide_index=True, use_container_width=True)

    st.divider()

    # ── VO2max context ────────────────────────────────────────────────
    if vo2max_est is not None:
        st.subheader("Estimated VO\u2082max")
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            # Contextual classification (Firstbeat / ACSM guidelines for men 30-39 as rough reference)
            if vo2max_est >= 60:
                v_class, v_color = "Elite", "#52e88a"
            elif vo2max_est >= 52:
                v_class, v_color = "Excellent", "#3dba6e"
            elif vo2max_est >= 45:
                v_class, v_color = "Good", "#fd8d3c"
            elif vo2max_est >= 35:
                v_class, v_color = "Average", "#fdae6b"
            else:
                v_class, v_color = "Below average", "#d62728"
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:{v_color}'>{vo2max_est:.1f}</div>"
                f"<div style='font-size:1rem;color:{v_color}'>{v_class}</div>"
                f"<div style='font-size:0.8rem;color:grey;margin-top:4px'>ml \u00b7 kg\u207b\u00b9 \u00b7 min\u207b\u00b9 (VDOT estimate)</div>",
                unsafe_allow_html=True,
            )
        with v_col2:
            _src_label = vo2max_source or "best recorded effort"
            st.caption(
                f"VDOT is derived from your **{_src_label}** using Jack Daniels' formula. "
                "It reflects race-pace fitness \u2014 not a lab VO\u2082max. A 1-point increase typically means "
                "~1\u20132% faster race times. Improve it by adding easy aerobic volume and one quality session per week."
            )
            # Reference table
            ref = pd.DataFrame({
                "Level": ["Beginner", "Average", "Good", "Excellent", "Elite"],
                "VO\u2082max (ml/kg/min)": ["< 35", "35\u201344", "45\u201351", "52\u201359", "60+"],
                "~5K time": ["> 30 min", "25\u201330 min", "20\u201325 min", "17\u201320 min", "< 17 min"],
            })
            st.dataframe(ref, hide_index=True, use_container_width=True)

    # Jack Daniels training paces derived from VDOT
    st.subheader("Jack Daniels training paces")
    _src_label = vo2max_source or "best recorded effort"
    st.caption(
        f"Prescribed training paces calculated from your VDOT of **{vo2max_est:.1f}** "
        f"(derived from your {_src_label}). "
        "Each zone targets a specific physiological adaptation. "
        "Source: *Daniels' Running Formula* (2005)."
    )
    _pace_rows = estimate_training_paces(vo2max_est, use_miles=use_miles)
    _pace_df = pd.DataFrame(_pace_rows)
    st.dataframe(_pace_df, hide_index=True, use_container_width=True)
    st.caption(
        "\u2139\ufe0f These are effort-based target paces, not GPS-enforced zones. "
        "Easy pace is intentionally slow \u2014 most runners train their easy runs too fast."
    )
