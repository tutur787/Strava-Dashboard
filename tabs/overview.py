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


def _acsm_vo2max_classify(vo2max: float, age: int, gender: str):
    """
    Return (label, color, thresholds_dict) using ACSM age/sex-specific norms.
    Thresholds: [Poor_max, Fair_max, Good_max, Excellent_max] — above Excellent = Superior/Elite.
    """
    # ACSM norms (ml/kg/min): [Poor_hi, Fair_hi, Good_hi, Excellent_hi]
    # Source: ACSM's Guidelines for Exercise Testing and Prescription, 10th ed.
    _men = {
        (15, 29): [37, 43, 50, 55],
        (30, 39): [33, 38, 48, 53],
        (40, 49): [29, 35, 44, 50],
        (50, 59): [24, 30, 38, 45],
        (60, 99): [19, 25, 33, 40],
    }
    _women = {
        (15, 29): [28, 34, 43, 48],
        (30, 39): [26, 31, 38, 44],
        (40, 49): [23, 28, 35, 41],
        (50, 59): [20, 24, 31, 37],
        (60, 99): [17, 22, 28, 35],
    }
    _gender_lower = gender.lower()
    if _gender_lower.startswith("m"):
        table = _men
    elif _gender_lower.startswith("w"):
        table = _women
    else:
        # "Non-binary" / "Prefer not to say" — average men and women thresholds
        table = None

    thresholds = [37, 43, 50, 55]  # fallback (men 30-39)
    if table is not None:
        for (lo, hi), vals in table.items():
            if lo <= age <= hi:
                thresholds = vals
                break
    else:
        # Average the two tables for the matching age bracket
        _m_thresh = [37, 43, 50, 55]
        _w_thresh = [26, 31, 38, 44]
        for (lo, hi), vals in _men.items():
            if lo <= age <= hi:
                _m_thresh = vals
                break
        for (lo, hi), vals in _women.items():
            if lo <= age <= hi:
                _w_thresh = vals
                break
        thresholds = [round((m + w) / 2) for m, w in zip(_m_thresh, _w_thresh)]
    poor_hi, fair_hi, good_hi, exc_hi = thresholds
    if vo2max > exc_hi:
        return "Superior / Elite", "#52e88a", thresholds
    elif vo2max > good_hi:
        return "Excellent", "#3dba6e", thresholds
    elif vo2max > fair_hi:
        return "Good", "#fd8d3c", thresholds
    elif vo2max > poor_hi:
        return "Fair", "#fdae6b", thresholds
    else:
        return "Poor", "#d62728", thresholds


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    age = settings.get("age", 35)
    gender = settings.get("gender", "Men")
    consistency = data["consistency"]
    vo2max_est = data["vo2max_est"]
    vo2max_low = data.get("vo2max_low")
    vo2max_high = data.get("vo2max_high")
    vo2max_source       = data.get("vo2max_source")
    vo2max_effort_date    = data.get("vo2max_effort_date")
    vo2max_effort_source  = data.get("vo2max_effort_source", "activity")
    vo2max_effort_pace    = data.get("vo2max_effort_pace")
    vo2max_effort_dist_km = data.get("vo2max_effort_dist_km")
    vo2max_effort_n       = data.get("vo2max_effort_n", 1)
    vo2max_effort_dates   = data.get("vo2max_effort_dates", [])
    vo2max_is_recent      = data.get("vo2max_is_recent", True)
    vo2max_needs_streams  = data.get("vo2max_needs_streams", False)
    vo2max_submax         = data.get("vo2max_submax", {})
    _sm_val_early = (vo2max_submax or {}).get("vo2max")  # available early for KPI card
    df_range = data["df_range"]
    activities = data["activities"]
    streams_by_id = data["streams_by_id"]
    bests = data["bests"]

    # ── All-time KPIs ────────────────────────────────────────────────
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Total runs", f"{consistency['total_runs']:,}")
    o2.metric("Total distance", _dist_fmt(consistency['total_km'], use_miles, decimals=0))
    o3.metric("Week streak", f"{consistency['week_streak']} wks",
              help="Consecutive weeks with at least one run.")
    o4.metric("Consistent weeks (last 12)", f"{consistency['pct_consistent_weeks']:.0f}%",
              help="Weeks with \u22653 runs in the last 12 weeks.")
    # Total elevation gain in selected period
    _total_elev = (
        df_range["total_elevation_gain"].fillna(0).sum()
        if "total_elevation_gain" in df_range.columns else 0
    )
    o6.metric(
        "Elevation gain",
        f"{int(_total_elev):,} m" if _total_elev > 0 else "\u2014",
        help="Total ascent in the selected date range. High-elevation running elevates HR at any given pace \u2014 "
             "factor this in when comparing paces across hilly and flat routes.",
    )
    # KPI card: show the higher of pace-based VDOT and HR-based estimate
    _kpi_candidates = {k: v for k, v in {"pace": vo2max_est, "hr": _sm_val_early}.items() if v is not None}
    if _kpi_candidates:
        _kpi_method = max(_kpi_candidates, key=lambda k: _kpi_candidates[k])
        _kpi_val = _kpi_candidates[_kpi_method]
        _kpi_label = "Aerobic HR estimate" if _kpi_method == "hr" else (vo2max_source or "pace-based VDOT")
        _src_label = vo2max_source or "best recorded effort"
        if _kpi_method == "pace" and vo2max_low is not None and vo2max_high is not None:
            _range_str = f"{vo2max_low:.0f}–{vo2max_high:.0f} across distances"
        elif _kpi_method == "hr":
            _sm_low_e  = (vo2max_submax or {}).get("vo2max_low")
            _sm_high_e = (vo2max_submax or {}).get("vo2max_high")
            _range_str = (
                f"IQR {_sm_low_e:.0f}–{_sm_high_e:.0f} · HR method"
                if _sm_low_e is not None and _sm_high_e is not None
                else "HR method"
            )
        else:
            _range_str = _kpi_label
        o5.metric(
            "Estimated VO\u2082max",
            f"{_kpi_val:.1f} ml/kg/min",
            _range_str,
            help=(
                f"Showing the higher of your two estimates: **{_kpi_label}** ({_kpi_val:.1f}). "
                "Pace-based VDOT uses your best race efforts; the HR-based method uses steady aerobic training runs. "
                "The higher value is shown as it better reflects your aerobic ceiling — especially during a training block. "
                "See the VO\u2082max section below for both estimates and a comparison."
            ),
        )
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
    st.caption("All-time personal bests \u2014 not filtered by the selected date range.")
    # m1: show finish time as headline, pace + full date as subtext
    pb_labels = {
        "best_5k":       ("Best 5K",           5.0),
        "best_10k":      ("Best 10K",           10.0),
        "best_hm":       ("Best half-marathon", 21.0975),
        "best_marathon": ("Best marathon",      42.195),
        "longest_run":   ("Longest run",        None),
    }
    pb_cols = st.columns(len(pb_labels))
    for col, (key, (label, dist_km)) in zip(pb_cols, pb_labels.items()):
        if key in bests:
            b = bests[key]
            full_date = pd.to_datetime(b["date"]).strftime("%d %b %Y")
            if key == "longest_run":
                col.metric(label, _dist_fmt(b["distance_km"], use_miles), full_date)
            else:
                # Finish time from best (fastest) pace
                _best_pace = b.get("pace_min_per_km_best", b["pace_min_per_km"])
                _total_min = _best_pace * dist_km
                _h = int(_total_min // 60)
                _m = int(_total_min % 60)
                _s = int(round((_total_min - int(_total_min)) * 60))
                if _s == 60: _s = 0; _m += 1
                _time_str = f"{_h}:{_m:02d}:{_s:02d}" if _h > 0 else f"{_m}:{_s:02d}"
                _n = b.get("n_efforts", 1)
                _n_label = f"median of {_n}" if _n > 1 else "single effort"
                col.metric(
                    label, _time_str,
                    f"{_format_pace(_best_pace, use_miles)} · {full_date}",
                    help=f"Fastest recorded effort ({_n_label}). Pace used for VDOT is the median of your top-{_n} efforts.",
                )
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

    # Weekly TRIMP load from daily aggregates
    _daily = data.get("daily_all", pd.DataFrame())
    _wlog_trimp = {}
    if len(_daily) > 0 and "date_ts" in _daily.columns and "daily_load_hr" in _daily.columns:
        _daily_copy = _daily.copy()
        _daily_copy["week_start"] = _daily_copy["date_ts"].dt.to_period("W-MON").apply(lambda p: p.start_time)
        _weekly_trimp = _daily_copy.groupby("week_start")["daily_load_hr"].sum().reset_index()
        _weekly_trimp.columns = ["week_start", "trimp_sum"]
        _wlog_trimp = dict(zip(_weekly_trimp["week_start"], _weekly_trimp["trimp_sum"]))

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
    _wlog_disp["Load (TRIMP)"] = _wlog_grp["week_start"].apply(
        lambda w: f"{_wlog_trimp.get(w, 0):.0f}" if _wlog_trimp.get(w, 0) > 0 else "\u2014"
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
    # S6: if no streams loaded and VDOT couldn't be calculated, explain why
    if vo2max_needs_streams:
        st.subheader("Estimated VO\u2082max")
        st.info(
            "\U0001f4f6 **GPS stream data required for pace-based VDOT.** "
            "Stream data ensures warmup and cooldown kilometres are excluded from your pace estimate — "
            "without it, average pace across the whole activity would understate your true effort. "
            "Streams are loaded automatically when your Strava activities include GPS. "
            "Try refreshing data from the sidebar, or ensure your runs are recorded with a GPS watch."
        )

    if vo2max_est is not None:
        st.subheader("Estimated VO\u2082max")
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            # Age/sex-adjusted ACSM VO2max classification
            v_class, v_color, _thresholds = _acsm_vo2max_classify(vo2max_est, int(age), str(gender))
            _range_line = ""
            if vo2max_low is not None and vo2max_high is not None and vo2max_low < vo2max_high:
                _range_line = (
                    f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:2px'>"
                    f"Range across distances: {vo2max_low:.0f}\u2013{vo2max_high:.0f}"
                    f"</div>"
                )
            # m2: actionable suggestion per classification level
            _vo2_action = {
                "Poor":             "Focus on easy aerobic volume — 4\u20135 runs/week at conversational pace for 8\u201312 weeks.",
                "Fair":             "Add one quality session/week (tempo or intervals) alongside consistent easy mileage.",
                "Good":             "Maintain consistency and add a dedicated long run each week to build your aerobic ceiling.",
                "Excellent":        "Refine race-specific work and ensure adequate recovery to express this fitness on race day.",
                "Superior / Elite": "Training is highly developed — focus on periodisation and race-specific sharpening.",
            }.get(v_class, "")
            _action_html = (
                f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.55);margin-top:6px'>"
                f"\u2192 {_vo2_action}</div>"
                if _vo2_action else ""
            )
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:{v_color}'>{vo2max_est:.1f}</div>"
                f"<div style='font-size:1rem;color:{v_color}'>{v_class}</div>"
                f"{_range_line}"
                f"{_action_html}"
                f"<div style='font-size:0.8rem;color:grey;margin-top:4px'>ml \u00b7 kg\u207b\u00b9 \u00b7 min\u207b\u00b9 (VDOT estimate)</div>",
                unsafe_allow_html=True,
            )
        with v_col2:
            # ── Build effort metadata strings ─────────────────────────────
            _src_date_str = ""
            _effort_dt = None
            _days_ago = None
            if vo2max_effort_date is not None:
                try:
                    _effort_dt = pd.to_datetime(vo2max_effort_date)
                    _days_ago = (pd.Timestamp.now() - _effort_dt.tz_localize(None)).days
                    _src_date_str = f"{_effort_dt.strftime('%d %b %Y')}, {_days_ago}d ago"
                except Exception:
                    pass

            # ── Format PB time (total time for the canonical distance) ────
            _pb_time_str = ""
            if vo2max_effort_pace is not None and vo2max_effort_dist_km is not None:
                try:
                    _total_min = float(vo2max_effort_pace) * float(vo2max_effort_dist_km)
                    _h = int(_total_min // 60)
                    _m = int(_total_min % 60)
                    _s = int(round((_total_min - int(_total_min)) * 60))
                    _pb_time_str = (
                        f"{_h}:{_m:02d}:{_s:02d}" if _h > 0 else f"{_m}:{_s:02d}"
                    )
                except Exception:
                    pass

            # ── PB summary line ───────────────────────────────────────────
            _src_label = vo2max_source or "best recorded effort"
            # Label: "fastest of N" vs "single effort"
            _effort_label = (
                f"fastest of {vo2max_effort_n} {_src_label} efforts"
                if vo2max_effort_n > 1
                else f"single {_src_label} effort"
            )
            _pb_parts = [
                f"**{_pb_time_str}** ({_effort_label})" if _pb_time_str
                else f"**{_effort_label}**"
            ]
            if _src_date_str:
                _pb_parts.append(_src_date_str)
            if vo2max_effort_source == "stream":
                _pb_parts.append("🔍 detected inside a longer run")
            st.markdown("  ·  ".join(_pb_parts))

            # ── Staleness warning ─────────────────────────────────────────
            if not vo2max_is_recent:
                _stale_msg = (
                    f"⚠️ No qualifying effort in the last 90 days — using an effort from "
                    f"**{_days_ago} days ago**. Run a recent time-trial or race for a current estimate."
                    if _days_ago else
                    f"⚠️ No qualifying effort in the last 90 days — VDOT is based on an older {_src_label} best."
                )
                st.warning(_stale_msg)

            # ── ACSM norm label ───────────────────────────────────────────
            _poor_hi, _fair_hi, _good_hi, _exc_hi = _thresholds
            _g_lower = str(gender).lower()
            if _g_lower.startswith("m"):
                _norm_label = f"Male, age {age}"
                _classification_note = f"Classification uses **ACSM norms for males aged {age}**"
            elif _g_lower.startswith("w"):
                _norm_label = f"Female, age {age}"
                _classification_note = f"Classification uses **ACSM norms for females aged {age}**"
            elif _g_lower.startswith("n"):
                _norm_label = f"Pooled (averaged), age {age}"
                _classification_note = f"Classification uses **pooled ACSM norms (male/female average) aged {age}**"
            else:
                _norm_label = f"Pooled (averaged), age {age}"
                _classification_note = f"Classification uses **pooled ACSM norms (male/female average) aged {age}**"

            # ── Collapsible detail ────────────────────────────────────────
            _range_caption = ""
            if vo2max_low is not None and vo2max_high is not None and vo2max_low < vo2max_high:
                _range_caption = (
                    f" The range **{vo2max_low:.0f}–{vo2max_high:.0f}** reflects estimates across your best efforts "
                    "at different distances — spread indicates course difficulty, pacing, or conditions on those days. "
                    "A tighter range means more consistent fitness expression."
                )

            with st.expander("See more"):
                # S3: build a readable list of all effort dates used in the median
                _effort_dates_str = ""
                if vo2max_effort_dates:
                    _fmt_dates = []
                    for _ed in vo2max_effort_dates:
                        try:
                            _fmt_dates.append(pd.to_datetime(_ed).strftime("%d %b %Y"))
                        except Exception:
                            pass
                    if _fmt_dates:
                        _effort_dates_str = f" Runs used: {', '.join(_fmt_dates)}."

                _method_note = (
                    f"VDOT is the **median of your {vo2max_effort_n} fastest {_src_label} efforts** "
                    f"(fastest: {_pb_time_str})" if vo2max_effort_n > 1 and _pb_time_str
                    else f"VDOT is derived from your **single {_src_label} effort** ({_pb_time_str})" if _pb_time_str
                    else f"VDOT is derived from your **{_src_label}**"
                )
                st.caption(
                    _method_note
                    + (f", recorded {_src_date_str}" if _src_date_str else "")
                    + "." + _effort_dates_str + " Using Jack Daniels' formula. "
                    "The median of multiple efforts is more reliable than a single best — it filters out "
                    "GPS drift, favourable conditions, or a one-off peak that isn't reproducible. "
                    "It reflects race-pace fitness \u2014 not a lab VO\u2082max. A 1-point increase typically means "
                    f"~1\u20132% faster race times.{_range_caption} "
                    f"{_classification_note} — brackets adjust automatically "
                    "as you update your age and sex in the sidebar. "
                    "Improve VO\u2082max by adding easy aerobic volume and one quality session per week."
                )

    # ── Submaximal HR-based VO2max card ───────────────────────────────────
    _sm = vo2max_submax or {}
    _sm_val  = _sm.get("vo2max")
    _sm_low  = _sm.get("vo2max_low")
    _sm_high = _sm.get("vo2max_high")
    _sm_n    = _sm.get("n_runs", 0)

    if _sm_val is not None:
        _sm_class, _sm_color, _ = _acsm_vo2max_classify(_sm_val, int(age), str(gender))
        _sm_col1, _sm_col2 = st.columns([1, 2])
        with _sm_col1:
            _sm_range_line = ""
            if _sm_low is not None and _sm_high is not None and _sm_low < _sm_high:
                _sm_range_line = (
                    f"<div style='font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:2px'>"
                    f"IQR: {_sm_low:.0f}\u2013{_sm_high:.0f}"
                    f"</div>"
                )
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:{_sm_color}'>{_sm_val:.1f}</div>"
                f"<div style='font-size:1rem;color:{_sm_color}'>{_sm_class}</div>"
                f"{_sm_range_line}"
                f"<div style='font-size:0.8rem;color:grey;margin-top:4px'>"
                f"ml \u00b7 kg\u207b\u00b9 \u00b7 min\u207b\u00b9 (aerobic HR estimate)</div>",
                unsafe_allow_html=True,
            )
        with _sm_col2:
            st.markdown(f"**Aerobic HR-based VO\u2082max** · from {_sm_n} training run{'s' if _sm_n != 1 else ''} (last 90 days)")
            with st.expander("See more"):
                st.caption(
                    "**Method:** For each steady-state aerobic run (HR reserve 40–90%, ≥ 20 min), "
                    "the ACSM running equation estimates your oxygen cost at that pace "
                    "(VO\u2082 = 0.2 \u00d7 speed + 3.5 ml/kg/min), then divides by your HR reserve fraction "
                    "to back-calculate VO\u2082max. The **median** across all qualifying runs is reported — "
                    "more robust than a single race effort and works entirely from training data. "
                    "The IQR (interquartile range) reflects day-to-day variability in HR and pacing. "
                    "Sources: ACSM Guidelines 11th ed.; Swain et al. (1994) Med Sci Sports Exerc."
                )
    elif _sm_n == 0:
        st.caption(
            "\U0001f4ac *HR-based VO\u2082max requires runs with heart rate data ≥ 20 min in the last 90 days. "
            "Ensure your HR monitor is paired and runs are syncing with HR.*"
        )

    # ── Comparison note (above ACSM table) ───────────────────────────────
    if vo2max_est is not None and _sm_val is not None:
        _diff = _sm_val - vo2max_est
        if abs(_diff) <= 3:
            st.success(
                f"✓ Both methods agree within {abs(_diff):.1f} points — high confidence in your aerobic fitness estimate."
            )
        elif _diff > 3:
            st.info(
                f"The HR method is **{_diff:.1f} points higher** than your pace-based VDOT. "
                "This often means your training runs are sub-maximal — you're fitter aerobically "
                "than your recent race times suggest. Typical during a training block."
            )
        else:
            st.info(
                f"The HR method is **{abs(_diff):.1f} points lower** than your pace-based VDOT. "
                "This can indicate elevated HR from fatigue, heat, or a fast course skewing the VDOT upward."
            )

    # ── Shared ACSM reference table (shown once, below both methods) ─────
    if vo2max_est is not None:
        ref = pd.DataFrame({
            "Level": ["Poor", "Fair", "Good", "Excellent", "Superior / Elite"],
            "VO\u2082max (ml/kg/min)": [
                f"\u2264 {_poor_hi}",
                f"{_poor_hi + 1}\u2013{_fair_hi}",
                f"{_fair_hi + 1}\u2013{_good_hi}",
                f"{_good_hi + 1}\u2013{_exc_hi}",
                f"> {_exc_hi}",
            ],
        })
        st.dataframe(ref, hide_index=True, use_container_width=True)
        st.caption(f"ACSM norms \u2014 {_norm_label}")
        st.info(
            "\U0001f3d4\ufe0f **Altitude note:** If you train above ~1,500 m / 5,000 ft, "
            "HR is elevated 5\u201315 bpm at equivalent effort, which will inflate both estimates. "
            "All HR-based metrics assume sea-level conditions. "
            "Race at altitude? Expect ~1% performance decline per 300 m above 1,500 m."
        )

    # Divider
    if vo2max_est is not None or _sm_val is not None:
        st.divider()

    # Jack Daniels training paces — pace-based VDOT only (HR estimate intentionally excluded)
    # Daniels' zones were calibrated from race performances; using an HR-derived number
    # would prescribe paces faster than the athlete can sustain in a race.
    st.subheader("Jack Daniels training paces")
    if vo2max_est is not None:
        _src_label = vo2max_source or "best recorded effort"
        st.caption(
            f"Prescribed training paces calculated from your pace-based VDOT of **{vo2max_est:.1f}** "
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
    else:
        st.info(
            "\U0001f3c3 Training paces require a pace-based VDOT estimate. "
            "Run a recent 5K\u2013marathon effort (or time trial) and refresh — "
            "paces are derived from race performance, not the HR-based estimate, "
            "to ensure they match what you can actually sustain on race day."
        )
