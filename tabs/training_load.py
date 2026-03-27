"""
tabs/training_load.py — Training Load tab render function.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    _d_unit,
    _format_pace,
    _to_display_dist,
    acwr_band,
    compute_compromised_runs,
    compute_hr_zones,
    forward_project_pmc,
)
from config import KM_TO_MILES, RUN_TYPE_COLORS


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    max_hr = settings["max_hr"]
    race_km = settings["race_km"]
    _hr_zones = settings["_hr_zones"]

    daily_all = data["daily_all"]
    weekly_all = data["weekly_all"]
    df_range = data["df_range"]
    streams_by_id = data["streams_by_id"]

    daily, weekly = daily_all, weekly_all

    # ── Tab insight banner ───────────────────────────────────────────
    def _banner(msg, level="info"):
        _s = {"success": "rgba(0,200,83,0.10);border-left:4px solid #00c853",
              "warning": "rgba(255,171,0,0.10);border-left:4px solid #ffab00",
              "info":    "rgba(0,148,255,0.10);border-left:4px solid #0094ff"}
        st.markdown(
            f'<div style="background:{_s.get(level,_s["info"])};color:rgba(255,255,255,0.9);'
            f'padding:0.5rem 1rem;border-radius:4px;margin-bottom:0.75rem;">{msg}</div>',
            unsafe_allow_html=True)
    if len(daily) >= 14:
        _d_sorted = daily.sort_values("date_ts")
        _ctl_now = float(_d_sorted.iloc[-1].get("chronic_load", 0) or 0)
        _idx_4w  = max(0, len(_d_sorted) - 29)
        _ctl_4w  = float(_d_sorted.iloc[_idx_4w].get("chronic_load", 0) or 0)
        if _ctl_4w > 1:
            _chg = (_ctl_now - _ctl_4w) / _ctl_4w * 100
            if _chg > 5:
                _banner(f"Aerobic fitness (CTL) has grown <strong>{abs(_chg):.0f}%</strong> over the last 4 weeks — base is building well.", "success")
            elif _chg < -5:
                _banner(f"Aerobic fitness (CTL) has declined <strong>{abs(_chg):.0f}%</strong> over the last 4 weeks — consider increasing volume.", "warning")
            else:
                _banner("Aerobic fitness (CTL) is holding steady over the last 4 weeks.")

    if len(daily) == 0 or len(weekly) == 0:
        st.info("Not enough data in the selected range to compute training load.")
        st.markdown(
            "**To see this tab:** Make sure your date range includes at least **4 weeks** of runs with "
            "heart rate data recorded. Expand your date range in the sidebar or sync more activities."
        )
        return

    # ── KPI row ─────────────────────────────────────────────────
    weekly = weekly.copy()
    weekly["weekly_distance_ratio"] = weekly["weekly_distance_km"] / race_km
    weekly["long_run_ratio"] = weekly["long_run_km"] / race_km

    sorted_weekly = weekly.sort_values("week_start")
    latest_week = sorted_weekly.iloc[-1]
    prev_week   = sorted_weekly.iloc[-2] if len(sorted_weekly) >= 2 else None
    latest_day  = daily.sort_values("date_ts").iloc[-1]
    acwr_label, acwr_emoji = acwr_band(float(latest_day["acwr"]) if pd.notna(latest_day["acwr"]) else np.nan)

    _wk_dist = _to_display_dist(latest_week["weekly_distance_km"], use_miles)
    _prev_dist = _to_display_dist(prev_week["weekly_distance_km"], use_miles) if prev_week is not None else None
    dist_delta = (
        f"{_wk_dist - _prev_dist:+.1f} {_d_unit(use_miles)} vs last wk"
        if _prev_dist is not None else None
    )
    load_delta = (
        f"{latest_week['weekly_load_hr'] - prev_week['weekly_load_hr']:+.1f} vs last wk"
        if prev_week is not None else None
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Weekly distance", f"{_wk_dist:.1f} {_d_unit(use_miles)}", dist_delta)
    k2.metric("Weekly HR load", f"{latest_week['weekly_load_hr']:.1f}", load_delta,
              help="Banister TRIMP: duration \u00d7 HRr \u00d7 exp(coefficient \u00d7 HRr), where HRr = (avgHR \u2212 restHR) \u00f7 (maxHR \u2212 restHR). Only runs with HR data contribute.")
    k3.metric("ACWR", f"{latest_day['acwr']:.2f}" if pd.notna(latest_day["acwr"]) else "N/A",
              f"{acwr_emoji} {acwr_label}",
              help="Acute:Chronic Workload Ratio. 0.8\u20131.3 is the balanced zone.")
    k4.metric("Long run % of race", f"{latest_week['long_run_ratio']*100:.0f}%")

    st.caption("HR load requires average HR data on each run. Distance and duration charts use all runs.")

    st.divider()

    # ── Performance Management Chart (PMC) ───────────────────────
    st.subheader("Performance Management Chart")
    st.caption(
        "**Fitness** (CTL, 28d EWMA) builds slowly. **Fatigue** (ATL, 7d EWMA) spikes fast. "
        "**Form / TSB** = Fitness \u2212 Fatigue. Positive TSB = fresh; negative = fatigued. "
        "Race-ready zone: TSB between +10 and +50 (full Banister TRIMP units). "
        "\u26a0\ufe0f The 28-day CTL and 7-day ATL time constants were originally derived from elite "
        "athlete data (Banister, 1991). At lower training volumes (< 60 km/week), CTL may "
        "respond more slowly than expected \u2014 use trends and direction rather than absolute values."
    )

    # Limit PMC to last 90 days for readability; EWMA was computed on full history
    _pmc_cutoff = daily.sort_values("date_ts")["date_ts"].max() - pd.Timedelta(days=90)
    pmc_daily = daily[daily["date_ts"] >= _pmc_cutoff].copy()

    pmc_fig = go.Figure()

    # TSB form-zone bands
    pmc_fig.add_hrect(y0=10, y1=50, fillcolor="rgba(82,232,138,0.08)", line_width=0,
                      annotation_text="\U0001f3c1 Race-ready", annotation_position="top left",
                      annotation=dict(font_size=11, font_color="rgba(82,232,138,0.7)"),
                      yref="y2")
    pmc_fig.add_hrect(y0=-75, y1=10, fillcolor="rgba(253,141,60,0.05)", line_width=0,
                      annotation_text="\u26a1 Productive training", annotation_position="top left",
                      annotation=dict(font_size=11, font_color="rgba(253,141,60,0.6)"),
                      yref="y2")
    pmc_fig.add_hrect(y0=-300, y1=-75, fillcolor="rgba(214,39,40,0.07)", line_width=0,
                      annotation_text="\U0001f623 High fatigue \u2014 recover", annotation_position="top left",
                      annotation=dict(font_size=11, font_color="rgba(214,39,40,0.6)"),
                      yref="y2")
    pmc_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1,
                      yref="y2")

    # CTL, ATL on left axis
    pmc_fig.add_trace(go.Scatter(
        x=pmc_daily["date_ts"], y=pmc_daily["chronic_load"], mode="lines",
        name="Fitness / CTL (28d)", line=dict(color="#3dba6e", width=2.5),
    ))
    pmc_fig.add_trace(go.Scatter(
        x=pmc_daily["date_ts"], y=pmc_daily["acute_load"], mode="lines",
        name="Fatigue / ATL (7d)", line=dict(color="#fd8d3c", width=2),
    ))

    # TSB on right axis
    pmc_fig.add_trace(go.Scatter(
        x=pmc_daily["date_ts"], y=pmc_daily["tsb"], mode="lines",
        name="Form / TSB", line=dict(color="#6baed6", width=2),
        yaxis="y2", fill="tozeroy", fillcolor="rgba(107,174,214,0.08)",
    ))

    pmc_fig.update_layout(
        height=400,
        margin=dict(l=10, r=60, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Date",
        yaxis=dict(title="Fitness / Fatigue (a.u.)", side="left"),
        yaxis2=dict(title="Form / TSB", overlaying="y", side="right", showgrid=False,
                    zeroline=False),
        hovermode="x unified",
    )
    # "You are here" marker on current TSB
    _today_row = pmc_daily.sort_values("date_ts").iloc[-1]
    _today_tsb_val = float(_today_row["tsb"]) if pd.notna(_today_row["tsb"]) else 0.0
    pmc_fig.add_annotation(
        x=_today_row["date_ts"], y=_today_tsb_val,
        text=f"Today<br>TSB {_today_tsb_val:+.0f}",
        showarrow=True, arrowhead=2, arrowcolor="#6baed6",
        font=dict(size=10, color="#6baed6"),
        bgcolor="rgba(20,20,30,0.7)", bordercolor="#6baed6", borderwidth=1,
        ax=40, ay=-40, yref="y2",
    )
    # HR coverage warning
    _hr_cov = int(data["df_range"]["avg_hr"].notna().sum()) if len(data.get("df_range", [])) > 0 else 0
    _total_runs = len(data.get("df_range", []))
    if _total_runs > 0:
        _hr_pct = _hr_cov / _total_runs * 100
        if _hr_pct < 70:
            st.warning(
                f"\u26a0\ufe0f Only **{_hr_pct:.0f}%** of runs in the selected range have HR data "
                f"({_hr_cov}/{_total_runs} runs). CTL and ATL may be underestimated \u2014 "
                "ensure your watch records HR on every run."
            )
    st.plotly_chart(pmc_fig, use_container_width=True)

    # TSB snapshot
    latest_tsb = float(daily.sort_values("date_ts").iloc[-1]["tsb"]) if "tsb" in daily.columns else 0.0
    if latest_tsb > 50:
        st.success(f"Current form (TSB): **+{latest_tsb:.0f}** \u2014 You're fresh. Good window to race or do a breakthrough session.")
    elif latest_tsb > 10:
        st.success(f"Current form (TSB): **+{latest_tsb:.0f}** \u2014 Good form. Training will be well absorbed.")
    elif latest_tsb > -25:
        st.info(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Productive training zone. Slightly fatigued but adapting.")
    elif latest_tsb > -75:
        st.warning(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Carrying fatigue. Monitor recovery closely.")
    else:
        st.error(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Heavy fatigue accumulated. Consider a recovery block.")

    st.caption(
        "TSB thresholds are reference values calibrated to population-level TRIMP averages. "
        "Your personal optimal race-day TSB may differ \u2014 use trends over time, not absolute numbers."
    )

    st.divider()

    # ── Race Day Planner ─────────────────────────────────────────
    st.subheader("Race day planner")
    st.caption(
        "Projects your fitness (CTL), fatigue (ATL), and form (TSB) forward to a target race date "
        "under assumed training + taper. This is a *planning tool* — it shows what your PMC will look "
        "like if you follow the plan, not a performance guarantee."
    )

    # Auto-default taper weeks based on race distance
    _taper_default = 1
    if race_km >= 38:
        _taper_default = 3
    elif race_km >= 18:
        _taper_default = 2

    # Load saved PMC planner prefs from session state
    import datetime as _dt
    _prefs = st.session_state.get("_prefs", {})
    _saved_race_date_str = _prefs.get("pmc_race_date")
    try:
        _default_race_date = _dt.date.fromisoformat(_saved_race_date_str) if _saved_race_date_str else _dt.date.today() + _dt.timedelta(weeks=12)
        if _default_race_date <= _dt.date.today():
            _default_race_date = _dt.date.today() + _dt.timedelta(weeks=12)
    except (ValueError, TypeError):
        _default_race_date = _dt.date.today() + _dt.timedelta(weeks=12)
    _saved_taper = int(_prefs.get("pmc_taper_weeks", _taper_default))
    _saved_taper = max(1, min(3, _saved_taper))
    _build_options = ["Maintain current load", "Build +5%/week", "Build +10%/week"]
    _saved_build = _prefs.get("pmc_build_choice", "Maintain current load")
    _saved_build_idx = _build_options.index(_saved_build) if _saved_build in _build_options else 0

    # Current 28d avg daily load as default pre-taper load
    _daily_sorted = daily.sort_values("date_ts")
    _recent_28 = _daily_sorted.tail(28)
    _avg_daily_load = float(_recent_28["daily_load_hr"].mean()) if "daily_load_hr" in _recent_28.columns and len(_recent_28) > 0 else 30.0
    _avg_daily_load = max(1.0, _avg_daily_load)

    _pc1, _pc2, _pc3 = st.columns(3)
    with _pc1:
        _race_date_input = st.date_input(
            "Target race date",
            value=_default_race_date,
            min_value=(_dt.date.today() + _dt.timedelta(days=1)),
            help="The date you want to peak for.",
            key="pmc_race_date",
        )
    with _pc2:
        _taper_weeks = st.slider(
            "Taper length (weeks)",
            min_value=1, max_value=3, value=_saved_taper,
            help=f"Weeks of reduced volume before race day. Auto-set to {_taper_default} weeks based on your target distance. "
                 "Science: Bosquet et al. (2007) meta-analysis recommends 8–14 days of taper with 40–60% volume reduction.",
            key="pmc_taper_weeks",
        )
    with _pc3:
        _build_choice = st.radio(
            "Pre-taper training",
            _build_options,
            index=_saved_build_idx,
            help="How your training load changes between now and the taper. Build options compound weekly — capped at 1.5× current load.",
            key="pmc_build_choice",
        )
    _build_pct = {"Maintain current load": 0.0, "Build +5%/week": 5.0, "Build +10%/week": 10.0}[_build_choice]

    _proj = forward_project_pmc(
        daily,
        race_date=pd.Timestamp(_race_date_input),
        taper_weeks=int(_taper_weeks),
        pre_taper_daily_load=_avg_daily_load,
        load_build_pct=_build_pct,
    )

    if len(_proj) == 0:
        st.info("Select a future race date to see the projection.")
    else:
        # Combine historical + projected for chart
        _hist_plot = pmc_daily[["date_ts", "chronic_load", "acute_load", "tsb"]].copy()
        _hist_plot = _hist_plot.rename(columns={"date_ts": "date", "chronic_load": "ctl", "acute_load": "atl"})

        _proj_fig = go.Figure()

        # Historical solid lines
        _proj_fig.add_trace(go.Scatter(
            x=_hist_plot["date"], y=_hist_plot["ctl"], mode="lines",
            name="Fitness / CTL (actual)", line=dict(color="#3dba6e", width=2),
        ))
        _proj_fig.add_trace(go.Scatter(
            x=_hist_plot["date"], y=_hist_plot["atl"], mode="lines",
            name="Fatigue / ATL (actual)", line=dict(color="#fd8d3c", width=2),
        ))
        _proj_fig.add_trace(go.Scatter(
            x=_hist_plot["date"], y=_hist_plot["tsb"], mode="lines",
            name="Form / TSB (actual)", line=dict(color="#6baed6", width=2),
            yaxis="y2",
        ))

        # Projected dashed lines
        _proj_fig.add_trace(go.Scatter(
            x=_proj["date"], y=_proj["ctl"], mode="lines",
            name="Fitness / CTL (projected)", line=dict(color="#3dba6e", width=2, dash="dash"),
        ))
        _proj_fig.add_trace(go.Scatter(
            x=_proj["date"], y=_proj["atl"], mode="lines",
            name="Fatigue / ATL (projected)", line=dict(color="#fd8d3c", width=2, dash="dash"),
        ))
        _proj_fig.add_trace(go.Scatter(
            x=_proj["date"], y=_proj["tsb"], mode="lines",
            name="Form / TSB (projected)", line=dict(color="#6baed6", width=2, dash="dash"),
            yaxis="y2",
        ))

        # Uncertainty bands on projected CTL, ATL, TSB
        # Cap std at 15% of average daily load to keep bands readable (real std can be
        # inflated by rest days with 0 load skewing variance on short 28d windows).
        _load_std_raw = float(_recent_28["daily_load_hr"].std()) if "daily_load_hr" in _recent_28.columns and len(_recent_28) > 3 else 5.0
        _load_std = max(1.0, min(_load_std_raw, _avg_daily_load * 0.15))

        # CTL uncertainty band (CTL = slow EMA so variance is smaller: ÷3)
        _ctl_std = _load_std / 3.0
        _ctl_upper = _proj["ctl"] + _ctl_std
        _ctl_lower = _proj["ctl"] - _ctl_std
        _proj_fig.add_trace(go.Scatter(
            x=pd.concat([_proj["date"], _proj["date"].iloc[::-1]]),
            y=pd.concat([_ctl_upper, _ctl_lower.iloc[::-1]]),
            fill="toself", fillcolor="rgba(61,186,110,0.10)", line_width=0,
            name="CTL uncertainty band", showlegend=True, hoverinfo="skip",
        ))

        # ATL uncertainty band (ATL = fast EMA so variance is larger: ÷1.5)
        _atl_std = _load_std / 1.5
        _atl_upper = _proj["atl"] + _atl_std
        _atl_lower = _proj["atl"] - _atl_std
        _proj_fig.add_trace(go.Scatter(
            x=pd.concat([_proj["date"], _proj["date"].iloc[::-1]]),
            y=pd.concat([_atl_upper, _atl_lower.iloc[::-1]]),
            fill="toself", fillcolor="rgba(253,141,60,0.10)", line_width=0,
            name="ATL uncertainty band", showlegend=True, hoverinfo="skip",
        ))

        # TSB uncertainty band (propagated from CTL + ATL)
        _tsb_upper = _proj["tsb"] + _load_std * 1.0
        _tsb_lower = _proj["tsb"] - _load_std * 1.0
        _proj_fig.add_trace(go.Scatter(
            x=pd.concat([_proj["date"], _proj["date"].iloc[::-1]]),
            y=pd.concat([_tsb_upper, _tsb_lower.iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(107,174,214,0.10)",
            line_width=0,
            name="TSB uncertainty band",
            showlegend=True,
            hoverinfo="skip",
            yaxis="y2",
        ))

        # Race date vertical line
        _proj_fig.add_vline(
            x=pd.Timestamp(_race_date_input).timestamp() * 1000,
            line_dash="dash", line_color="rgba(255,255,255,0.5)",
            annotation_text="\U0001f3c1 Race day", annotation_position="top right",
            annotation_font_color="rgba(255,255,255,0.7)",
        )

        # Taper start shading
        _taper_start_ts = pd.Timestamp(_race_date_input) - pd.Timedelta(weeks=int(_taper_weeks))
        _proj_fig.add_vrect(
            x0=_taper_start_ts, x1=pd.Timestamp(_race_date_input),
            fillcolor="rgba(107,174,214,0.07)", line_width=0,
            annotation_text="Taper", annotation_position="top left",
            annotation_font_color="rgba(107,174,214,0.6)",
        )

        # Race-ready TSB band on y2
        _proj_fig.add_hrect(y0=10, y1=50, fillcolor="rgba(82,232,138,0.06)", line_width=0,
                            annotation_text="\U0001f3c1 Race-ready TSB", annotation_position="top left",
                            annotation=dict(font_size=10, font_color="rgba(82,232,138,0.6)"),
                            yref="y2")

        _proj_fig.update_layout(
            height=420,
            margin=dict(l=10, r=60, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Date",
            yaxis=dict(title="Fitness / Fatigue", side="left"),
            yaxis2=dict(title="Form / TSB", overlaying="y", side="right", showgrid=False, zeroline=False),
            hovermode="x unified",
        )
        st.plotly_chart(_proj_fig, use_container_width=True)

        # Race day summary
        _race_row = _proj.iloc[-1]
        _race_tsb = float(_race_row["tsb"])
        _race_ctl = float(_race_row["ctl"])
        _taper_load_day1 = float(_proj[_proj["date"] >= _taper_start_ts].iloc[0]["daily_load"]) if len(_proj[_proj["date"] >= _taper_start_ts]) > 0 else _avg_daily_load

        _rc1, _rc2, _rc3 = st.columns(3)
        _rc1.metric("Projected CTL on race day", f"{_race_ctl:.1f}", help="Fitness level at race date.")
        _rc2.metric("Projected TSB on race day", f"{_race_tsb:+.0f}",
                    help="Positive = fresh. Race-ready window is +10 to +50.")
        _taper_start_str = _taper_start_ts.strftime("%b %d")
        _rc3.metric("Taper starts", _taper_start_str,
                    help=f"{_taper_weeks}-week taper beginning {_taper_start_str}. Load drops to ~{_taper_load_day1:.0f} TRIMP/day in week 1.")

        if _race_tsb >= 10 and _race_tsb <= 50:
            st.success(f"Projected TSB on race day: **{_race_tsb:+.0f}** — inside the race-ready window. Taper plan looks good.")
        elif _race_tsb > 50:
            st.warning(f"Projected TSB: **{_race_tsb:+.0f}** — very fresh, potentially undertrained. Consider a longer build phase or shorter taper.")
        elif _race_tsb > -25:
            st.warning(f"Projected TSB: **{_race_tsb:+.0f}** — slightly fatigued on race day. Consider adding 1 more taper week or reducing pre-taper load.")
        else:
            st.error(f"Projected TSB: **{_race_tsb:+.0f}** — carrying heavy fatigue into race day. Extend the taper or reduce build load.")

        st.caption(
            f"Assumes **{_build_choice.lower()}** until taper, then ~40%/week volume reduction over {_taper_weeks} week(s). "
            "Pre-taper daily load is your 28-day average. Solid lines = actual data; dashed = projection."
        )

    st.divider()

    # ── Training polarization ─────────────────────────────────────
    st.subheader("Training balance")
    st.caption(
        "Research-backed endurance training targets ~80% easy/long **time** and ~20% quality (tempo + workout). "
        "This is the '80/20 polarized' model (Seiler, 2010). Measured in **minutes** — not distance — "
        "because a tempo run covers more ground per minute than an easy run, making distance a biased metric."
    )

    if "run_type" in df_range.columns:
        # C1 FIX: use duration_min (time-in-zone), not distance_km.
        # Seiler's 80/20 model is defined by TIME, not kilometres.
        type_min = df_range.groupby("run_type")["duration_min"].sum().reindex(
            list(RUN_TYPE_COLORS.keys()), fill_value=0
        )
        total_min_typed = type_min.sum()

        pol_c1, pol_c2 = st.columns([1, 2])
        with pol_c1:
            easy_min = type_min.get("Easy", 0) + type_min.get("Long Run", 0) + type_min.get("General", 0)
            hard_min = type_min.get("Tempo", 0) + type_min.get("Workout", 0) + type_min.get("Race", 0)
            easy_pct = easy_min / total_min_typed * 100 if total_min_typed > 0 else 0
            hard_pct = hard_min / total_min_typed * 100 if total_min_typed > 0 else 0
            st.metric("Easy / Long Run", f"{easy_pct:.0f}%", help="Easy + General + Long Run time (minutes)")
            st.metric("Quality (Tempo + Workout)", f"{hard_pct:.0f}%", help="Tempo + Workout + Race time (minutes)")
            if easy_pct >= 75:
                st.success("Good polarization \u2014 mostly aerobic base building.")
            elif easy_pct >= 55:
                st.warning("Slightly high proportion of quality work. Consider adding more easy runs.")
            else:
                st.error("Very high quality load. Reduce intensity to avoid burnout.")

        with pol_c2:
            fig_pol = go.Figure(go.Bar(
                x=type_min.index, y=type_min.values,
                marker_color=[RUN_TYPE_COLORS[t] for t in type_min.index],
                text=[f"{v:.0f} min" for v in type_min.values],
                textposition="outside",
            ))
            fig_pol.update_layout(
                height=300, title="Time by run type — selected range (minutes)",
                xaxis_title="", yaxis_title="minutes",
                margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
            )
            st.plotly_chart(fig_pol, use_container_width=True)

    st.divider()

    # ── Weekly volume stacked by type ────────────────────────────
    st.subheader("Weekly volume")
    _dist_unit = "mi" if use_miles else "km"
    _dist_factor = KM_TO_MILES if use_miles else 1.0
    if "run_type" in df_range.columns:
        weekly_by_type = (
            df_range.assign(
                week_start=df_range["start_dt_local"].dt.to_period("W").apply(lambda p: p.start_time)
            )
            .groupby(["week_start", "run_type"])["distance_km"]
            .sum()
            .reset_index()
        )
        weekly_by_type["distance_disp"] = weekly_by_type["distance_km"] * _dist_factor
        fig_weekly_dist = px.bar(
            weekly_by_type, x="week_start", y="distance_disp", color="run_type",
            color_discrete_map=RUN_TYPE_COLORS,
            title=f"Weekly distance by run type ({_dist_unit})",
            labels={"week_start": "Week", "distance_disp": _dist_unit, "run_type": "Type"},
            category_orders={"run_type": list(RUN_TYPE_COLORS.keys())},
        )
        fig_weekly_dist.update_layout(
            height=340, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_weekly_dist, use_container_width=True)
    else:
        _weekly_disp = weekly.copy()
        _weekly_disp["distance_disp"] = _weekly_disp["weekly_distance_km"] * _dist_factor
        fig_weekly_dist = px.bar(
            _weekly_disp, x="week_start", y="distance_disp",
            title=f"Weekly distance ({_dist_unit})",
            labels={"week_start": "Week", "distance_disp": _dist_unit},
        )
        fig_weekly_dist.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_weekly_dist, use_container_width=True)

    st.divider()

    # ── HR zone breakdown ────────────────────────────────────────
    st.subheader("Heart rate zone breakdown")
    st.caption(
        "Time spent in each HR zone, calculated from per-second stream data where available. "
        "Elite endurance athletes spend ~80% of their time in Z1\u2013Z2."
    )
    zone_df = compute_hr_zones(df_range, streams_by_id, int(max_hr), hr_zones=_hr_zones)
    zone_df = zone_df[zone_df["Minutes"] > 0]
    if len(zone_df) == 0:
        st.info("No HR data available in the selected range to compute zones.")
    else:
        zone_colors = ["#6baed6", "#74c476", "#fd8d3c", "#e6550d", "#d62728"]
        zone_descriptions = {
            "Z1 Recovery":                    "Very easy \u2014 active recovery, cool-downs",
            "Z2 Aerobic":                     "Comfortable \u2014 conversational pace, fat-burning base",
            "Z3 Marathon / Aerobic Threshold":"Moderate \u2014 marathon to half-marathon effort",
            "Z4 Lactate Threshold / Tempo":   "Hard \u2014 10K race effort, lactate threshold",
            "Z5 VO\u2082max":                  "Maximum \u2014 short intervals, 5K race effort",
        }
        zone_df["Description"] = zone_df["Zone"].map(zone_descriptions).fillna("")
        zone_df["Hours"] = (zone_df["Minutes"] / 60).round(1)
        zone_df["Pct"] = (zone_df["Minutes"] / zone_df["Minutes"].sum() * 100).round(1)

        _zone_feelings = {
            "Z1 Recovery": "Recovery", "Z2 Aerobic": "Easy / Base",
            "Z3 Marathon / Aerobic Threshold": "Comfortably hard", "Z4 Lactate Threshold / Tempo": "Race effort",
            "Z5 VO\u2082max": "Max intensity",
        }
        zone_df["Feeling"] = zone_df["Zone"].map(_zone_feelings).fillna("")
        fig_zones = px.bar(
            zone_df, x="Zone", y="Minutes",
            color="Zone",
            color_discrete_sequence=zone_colors,
            labels={"Minutes": "Time (min)"},
            custom_data=["Description", "Hours", "Pct", "Feeling"],
            text="Feeling",
        )
        fig_zones.update_traces(
            hovertemplate="<b>%{x}</b><br>%{customdata[0]}<br>%{y:.0f} min (%{customdata[2]:.1f}%)<extra></extra>",
            textposition="outside",
            textfont=dict(size=10),
        )
        fig_zones.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
                                uniformtext_minsize=8, uniformtext_mode="hide")
        st.plotly_chart(fig_zones, use_container_width=True)

        total_min = zone_df["Minutes"].sum()
        z1z2_pct = float(zone_df.loc[zone_df["Zone"].isin(["Z1 Recovery", "Z2 Aerobic"]), "Minutes"].sum()) / total_min * 100 if total_min > 0 else 0
        z2_pct = float(zone_df.loc[zone_df["Zone"] == "Z2 Aerobic", "Minutes"].sum()) / total_min * 100 if total_min > 0 else 0

        if z1z2_pct >= 75:
            st.success(f"**{z1z2_pct:.0f}%** of training in Z1\u2013Z2 \u2713 \u2014 well within the 80/20 polarized model. Z2 alone: {z2_pct:.0f}%.")
        elif z1z2_pct >= 60:
            st.warning(f"**{z1z2_pct:.0f}%** in Z1\u2013Z2 \u2014 slightly high intensity. Try adding more easy runs to reach 75\u201380%.")
        else:
            st.error(f"Only **{z1z2_pct:.0f}%** in Z1\u2013Z2 \u2014 training hard. Add easy days to avoid accumulated fatigue.")

        st.caption(
            "\u26a0\ufe0f **Time \u2260 training stress.** A Z4 minute carries roughly 3\u20134\u00d7 the physiological "
            "load of a Z2 minute. This chart is a *polarization check* \u2014 for true workload, "
            "refer to the TRIMP load in the weekly log and PMC chart above."
        )

        # Zone reference table
        with st.expander("Zone reference guide"):
            _zone_bpm_rows = []
            for (zname, lo_frac, hi_frac) in _hr_zones:
                lo_bpm = int(lo_frac * max_hr)
                hi_bpm = int(min(hi_frac, 1.0) * max_hr)
                _zone_bpm_rows.append({
                    "Zone": zname,
                    "BPM range": f"{lo_bpm}\u2013{hi_bpm}" if hi_frac < 2 else f">{lo_bpm}",
                    "Feel": zone_descriptions.get(zname, ""),
                    "Training purpose": {
                        "Z1 Recovery": "Active recovery, blood flow, easy distance",
                        "Z2 Aerobic": "Aerobic base, mitochondrial density, fat adaptation",
                        "Z3 Marathon / Aerobic Threshold": "Lactate clearance, marathon fitness, 'comfortably hard'",
                        "Z4 Lactate Threshold / Tempo": "Raise lactate threshold, 10K\u2013HM race pace",
                        "Z5 VO\u2082max": "Maximal oxygen uptake, short sharp intervals",
                    }.get(zname, ""),
                })
            st.dataframe(pd.DataFrame(_zone_bpm_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Training monotony ─────────────────────────────────────────
    st.subheader("Training monotony")
    st.caption(
        "Monotony = mean \u00f7 std of daily load for the week. Higher values = less day-to-day variation = "
        "doing the same thing every day. Aim to keep monotony **below 2.0** by mixing easy, hard, and rest days. "
        "Originally calibrated for RPE-based load (Foster, 1998) \u2014 treat as a directional indicator."
    )

    # Surface a live monotony warning if the latest week is high
    if "monotony" in weekly.columns:
        _latest_mono = float(weekly.sort_values("week_start").iloc[-1].get("monotony", 0) or 0)
        if _latest_mono >= 2.0:
            st.warning(
                f"\u26a0\ufe0f **High monotony this week ({_latest_mono:.1f})** \u2014 low training variation. "
                "Add an easy/rest day or vary session intensity to reduce injury risk."
            )
        elif _latest_mono > 0:
            st.success(f"\u2713 Monotony this week: **{_latest_mono:.1f}** \u2014 good variation in training stimulus.")

    _mono_cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
    _weekly_90 = weekly[weekly["week_start"] >= _mono_cutoff].copy()

    r1, r2 = st.columns(2)
    with r1:
        fig_mono = px.line(
            _weekly_90, x="week_start", y="monotony",
            title="Training monotony \u2014 last 90 days",
            labels={"week_start": "Week", "monotony": "Monotony"},
        )
        fig_mono.add_hline(y=2.0, line_dash="dash", line_color="orange",
                           annotation_text="High monotony \u2014 low training variation", annotation_position="top left")
        fig_mono.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_mono, use_container_width=True)

    with r2:
        fig_strain = px.line(
            _weekly_90, x="week_start", y="strain",
            title="Training strain \u2014 last 90 days",
            labels={"week_start": "Week", "strain": "Strain"},
        )
        fig_strain.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            text="High strain = big week with little variation \u2014 highest injury risk combination",
            showarrow=False, font=dict(size=10, color="rgba(255,255,255,0.5)"),
            xanchor="left", yanchor="top",
        )
        fig_strain.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_strain, use_container_width=True)

    st.divider()

    # ── Run efficiency detector ───────────────────────────────────────
    st.subheader("Run efficiency detector")
    st.caption(
        "Flags runs where your speed-per-HR was below your rolling 20th-percentile baseline — "
        "a signal of residual fatigue, illness, or heat stress. Baseline is computed from easy "
        "runs only so tempo and race sessions don't inflate the reference."
    )

    comp = compute_compromised_runs(df_range, max_hr=max_hr)
    if len(comp) == 0 or comp["eff_q20"].isna().all():
        st.info("Not enough HR-bearing easy runs to build a baseline yet (need ~6+). Widen the date range or ensure HR is recorded on all runs.")
    else:
        n_flagged = int(comp["compromised"].sum())
        rate = 100.0 * comp["compromised"].mean() if len(comp) > 0 else float("nan")

        _ec1, _ec2 = st.columns(2)
        with _ec1:
            st.metric(
                "Low-efficiency runs",
                f"{n_flagged} / {len(comp)}",
                f"{rate:.0f}% of runs in range",
                help="Runs where speed÷HR fell below your rolling 20th-percentile easy-run baseline.",
            )
            comp["Status"] = comp["compromised"].map({1: "Low efficiency", 0: "Normal"})
            fig_comp = px.scatter(
                comp,
                x="start_dt_local", y="speed_per_hr",
                color="Status",
                color_discrete_map={"Low efficiency": "#d62728", "Normal": "#74c476"},
                title="Aerobic efficiency over time",
                labels={"start_dt_local": "Date", "speed_per_hr": "Speed ÷ HR (m/s per bpm)"},
            )
            fig_comp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_comp, use_container_width=True)

        with _ec2:
            fig_delta = px.histogram(
                comp.dropna(subset=["eff_delta"]),
                x="eff_delta", nbins=25,
                title="Efficiency vs your easy-run baseline",
                labels={"eff_delta": "Speed/HR minus 20th-percentile baseline"},
            )
            fig_delta.add_vline(x=0, line_dash="dash", line_color="red",
                                annotation_text="Baseline", annotation_position="top right")
            fig_delta.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_delta, use_container_width=True)

        with st.expander("Show full efficiency table"):
            _disp_cols = ["start_dt_local", "name", "distance_km", "duration_min",
                          "avg_hr", "pace_min_per_km", "speed_per_hr", "eff_q20", "eff_delta", "compromised"]
            _disp_cols = [c for c in _disp_cols if c in comp.columns]
            _disp = comp[_disp_cols].copy()
            if "pace_min_per_km" in _disp.columns:
                _disp["pace_min_per_km"] = _disp["pace_min_per_km"].apply(lambda x: _format_pace(x, use_miles))
            _disp["compromised"] = _disp["compromised"].map({1: "⚠️ Low", 0: "✓ Normal"})
            _disp = _disp.rename(columns={
                "start_dt_local": "Date", "name": "Run",
                "distance_km": "Dist (km)", "duration_min": "Time (min)",
                "avg_hr": "Avg HR", "pace_min_per_km": "Pace",
                "speed_per_hr": "Speed/HR", "eff_q20": "Baseline (q20)",
                "eff_delta": "Δ baseline", "compromised": "Efficiency",
            })
            st.dataframe(_disp.sort_values("Date", ascending=False), hide_index=True, use_container_width=True)
