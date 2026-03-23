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
    _to_display_dist,
    acwr_band,
    compute_hr_zones,
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
              help="load = duration_min \u00d7 (avgHR / maxHR). Only runs with HR data contribute.")
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
        "Race-ready zone: TSB between +5 and +25."
    )

    # Limit PMC to last 90 days for readability; EWMA was computed on full history
    _pmc_cutoff = daily.sort_values("date_ts")["date_ts"].max() - pd.Timedelta(days=90)
    pmc_daily = daily[daily["date_ts"] >= _pmc_cutoff].copy()

    pmc_fig = go.Figure()

    # TSB form-zone bands
    pmc_fig.add_hrect(y0=5, y1=25, fillcolor="rgba(82,232,138,0.08)", line_width=0,
                      annotation_text="\U0001f3c1 Race-ready", annotation_position="top left",
                      annotation=dict(font_size=11, font_color="rgba(82,232,138,0.7)"),
                      yref="y2")
    pmc_fig.add_hrect(y0=-30, y1=5, fillcolor="rgba(253,141,60,0.05)", line_width=0,
                      annotation_text="\u26a1 Productive training", annotation_position="top left",
                      annotation=dict(font_size=11, font_color="rgba(253,141,60,0.6)"),
                      yref="y2")
    pmc_fig.add_hrect(y0=-100, y1=-30, fillcolor="rgba(214,39,40,0.07)", line_width=0,
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
        name="Fatigue / ATL (7d)", line=dict(color="#fd8d3c", width=2, dash="dot"),
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
    st.plotly_chart(pmc_fig, use_container_width=True)

    # TSB snapshot
    latest_tsb = float(daily.sort_values("date_ts").iloc[-1]["tsb"]) if "tsb" in daily.columns else 0.0
    if latest_tsb > 20:
        st.success(f"Current form (TSB): **+{latest_tsb:.0f}** \u2014 You're fresh. Good window to race or do a breakthrough session.")
    elif latest_tsb > 5:
        st.success(f"Current form (TSB): **+{latest_tsb:.0f}** \u2014 Good form. Training will be well absorbed.")
    elif latest_tsb > -10:
        st.info(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Productive training zone. Slightly fatigued but adapting.")
    elif latest_tsb > -30:
        st.warning(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Carrying fatigue. Monitor recovery closely.")
    else:
        st.error(f"Current form (TSB): **{latest_tsb:.0f}** \u2014 Heavy fatigue accumulated. Consider a recovery block.")

    st.divider()

    # ── Training polarization ─────────────────────────────────────
    st.subheader("Training balance")
    st.caption(
        "Research-backed endurance training targets ~80% easy/long volume and ~20% quality (tempo + workout). "
        "This is the '80/20 polarized' model used by most elite coaches."
    )

    if "run_type" in df_range.columns:
        type_km = df_range.groupby("run_type")["distance_km"].sum().reindex(
            list(RUN_TYPE_COLORS.keys()), fill_value=0
        )
        total_km_typed = type_km.sum()

        pol_c1, pol_c2 = st.columns([1, 2])
        with pol_c1:
            easy_km  = type_km.get("Easy", 0) + type_km.get("Long Run", 0) + type_km.get("General", 0)
            hard_km  = type_km.get("Tempo", 0) + type_km.get("Workout", 0) + type_km.get("Race", 0)
            easy_pct = easy_km / total_km_typed * 100 if total_km_typed > 0 else 0
            hard_pct = hard_km / total_km_typed * 100 if total_km_typed > 0 else 0
            st.metric("Easy / Long Run", f"{easy_pct:.0f}%", help="Easy + General + Long Run volume")
            st.metric("Quality (Tempo + Workout)", f"{hard_pct:.0f}%", help="Tempo + Workout + Race volume")
            if easy_pct >= 75:
                st.success("Good polarization \u2014 mostly aerobic base building.")
            elif easy_pct >= 55:
                st.warning("Slightly high proportion of quality work. Consider adding more easy runs.")
            else:
                st.error("Very high quality load. Reduce intensity to avoid burnout.")

        with pol_c2:
            _type_disp = type_km * (KM_TO_MILES if use_miles else 1.0)
            _dist_unit = "mi" if use_miles else "km"
            fig_pol = go.Figure(go.Bar(
                x=_type_disp.index, y=_type_disp.values,
                marker_color=[RUN_TYPE_COLORS[t] for t in _type_disp.index],
                text=[f"{v:.1f} {_dist_unit}" for v in _type_disp.values],
                textposition="outside",
            ))
            fig_pol.update_layout(
                height=300, title="Distance by run type (selected range)",
                xaxis_title="", yaxis_title="mi" if use_miles else "km",
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
            "Z1 Recovery":  "Very easy \u2014 active recovery, cool-downs",
            "Z2 Aerobic":   "Comfortable \u2014 conversational pace, fat-burning base",
            "Z3 Tempo":     "Moderate \u2014 marathon to half-marathon effort",
            "Z4 Threshold": "Hard \u2014 10K race effort, lactate threshold",
            "Z5 VO\u2082max":   "Maximum \u2014 short intervals, 5K race effort",
        }
        zone_df["Description"] = zone_df["Zone"].map(zone_descriptions).fillna("")
        zone_df["Hours"] = (zone_df["Minutes"] / 60).round(1)
        zone_df["Pct"] = (zone_df["Minutes"] / zone_df["Minutes"].sum() * 100).round(1)

        _zone_feelings = {
            "Z1 Recovery": "Recovery", "Z2 Aerobic": "Easy / Base",
            "Z3 Tempo": "Comfortably hard", "Z4 Threshold": "Race effort",
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
                        "Z3 Tempo": "Lactate clearance, marathon fitness, 'comfortably hard'",
                        "Z4 Threshold": "Raise lactate threshold, 10K\u2013HM race pace",
                        "Z5 VO\u2082max": "Maximal oxygen uptake, short sharp intervals",
                    }.get(zname, ""),
                })
            st.dataframe(pd.DataFrame(_zone_bpm_rows), hide_index=True, use_container_width=True)

    st.divider()

    # ── Advanced load metrics (collapsed by default) ─────────────
    with st.expander("Advanced metrics (for coaches) \u2014 monotony & strain"):
        st.caption("Monotony = mean daily load \u00f7 std. Strain = weekly load \u00d7 monotony. High monotony with high load = little variation in a big week \u2014 watch recovery.")
        r1, r2 = st.columns(2)
        with r1:
            fig_mono = px.line(
                weekly, x="week_start", y="monotony",
                title="Training monotony",
                labels={"week_start": "Week", "monotony": "Monotony"},
            )
            fig_mono.add_hline(y=2.0, line_dash="dash", line_color="orange",
                               annotation_text="Watch zone (>2)", annotation_position="top left")
            fig_mono.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_mono, use_container_width=True)

        with r2:
            fig_strain = px.line(
                weekly, x="week_start", y="strain",
                title="Training strain (load \u00d7 monotony)",
                labels={"week_start": "Week", "strain": "Strain"},
            )
            fig_strain.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_strain, use_container_width=True)
