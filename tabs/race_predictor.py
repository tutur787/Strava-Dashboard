"""
tabs/race_predictor.py — Race Predictor tab render function.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    _d_unit,
    _format_pace,
    _p_unit,
    compute_efficiency_adjustment,
    compute_risk_table,
    compute_risk_penalty,
    format_hms,
    predict_race_time_riegel,
)
from config import KM_TO_MILES


def render(data: dict, settings: dict) -> None:
    use_miles = settings["use_miles"]
    max_hr = settings["max_hr"]
    race_km = settings["race_km"]
    effort_band = settings["effort_band"]
    prediction_lookback_days = settings["prediction_lookback_days"]

    activities = data["activities"]
    daily_all = data["daily_all"]
    weekly_all = data["weekly_all"]
    bests = data["bests"]
    vo2max_submax = data.get("vo2max_submax", {})

    st.subheader("Race predictor")

    # Determine end_ts from the activities to reconstruct what model_end was
    end_ts = activities["start_dt_local"].max()
    # app.py passes end_ts implicitly via the date range; here we use a conservative proxy
    # (the max date in df_range is effectively end_ts)
    df_range = data["df_range"]
    if len(df_range) > 0:
        end_ts = df_range["start_dt_local"].max()

    model_end = end_ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    model_start = model_end - pd.Timedelta(days=int(prediction_lookback_days))

    model_runs = activities[(activities["start_dt_local"] >= model_start) & (activities["start_dt_local"] <= model_end)].copy()

    st.caption(
        f"Using **{len(model_runs)} runs** from the last {int(prediction_lookback_days)} days. "
        "The Riegel formula selects the run that gives the best equivalent performance at your target distance."
    )

    st.caption(
        "\u26a0\ufe0f **Accuracy note:** Riegel's formula was validated on race performances. "
        "Predictions derived from training runs are indicative only \u2014 for best accuracy, "
        "include timed race efforts (parkruns, time trials, races) in your recent history."
    )

    colA, colB = st.columns([1.2, 1])
    with colA:
        effort_preset = st.selectbox(
            "Prediction mode",
            ["Balanced", "Optimistic", "Conservative"],
            help=(
                "Controls how much performance drops off at longer distances (Riegel fatigue exponent). "
                "**Balanced** = population average (1.06). "
                "**Optimistic** = less drop-off (1.03) — suits well-trained aerobic runners. "
                "**Conservative** = more drop-off (1.09) — suits newer runners or those who typically fade."
            ),
        )
        exp = {"Balanced": 1.06, "Optimistic": 1.03, "Conservative": 1.09}[effort_preset]
    with colB:
        _min_dist_default = 5.0 / KM_TO_MILES if use_miles else 5.0
        _min_dist_max = 20.0 / KM_TO_MILES if use_miles else 20.0
        min_dist_input = st.number_input(
            f"Ignore runs shorter than ({_d_unit(use_miles)})",
            min_value=0.5, max_value=round(_min_dist_max, 1),
            value=round(_min_dist_default, 1), step=0.5,
            help="Short runs (intervals, warm-ups) are excluded from the prediction to reduce noise.",
        )
        min_dist = min_dist_input * KM_TO_MILES if use_miles else min_dist_input

    # C2: race-only checkbox
    _has_race_type = "run_type" in model_runs.columns and (model_runs["run_type"] == "Race").any()
    race_only = st.checkbox(
        "Only use Race-classified runs",
        value=False,
        help="Restricts the Riegel formula to runs tagged as 'Race' (e.g. parkruns, time trials, races). "
             "Riegel was validated on race performances — training runs introduce noise. "
             "Requires at least one Race-classified activity in the lookback window.",
        disabled=not _has_race_type,
    )
    if race_only and _has_race_type:
        _model_runs_used = model_runs[model_runs["run_type"] == "Race"].copy()
    else:
        _model_runs_used = model_runs
    if race_only and not _has_race_type:
        st.caption("⚠️ No Race-classified runs in the lookback window — using all runs.")

    # Baseline: best equivalent performance
    baseline_sec, source = predict_race_time_riegel(
        _model_runs_used, target_km=float(race_km), exponent=float(exp), min_km=float(min_dist),
    )

    if baseline_sec is None or source is None:
        st.info("Not enough runs in this window to predict. Try increasing the lookback window or lowering the minimum distance.")
        return

    # M4: soft warning when the 3× extrapolation cap may be excluding runs
    _max_ratio = 3.0
    _min_source_km = float(race_km) / _max_ratio
    _all_eligible = _model_runs_used[
        pd.notna(_model_runs_used["distance_km"]) &
        (_model_runs_used["distance_km"] >= float(min_dist)) &
        (_model_runs_used["duration_min"] > 3)
    ]
    _cap_excluded = int((_all_eligible["distance_km"] < _min_source_km).sum())
    if _cap_excluded > 0:
        st.caption(
            f"⚠️ **{_cap_excluded} run{'s' if _cap_excluded != 1 else ''} excluded** by the 3× extrapolation cap "
            f"(source run must be ≥ {_min_source_km:.1f} km for a {race_km:.1f} km target). "
            "Riegel's accuracy degrades beyond 3× — include longer runs or a tune-up race for a better prediction."
        )

    eff_factor = compute_efficiency_adjustment(
        model_runs, max_hr=int(max_hr), effort_band=effort_band,
        lookback_days=int(max(30, prediction_lookback_days // 3)), end_ts=model_end,
    )
    daily_risk, weekly_risk = compute_risk_table(daily_all, weekly_all)
    risk_factor = compute_risk_penalty(daily_risk)

    pred_sec = baseline_sec * eff_factor * risk_factor

    low_sec, _  = predict_race_time_riegel(_model_runs_used, float(race_km), exponent=max(1.02, exp - 0.03), min_km=float(min_dist))
    high_sec, _ = predict_race_time_riegel(_model_runs_used, float(race_km), exponent=min(1.12, exp + 0.03), min_km=float(min_dist))
    if low_sec is not None and high_sec is not None:
        pred_lo = low_sec * eff_factor * risk_factor
        pred_hi = high_sec * eff_factor * risk_factor
    else:
        pred_lo, pred_hi = None, None

    pace_sec_per_km = pred_sec / float(race_km)
    pace_min = pace_sec_per_km / 60.0

    # Plain-English efficiency and risk descriptions
    eff_delta_sec_km = (eff_factor - 1.0) * pace_sec_per_km
    _eff_inactive = abs(eff_factor - 1.0) < 0.005
    if _eff_inactive:
        eff_label = "Not enough data"
    elif eff_factor < 1.0:
        eff_label = f"Improving (\u2212{abs(eff_delta_sec_km):.0f}s/km)"
    else:
        eff_label = f"Declining (+{eff_delta_sec_km:.0f}s/km)"

    risk_delta_sec_km = (risk_factor - 1.0) * pace_sec_per_km
    _risk_inactive = risk_factor <= 1.001
    risk_label = "No adjustment" if _risk_inactive else f"High load (+{risk_delta_sec_km:.0f}s/km)"

    # Best actual result at this distance (if any)
    best_actual = bests.get(
        {"5.0": "best_5k", "10.0": "best_10k", "21.0975": "best_hm", "42.195": "best_marathon"}.get(
            f"{race_km:.4f}", ""
        )
    )

    # M2: compute range label immediately so it shows on the headline metric card
    _range_delta = None
    if pred_lo is not None and pred_hi is not None and np.isfinite(pred_lo) and np.isfinite(pred_hi):
        _range_delta = f"{format_hms(pred_lo)} – {format_hms(pred_hi)}"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted finish time", format_hms(pred_sec), _range_delta,
              help="Central estimate using your personal/balanced Riegel exponent. "
                   "Range shown below is the spread between optimistic (−0.03) and conservative (+0.03) exponents.")
    k2.metric("Predicted pace", _format_pace(pace_min, use_miles))
    k3.metric("Efficiency signal", eff_label,
              help="Requires 6+ runs with HR data inside your effort-band range. Shows 'Not enough data' otherwise. "
                   "Heuristic indicator based on speed/HR ratio trend \u2014 directionally useful but not a calibrated measurement.")
    k4.metric("Load adjustment", risk_label,
              help="Applies a small penalty when recent ACWR risk score exceeds 55.")

    # Explain inactive adjustments so the user knows why changing HR settings has no effect
    _notes = []
    if _eff_inactive:
        lo_pct, hi_pct = int(effort_band[0] * 100), int(effort_band[1] * 100)
        _notes.append(
            f"**Efficiency signal** is inactive — needs 6+ runs with HR between "
            f"{lo_pct}–{hi_pct}% of max HR ({int(effort_band[0]*max_hr)}–{int(effort_band[1]*max_hr)} bpm) "
            f"in the lookback window. The prediction is driven purely by Riegel extrapolation."
        )
    if _risk_inactive:
        _notes.append("**Load adjustment** is inactive — current fatigue risk score is within the safe zone.")
    if _notes:
        st.caption("  \n".join(_notes))

    if pred_lo is not None and pred_hi is not None and np.isfinite(pred_lo) and np.isfinite(pred_hi):
        st.caption(
            f"\U0001f3af Range **{format_hms(pred_lo)} \u2013 {format_hms(pred_hi)}** "
            f"(\u00b1{abs(pred_hi - pred_lo)/60:.0f} min spread). "
            "Riegel accuracy improves with more race-effort runs in the window."
        )

    # M5: HR-based VDOT alternative prediction.
    # For runners who train on hills, pace-based Riegel underestimates capability because
    # all training paces are inflated by elevation. HR-derived VDOT reflects aerobic fitness
    # independently of terrain, giving a fairer flat-course race time estimate.
    _hr_vdot = (vo2max_submax or {}).get("vo2max")
    if _hr_vdot is not None:
        # Race-specific %VO2max fractions (Daniels 2005, table lookup by distance)
        _race_fracs = [(5.0, 0.975), (10.0, 0.940), (21.0975, 0.870), (42.195, 0.810)]
        _closest_frac = min(_race_fracs, key=lambda x: abs(x[0] - race_km))[1]
        _target_vo2 = _closest_frac * _hr_vdot
        _a, _b, _c = 0.000104, 0.182258, -4.60 - _target_vo2
        _disc = _b ** 2 - 4 * _a * _c
        if _disc >= 0:
            _v = (-_b + float(np.sqrt(_disc))) / (2 * _a)   # m/min
            if _v > 0:
                _hr_pace_km = 1000.0 / _v                   # min/km
                _hr_pred_sec = _hr_pace_km * race_km * 60.0
                _diff_sec = pred_sec - _hr_pred_sec
                _diff_min = abs(_diff_sec) / 60.0
                _direction = "faster" if _diff_sec > 0 else "slower"
                _diff_label = f"{int(_diff_min)}:{int((_diff_min % 1)*60):02d} {_direction}"
                with st.expander(
                    f"\U0001f4a1 HR-fitness estimate: **{format_hms(_hr_pred_sec)}** "
                    f"({_diff_label} than Riegel) \u2014 click to learn why"
                ):
                    st.caption(
                        f"Your aerobic HR-based VDOT is **{_hr_vdot:.1f}**, derived from submaximal training runs "
                        f"rather than from race pace. Plugging this into Jack Daniels' formula at "
                        f"**{_closest_frac*100:.0f}% VO\u2082max** (the typical effort fraction at this distance) "
                        f"gives a predicted pace of **{_format_pace(_hr_pace_km, use_miles)}** and finish of **{format_hms(_hr_pred_sec)}**. "
                        "This estimate is particularly useful if your training is mostly on hilly terrain — "
                        "your flat-course paces may be slower than your aerobic engine can actually sustain. "
                        "If the HR estimate is notably faster, consider including flat time-trials in your training "
                        "to verify which estimate better reflects your current capability."
                    )

    if best_actual is not None:
        best_actual_sec = best_actual["pace_min_per_km"] * race_km * 60.0
        diff = pred_sec - best_actual_sec
        sign = "+" if diff >= 0 else "\u2212"
        st.info(
            f"Your best recorded effort at this distance was **{_format_pace(best_actual['pace_min_per_km'], use_miles)}** "
            f"({pd.to_datetime(best_actual['date']).strftime('%b %Y')}) \u2014 "
            f"equivalent to {format_hms(best_actual_sec)}. "
            f"Prediction is {sign}{abs(diff/60):.0f} min relative to that."
        )

    st.divider()
    with st.expander("\U0001f50d Show source runs & model data"):
        st.write(
            f"**Best source run:** {pd.to_datetime(source['start_dt_local']).strftime('%Y-%m-%d')} "
            f"\u2014 {float(source['distance_km']):.1f} km in {format_hms(float(source['duration_min'])*60.0)}"
        )

        # Show the top equivalent performances to be transparent
        df = _model_runs_used[pd.notna(_model_runs_used["distance_km"]) & pd.notna(_model_runs_used["duration_min"])].copy()
        df = df[(df["distance_km"] >= float(min_dist)) & (df["duration_min"] > 3)].copy()
        df["time_sec"] = df["duration_min"] * 60.0
        df["equiv_target_sec"] = df["time_sec"] * (float(race_km) / df["distance_km"]) ** float(exp)
        df["equiv_time"] = df["equiv_target_sec"].apply(format_hms)
        df = df.sort_values("equiv_target_sec").head(12)

        # Build horizontal bar chart with numeric x (seconds) so bar widths are
        # proportional to actual time — then override tick labels with H:MM:SS.
        _x_min = df["equiv_target_sec"].min() * 0.97
        _x_max = df["equiv_target_sec"].max() * 1.03
        _tick_vals = np.linspace(_x_min, _x_max, 6)
        _tick_text = [format_hms(v) for v in _tick_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["equiv_target_sec"],
            y=df["start_dt_local"].dt.strftime("%Y-%m-%d"),
            orientation="h",
            name="Equivalent finish time",
            text=df["equiv_time"],
            textposition="outside",
            marker_color="#6baed6",
        ))
        fig.update_layout(
            height=420, margin=dict(l=10, r=120, t=60, b=10),
            title="Top runs by equivalent race performance (shorter bar = better)",
            xaxis=dict(
                title="Equivalent finish time",
                tickvals=_tick_vals,
                ticktext=_tick_text,
                range=[_x_min, _x_max],
            ),
            yaxis_title="Run date",
        )
        st.plotly_chart(fig, use_container_width=True)

        show = df[["start_dt_local", "name", "distance_km", "duration_min", "equiv_time"]].copy()
        show = show.rename(columns={
            "start_dt_local": "Date", "name": "Run name",
            "distance_km": "Distance (km)", "duration_min": "Duration (min)",
            "equiv_time": "Equiv. finish time",
        })
        st.dataframe(show.sort_values("Equiv. finish time"), hide_index=True, use_container_width=True)

    st.divider()

    # ── Pacing strategy ──────────────────────────────────────────
    st.subheader("Pacing strategy")
    st.caption(
        f"Target splits for your predicted {format_hms(pred_sec)} at {race_km:.2f} km. "
        "Negative split = start conservatively and finish strong."
    )

    # Build splits in km internally, convert display values to miles if needed
    n_km = int(np.ceil(float(race_km)))
    km_marks = list(range(1, n_km + 1))
    _split_dist_factor = KM_TO_MILES if use_miles else 1.0
    _split_pace_factor = (1.0 / KM_TO_MILES) if use_miles else 1.0
    split_marks_disp = [round(k * _split_dist_factor, 2) for k in km_marks]
    _pace_lbl = f"Pace ({_p_unit(use_miles)})"

    even_pace     = pace_min  # in min/km always
    neg_first     = even_pace * 1.025
    neg_second    = even_pace * 0.975
    prog_paces    = [neg_first if k <= race_km / 2 else neg_second for k in km_marks]
    even_paces    = [even_pace] * n_km
    even_paces_d  = [p * _split_pace_factor for p in even_paces]
    prog_paces_d  = [p * _split_pace_factor for p in prog_paces]

    def fmt_p(p_disp):
        m = int(p_disp); s = int(round((p_disp - m) * 60))
        return f"{m}:{s:02d}"

    splits_df = pd.DataFrame({
        _d_unit(use_miles): split_marks_disp,
        f"Even ({_p_unit(use_miles)})": [fmt_p(p) for p in even_paces_d],
        f"Negative ({_p_unit(use_miles)})": [fmt_p(p) for p in prog_paces_d],
    })

    fig_splits = go.Figure()
    fig_splits.add_trace(go.Bar(
        x=split_marks_disp, y=even_paces_d, name="Even split",
        marker_color="rgba(107,174,214,0.7)", text=[fmt_p(p) for p in even_paces_d],
        textposition="outside",
    ))
    fig_splits.add_trace(go.Bar(
        x=split_marks_disp, y=prog_paces_d, name="Negative split",
        marker_color="rgba(82,232,138,0.7)", text=[fmt_p(p) for p in prog_paces_d],
        textposition="outside",
    ))
    fig_splits.update_yaxes(autorange="reversed", title=_pace_lbl)
    fig_splits.update_layout(
        height=360, barmode="group", xaxis_title=_d_unit(use_miles),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_splits, use_container_width=True)

    with st.expander("Show full split table"):
        st.dataframe(splits_df, hide_index=True, use_container_width=True)
