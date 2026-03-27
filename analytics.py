"""
analytics.py — pure computation (no Streamlit).
All analytics functions extracted from app.py.
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    KM_TO_MILES,
    LONG_RUN_DEFAULTS,
    RUN_TYPE_COLORS,
    RUN_TYPE_PRIORITY,
    WORKOUT_TYPE_MAP,
    _DEFAULT_HR_ZONES,
    _EASY_KW,
    _LONG_KW,
    _RACE_KW,
    _WORKOUT_KW,
)


# -----------------------------
# Run type classification
# -----------------------------
def classify_run(
    row: pd.Series,
    streams: Optional[Dict[str, Any]],
    long_run_km: float,
    max_hr: int,
    easy_thresh: float = 0.80,
    tempo_thresh: float = 0.87,
) -> str:
    # 1. Strava workout_type field (set by user in app)
    try:
        wt = int(row.get("workout_type")) if pd.notna(row.get("workout_type")) else None
    except (TypeError, ValueError):
        wt = None
    if wt == 1:
        return "Race"
    if wt == 2:
        return "Long Run"

    dist = float(row.get("distance_km") or 0)
    name = str(row.get("name") or "").lower()

    # 2. Name keyword matching — user intent is the strongest signal
    if any(k in name for k in _RACE_KW):
        return "Race"
    if any(k in name for k in _WORKOUT_KW):
        return "Workout"
    if any(k in name for k in _LONG_KW) and dist >= long_run_km * 0.70:
        return "Long Run"
    if any(k in name for k in _EASY_KW):
        return "Easy"

    # 3. Distance threshold → Long Run
    if dist >= long_run_km:
        return "Long Run"

    # 4. Streams-based classification
    if isinstance(streams, dict):
        v_obj  = streams.get("velocity_smooth", {})
        hr_obj = streams.get("heartrate", {})
        v_data  = v_obj.get("data",  []) if isinstance(v_obj,  dict) else []
        hr_data = hr_obj.get("data", []) if isinstance(hr_obj, dict) else []

        if len(v_data) > 60:
            v = np.array(v_data, dtype=float)
            v = v[np.isfinite(v) & (v > 0.5)]
            if len(v) > 60 and np.mean(v) > 1e-6:
                cv = float(np.std(v) / np.mean(v))
                if cv > 0.28:
                    return "Workout"

        if len(hr_data) > 60:
            hr = np.array(hr_data, dtype=float)
            hr = hr[np.isfinite(hr) & (hr > 40)]
            if len(hr) > 60:
                # Use the middle 80% of the run to exclude warmup/cooldown
                lo_i = int(len(hr) * 0.10)
                hi_i = int(len(hr) * 0.90)
                hr_mid = hr[lo_i:hi_i]
                pct = float(np.median(hr_mid)) / float(max_hr)
                if pct >= tempo_thresh:
                    return "Tempo"
                if pct < easy_thresh:
                    return "Easy"

    # 5. Activity-level avg_hr fallback
    avg_hr = row.get("avg_hr")
    if pd.notna(avg_hr):
        pct = float(avg_hr) / float(max_hr)
        if pct >= tempo_thresh:
            return "Tempo"
        if pct < easy_thresh:
            return "Easy"

    return "General"


def classify_all_runs(
    activities: pd.DataFrame,
    streams_by_id: Dict[int, Dict[str, Any]],
    long_run_km: float,
    max_hr: int,
    easy_thresh: float = 0.80,
    tempo_thresh: float = 0.87,
) -> pd.Series:
    labels = [
        classify_run(row, streams_by_id.get(int(row["id"])), long_run_km, max_hr, easy_thresh, tempo_thresh)
        for _, row in activities.iterrows()
    ]
    return pd.Series(labels, index=activities.index, name="run_type")


# -----------------------------
# Tab 1 computations
# -----------------------------
def build_daily_weekly(
    activities: pd.DataFrame,
    max_hr: int,
    date_range: Tuple[pd.Timestamp, pd.Timestamp],
    rest_hr: int = 50,
    gender: str = "Men",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start, end = date_range
    # Use ALL activities for EWMA so chronic load (28d) has proper history even when
    # the selected date range is short. We filter the returned data to date_range at the end.
    df = activities.copy()

    df["hr_intensity"] = df["avg_hr"] / float(max_hr)

    # Full Banister TRIMP using heart-rate reserve (HRr).
    # Men:   TRIMP = t × HRr × 0.64 × e^(1.92·HRr)
    # Women: TRIMP = t × HRr × 0.86 × e^(1.67·HRr)
    # Source: Banister EW (1991). Physiological Testing of Elite Athletes.
    _hr_range = max(float(max_hr) - float(rest_hr), 1.0)
    hr_reserve = ((df["avg_hr"] - float(rest_hr)) / _hr_range).clip(lower=0.0, upper=1.0)
    if gender == "Women":
        _trimp = df["duration_min"] * hr_reserve * 0.86 * np.exp(1.67 * hr_reserve)
    elif gender in ("Non-binary", "Prefer not to say"):
        # Averaged Banister coefficients across both sexes
        # b = (0.64 + 0.86) / 2 = 0.75 ; e_coef = (1.92 + 1.67) / 2 = 1.795
        _trimp = df["duration_min"] * hr_reserve * 0.75 * np.exp(1.795 * hr_reserve)
    else:  # "Men" or unrecognised
        _trimp = df["duration_min"] * hr_reserve * 0.64 * np.exp(1.92 * hr_reserve)
    df["load_hr"] = _trimp.where(df["avg_hr"].notna(), 0.0)

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            daily_distance_km=("distance_km", "sum"),
            daily_duration_min=("duration_min", "sum"),
            daily_load_hr=("load_hr", "sum"),
            runs=("id", "count"),
            hr_runs=("avg_hr", lambda s: int(pd.notna(s).sum())),
        )
        .copy()
    )

    if len(daily) == 0:
        return pd.DataFrame(), pd.DataFrame()

    min_d = pd.to_datetime(daily["date"]).min()
    max_d = pd.to_datetime(daily["date"]).max()
    all_days = pd.date_range(min_d, max_d, freq="D")
    daily_full = pd.DataFrame({"date": all_days.date}).merge(daily, on="date", how="left")

    for c in ["daily_distance_km", "daily_duration_min", "daily_load_hr", "runs", "hr_runs"]:
        daily_full[c] = daily_full[c].fillna(0.0)

    daily_full["acute_load"] = daily_full["daily_load_hr"].ewm(span=7, adjust=False).mean()
    daily_full["chronic_load"] = daily_full["daily_load_hr"].ewm(span=28, adjust=False).mean()
    daily_full["tsb"] = daily_full["chronic_load"] - daily_full["acute_load"]
    daily_full["acwr"] = np.where(
        daily_full["chronic_load"] > 1e-6,
        daily_full["acute_load"] / daily_full["chronic_load"],
        np.nan,
    )

    daily_full["date_ts"] = pd.to_datetime(daily_full["date"])
    daily_full["week_start"] = daily_full["date_ts"] - pd.to_timedelta(daily_full["date_ts"].dt.weekday, unit="D")

    weekly = (
        daily_full.groupby("week_start", as_index=False)
        .agg(
            weekly_distance_km=("daily_distance_km", "sum"),
            weekly_duration_min=("daily_duration_min", "sum"),
            weekly_load_hr=("daily_load_hr", "sum"),
        )
        .copy()
    )

    df["date_ts"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date_ts"] - pd.to_timedelta(df["date_ts"].dt.weekday, unit="D")
    long_run = df.groupby("week_start", as_index=False).agg(long_run_km=("distance_km", "max"))
    weekly = weekly.merge(long_run, on="week_start", how="left")
    weekly["long_run_km"] = weekly["long_run_km"].fillna(0.0)

    def monotony_for_week(x: pd.Series) -> float:
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=0))
        if sd < 1e-6:
            return np.nan
        return mu / sd

    m = daily_full.groupby("week_start")["daily_load_hr"].apply(monotony_for_week).reset_index(name="monotony")
    weekly = weekly.merge(m, on="week_start", how="left")
    weekly["strain"] = weekly["weekly_load_hr"] * weekly["monotony"]

    # Filter to the selected date range for display (EWMA was computed on full history above)
    daily_out = daily_full[(daily_full["date_ts"] >= start) & (daily_full["date_ts"] <= end)].copy()
    weekly_out = weekly[
        (weekly["week_start"] >= (start - pd.Timedelta(days=6))) & (weekly["week_start"] <= end)
    ].copy()

    return daily_out, weekly_out


def acwr_band(acwr: float) -> Tuple[str, str]:
    if acwr is None or (isinstance(acwr, float) and np.isnan(acwr)):
        return ("N/A", "\u26aa\ufe0f")
    if acwr < 0.8:
        return ("Low", "\U0001f535")
    if acwr <= 1.3:
        return ("Good", "\U0001f7e2")
    if acwr <= 1.7:
        return ("High", "\U0001f7e0")
    return ("Very high", "\U0001f534")


# -----------------------------
# Unit helpers
# -----------------------------
def _pace_axis_ticks(min_val: float, max_val: float, step: float = 0.5):
    """
    Generate (tickvals, ticktext) for a Plotly pace axis in M:SS format.
    step is in minutes (default 0.5 = every 30 seconds).
    """
    import math
    if not (np.isfinite(min_val) and np.isfinite(max_val)) or min_val >= max_val:
        return [], []
    lo = math.floor(min_val / step) * step
    hi = math.ceil(max_val / step) * step
    vals, labels = [], []
    v = lo
    while v <= hi + 1e-9:
        m = int(v); s = int(round((v - m) * 60))
        if s == 60: m += 1; s = 0
        vals.append(round(v, 6))
        labels.append(f"{m}:{s:02d}")
        v += step
    return vals, labels


def _format_pace(min_per_km: float, use_miles: bool = False) -> str:
    """Format pace as M:SS/km or M:SS/mi."""
    if min_per_km is None or (isinstance(min_per_km, float) and np.isnan(min_per_km)):
        return "\u2014"
    val = min_per_km / KM_TO_MILES if use_miles else min_per_km
    m = int(val)
    s = int(round((val - m) * 60))
    if s == 60:
        m += 1; s = 0
    unit = "mi" if use_miles else "km"
    return f"{m}:{s:02d}/{unit}"


def _format_min_per_km(x: float, use_miles: bool = False) -> str:
    """Legacy alias kept for compatibility."""
    return _format_pace(x, use_miles)


def _dist_fmt(km: float, use_miles: bool = False, decimals: int = 1) -> str:
    """Format distance with unit label."""
    if use_miles:
        return f"{km * KM_TO_MILES:.{decimals}f} mi"
    return f"{km:.{decimals}f} km"


def _d_unit(use_miles: bool = False) -> str:
    return "mi" if use_miles else "km"


def _p_unit(use_miles: bool = False) -> str:
    return "min/mi" if use_miles else "min/km"


def _to_display_dist(km: float, use_miles: bool = False) -> float:
    """Convert km value to display units (km or mi)."""
    return km * KM_TO_MILES if use_miles else km


# -----------------------------
# Tab 3 computations (streams)
# -----------------------------
def _safe_array(streams: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    obj = streams.get(key)
    if not isinstance(obj, dict):
        return None
    data = obj.get("data")
    if not isinstance(data, list) or len(data) == 0:
        return None
    return np.asarray(data, dtype=float)


def compute_fatigue_metrics_for_activity(activity_row: pd.Series, streams: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = _safe_array(streams, "time")
    d = _safe_array(streams, "distance")
    v = _safe_array(streams, "velocity_smooth")
    hr = _safe_array(streams, "heartrate")

    if t is None or d is None or v is None:
        return None

    n = min(len(t), len(d), len(v), len(hr) if hr is not None else len(t))
    t = t[:n]
    d = d[:n]
    v = v[:n]
    if hr is not None:
        hr = hr[:n]

    v = np.where(v > 0, v, np.nan)
    pace_min_km = (1000.0 / v) / 60.0
    speed_mps = v

    total_time = float(np.nanmax(t)) if np.isfinite(np.nanmax(t)) else float(len(t))
    if total_time <= 0:
        return None

    # Trim first 90 seconds to remove GPS lock / watch-start artifacts
    _warmup_mask = t >= 90.0
    if np.sum(_warmup_mask) > 60:  # only trim if enough samples remain
        t = t[_warmup_mask]
        d = d[_warmup_mask]
        v = v[_warmup_mask]
        pace_min_km = pace_min_km[_warmup_mask]
        speed_mps = speed_mps[_warmup_mask]
        if hr is not None:
            hr = hr[_warmup_mask]
        total_time = float(np.nanmax(t)) - float(np.nanmin(t))

    _t_start = float(np.nanmin(t))
    half = _t_start + total_time * 0.5
    first = t <= half
    second = t > half

    def _nanmean(x):
        return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan

    pace_first = _nanmean(pace_min_km[first])
    pace_second = _nanmean(pace_min_km[second])

    pace_fade_pct = np.nan
    if np.isfinite(pace_first) and pace_first > 0 and np.isfinite(pace_second):
        pace_fade_pct = (pace_second - pace_first) / pace_first

    hr_drift_pct = np.nan
    decoupling = np.nan
    if hr is not None and np.any(np.isfinite(hr[first])) and np.any(np.isfinite(hr[second])):
        hr_first = _nanmean(hr[first])
        hr_second = _nanmean(hr[second])
        if np.isfinite(hr_first) and hr_first > 0 and np.isfinite(hr_second):
            hr_drift_pct = (hr_second - hr_first) / hr_first

        sp_first = _nanmean(speed_mps[first])
        sp_second = _nanmean(speed_mps[second])
        if np.isfinite(sp_first) and np.isfinite(sp_second) and np.isfinite(hr_first) and np.isfinite(hr_second) and hr_first > 0 and hr_second > 0:
            eff_first = sp_first / hr_first
            eff_second = sp_second / hr_second
            if eff_first > 0:
                decoupling = (eff_second / eff_first) - 1.0

    out = {
        "id": int(activity_row["id"]),
        "start_dt_local": activity_row["start_dt_local"],
        "name": activity_row.get("name", ""),
        "distance_km": float(activity_row["distance_km"]),
        "duration_min": float(activity_row["duration_min"]),
        "pace_fade_pct": pace_fade_pct,
        "hr_drift_pct": hr_drift_pct,
        "decoupling": decoupling,
        "samples": int(n),
        "has_hr_stream": hr is not None,
    }
    return out


def build_fatigue_table(activities_range: pd.DataFrame, streams_by_id: Dict[int, Dict[str, Any]], long_run_min_km: float) -> pd.DataFrame:
    candidates = activities_range[activities_range["distance_km"] >= long_run_min_km].copy()
    candidates = candidates.sort_values("start_dt_local")

    rows = []
    for _, r in candidates.iterrows():
        aid = int(r["id"])
        s = streams_by_id.get(aid)
        if not isinstance(s, dict):
            continue
        m = compute_fatigue_metrics_for_activity(r, s)
        if m is not None:
            rows.append(m)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("start_dt_local")


def build_within_run_df(streams: Dict[str, Any]) -> Optional[pd.DataFrame]:
    t = _safe_array(streams, "time")
    d = _safe_array(streams, "distance")
    v = _safe_array(streams, "velocity_smooth")
    hr = _safe_array(streams, "heartrate")

    if t is None or d is None or v is None:
        return None

    n = min(len(t), len(d), len(v), len(hr) if hr is not None else len(t))
    t = t[:n]
    d = d[:n]
    v = v[:n]
    hr = hr[:n] if hr is not None else np.full(n, np.nan)

    # Trim pre-run standing period (GPS noise / watch-start before first stride)
    # Find the first index where velocity clearly indicates running (> 1 m/s ≈ 16 min/km)
    _first_move = int(np.argmax(v > 1.0)) if np.any(v > 1.0) else 0
    if _first_move > 0:
        t = t[_first_move:]
        d = d[_first_move:]
        v = v[_first_move:]
        hr = hr[_first_move:]
        d = d - d[0]  # reset distance to 0 from start of actual running

    v = np.where(v > 0, v, np.nan)
    pace_min_km = (1000.0 / v) / 60.0

    df = pd.DataFrame({
        "time_s": t,
        "distance_km": d / 1000.0,
        "pace_min_km": pace_min_km,
        "heartrate": hr,
    })

    df["pace_smooth"] = df["pace_min_km"].rolling(window=30, min_periods=1).median()
    df["hr_smooth"] = df["heartrate"].rolling(window=30, min_periods=1).median()

    return df


# -----------------------------
# Tab 4 computations (readiness/risk proxies)
# -----------------------------
def zscore(x: pd.Series) -> pd.Series:
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < 1e-9:
        return (x * 0.0)
    return (x - mu) / sd


def compute_risk_table(daily: pd.DataFrame, weekly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_risk, weekly_risk) with rule-based risk flags and a composite score."""
    d = daily.copy().sort_values("date_ts")
    w = weekly.copy().sort_values("week_start")

    # Daily: add recent rest indicator
    d["is_rest_day"] = (d["daily_load_hr"] <= 1e-9).astype(int)
    d["rest_days_last7"] = d["is_rest_day"].rolling(window=7, min_periods=1).sum()

    # Composite daily risk score [0–100].
    # Each component is normalised to [0, 1] before weighting so the weighted
    # sum stays in [0, 1] and multiplying by 100 gives a true percentage.
    #
    # Component definitions:
    #   acwr_component : 0 at ACWR ≤ 1.0 (balanced), 1.0 at ACWR ≥ 2.0 (very high)
    #   acute_component: 0 at or below the historical mean ATL,
    #                    1.0 at +2 standard deviations above it
    #   rest_component : 0 with ≥ 3 rest days in the past 7 days, 1.0 with 0 rest days
    acwr = d["acwr"].copy()
    acwr_component = ((acwr.clip(lower=0.0, upper=2.0) - 1.0) / 1.0).clip(lower=0.0, upper=1.0)
    acute_component = (zscore(d["acute_load"]).clip(lower=0.0, upper=2.0) / 2.0).fillna(0.0)
    rest_component  = ((3.0 - d["rest_days_last7"]).clip(lower=0.0) / 3.0).clip(upper=1.0)

    d["risk_score"] = 100.0 * (
        0.50 * acwr_component.fillna(0.0)
        + 0.30 * acute_component
        + 0.20 * rest_component.fillna(0.0)
    )
    d["risk_score"] = d["risk_score"].clip(lower=0.0, upper=100.0)

    # Daily flags
    d["flag_acwr_high"] = (d["acwr"] > 1.5).astype(int)
    d["flag_acwr_very_high"] = (d["acwr"] > 1.8).astype(int)
    d["flag_low_rest"] = (d["rest_days_last7"] < 2).astype(int)
    d["flag_big_day"] = (d["daily_load_hr"] > d["daily_load_hr"].rolling(28, min_periods=7).mean() + 2.0 * d["daily_load_hr"].rolling(28, min_periods=7).std()).astype(int)

    # Weekly: spikes and monotony
    w["weekly_load_4wk_avg"] = w["weekly_load_hr"].rolling(window=4, min_periods=1).mean()
    w["load_change_pct"] = np.where(w["weekly_load_4wk_avg"] > 1e-6, (w["weekly_load_hr"] / w["weekly_load_4wk_avg"]) - 1.0, np.nan)

    w["flag_load_spike"] = (w["load_change_pct"] > 0.25).astype(int)  # >25% above 4wk avg
    w["flag_monotony_high"] = (w["monotony"] > 2.0).astype(int)
    w["flag_strain_high"] = (w["strain"] > w["strain"].rolling(8, min_periods=4).quantile(0.85)).astype(int)

    return d, w


def compute_compromised_runs(df_range: pd.DataFrame, max_hr: int) -> pd.DataFrame:
    """
    Label 'compromised' runs as those with unusually low efficiency at a given HR intensity.
    Proxy: speed_per_hr below rolling 20th percentile of last 12 HR-bearing runs.

    C2 FIX: baseline is computed from easy-intensity runs only (Easy, Long Run, General).
    Mixing tempo/race efforts into the baseline skews it upward, causing recovery runs
    to appear 'compromised' simply because they are slower than a mixed baseline.
    The detector now compares like-for-like: easy run vs. easy-run baseline.
    """
    d = df_range.copy()
    d = d[pd.notna(d["avg_hr"]) & pd.notna(d["avg_speed_mps"])].copy()
    if len(d) == 0:
        return pd.DataFrame()

    d["hr_intensity"] = d["avg_hr"] / float(max_hr)
    d["speed_per_hr"] = d["avg_speed_mps"] / d["avg_hr"]
    d = d.sort_values("start_dt_local").reset_index(drop=True)

    # Build baseline from easy-intensity runs only so quality sessions don't
    # inflate the reference, causing low false-positive flags on recovery days.
    _easy_types = {"Easy", "Long Run", "General"}
    _has_type = "run_type" in d.columns
    _easy_mask = d["run_type"].isin(_easy_types) if _has_type else pd.Series(True, index=d.index)
    _easy_runs = d[_easy_mask][["start_dt_local", "speed_per_hr"]].copy()

    def roll_q(x: pd.Series, q: float) -> pd.Series:
        return x.rolling(window=12, min_periods=6).quantile(q)

    # Compute rolling 20th-percentile baseline on the easy subset, then merge back
    _easy_runs = _easy_runs.sort_values("start_dt_local").reset_index(drop=True)
    _easy_runs["eff_q20"] = roll_q(_easy_runs["speed_per_hr"], 0.20)

    d = d.merge(
        _easy_runs[["start_dt_local", "eff_q20"]],
        on="start_dt_local", how="left",
    )
    # Forward-fill so non-easy runs get the most recent easy-run baseline
    d["eff_q20"] = d["eff_q20"].ffill()

    d["compromised"] = (d["speed_per_hr"] < d["eff_q20"]).astype(int)
    d["eff_delta"] = d["speed_per_hr"] - d["eff_q20"]

    return d


# -----------------------------
# Tab 5 computations (race prediction)
# -----------------------------
def calibrate_riegel_exponent(bests: dict) -> Optional[float]:
    """
    Fit the individual's personal Riegel fatigue exponent from their recorded efforts
    at ≥2 standard distances using ordinary least-squares on the log-log relationship:

        log(T2/T1) = exponent × log(D2/D1)

    The population default of 1.06 is a cross-sectional average; better runners
    typically cluster around 1.02–1.04 and recreational runners around 1.07–1.12.
    A personal exponent calibrated from the athlete's own data is more accurate.

    Returns None if fewer than 2 distances are available.
    Clipped to [1.00, 1.20] to exclude physiologically implausible values.
    """
    from itertools import combinations as _comb
    _dist_map = {
        "best_5k":       5.0,
        "best_10k":      10.0,
        "best_hm":       21.0975,
        "best_marathon": 42.195,
    }
    efforts: Dict[float, float] = {}
    for key, dist_km in _dist_map.items():
        if key in bests:
            b = bests[key]
            # Use median pace (more stable than single-best)
            time_sec = b["pace_min_per_km"] * dist_km * 60.0
            if time_sec > 0:
                efforts[dist_km] = time_sec

    if len(efforts) < 2:
        return None

    log_d, log_t = [], []
    for d1, d2 in _comb(sorted(efforts.keys()), 2):
        t1, t2 = efforts[d1], efforts[d2]
        if t1 > 0 and t2 > 0:
            log_d.append(float(np.log(d2 / d1)))
            log_t.append(float(np.log(t2 / t1)))

    if not log_d:
        return None

    ld = np.array(log_d)
    lt = np.array(log_t)
    # OLS through origin: exponent = Σ(ld·lt) / Σ(ld²)
    exp_val = float(np.dot(ld, lt) / np.dot(ld, ld))
    return round(float(np.clip(exp_val, 1.00, 1.20)), 3)


def format_hms(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "\u2014"
    seconds = float(max(0.0, seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(round(seconds % 60))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        h += 1
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


def predict_race_time_riegel(
    runs: pd.DataFrame,
    target_km: float,
    exponent: float = 1.06,
    min_km: float = 5.0,
    max_km: float = 1000.0,
    max_ratio: float = 3.0,
    top_n: int = 5,
) -> Tuple[Optional[float], Optional[pd.Series]]:
    """
    Predict race time using the Riegel power-law model:
        T2 = T1 × (D2 / D1)^exponent

    Professional-grade implementation:
    - min_km / max_km filter removes very short or unrealistically long runs.
    - max_ratio cap (default 3.0): source distance must be ≥ target_km / max_ratio.
      Riegel's formula is most accurate within 3× extrapolation; beyond this,
      glycogen dynamics and mechanical fatigue diverge significantly from the model.
    - Returns the *median of the top_n best* equivalent-time predictions rather
      than the raw minimum, which is too sensitive to a single exceptional day.
    - Source row returned is the single best individual run (for display).

    Returns (pred_seconds, source_run_row).  Both are None if no eligible runs.
    """
    if runs is None or len(runs) == 0 or target_km <= 0:
        return None, None

    df = runs.copy()
    df = df[pd.notna(df["distance_km"]) & pd.notna(df["duration_min"])].copy()

    # Basic sanity gates
    min_source_km = max(float(min_km), float(target_km) / float(max_ratio))
    df = df[
        (df["distance_km"] >= min_source_km)
        & (df["distance_km"] <= float(max_km))
        & (df["duration_min"] > 3)
        & (df["distance_km"] > 0)
    ].copy()
    if len(df) == 0:
        return None, None

    df["time_sec"] = df["duration_min"] * 60.0
    # Guard against degenerate paces (< 2 min/km or > 15 min/km)
    df["pace_min_km"] = df["time_sec"] / 60.0 / df["distance_km"]
    df = df[(df["pace_min_km"] >= 2.0) & (df["pace_min_km"] <= 15.0)].copy()
    if len(df) == 0:
        return None, None

    df["pred_sec"] = df["time_sec"] * (float(target_km) / df["distance_km"]) ** float(exponent)

    # Use median of top_n to reduce single-run fluke sensitivity
    top = df.nsmallest(min(top_n, len(df)), "pred_sec")
    pred_sec = float(top["pred_sec"].median())
    source_row = top.iloc[0]  # fastest individual run for display
    return pred_sec, source_row


def compute_efficiency_adjustment(
    runs: pd.DataFrame,
    max_hr: int,
    effort_band: Tuple[float, float],
    lookback_days: int,
    end_ts: pd.Timestamp,
) -> float:
    """
    Adjustment factor based on race-effort efficiency trend.
    - Compute speed_per_hr = avg_speed_mps / avg_hr (higher is better).
    - Compare recent median (last ~lookback_days) vs baseline median (prior history).
    Returns multiplicative time factor ( <1 => faster, >1 => slower ).
    """
    if runs is None or len(runs) == 0:
        return 1.0

    d = runs.copy()
    d = d[pd.notna(d["avg_hr"]) & pd.notna(d["avg_speed_mps"])].copy()
    if len(d) < 8:
        return 1.0

    d["hr_intensity"] = d["avg_hr"] / float(max_hr)
    lo, hi = effort_band
    d = d[(d["hr_intensity"] >= lo) & (d["hr_intensity"] <= hi)].copy()
    if len(d) < 6:
        return 1.0

    d["speed_per_hr"] = d["avg_speed_mps"] / d["avg_hr"]
    d = d.sort_values("start_dt_local")

    cutoff = end_ts - pd.Timedelta(days=int(lookback_days))
    recent = d[d["start_dt_local"] >= cutoff]
    baseline = d[d["start_dt_local"] < cutoff]

    if len(recent) < 3 or len(baseline) < 3:
        return 1.0

    recent_med = float(np.nanmedian(recent["speed_per_hr"]))
    base_med = float(np.nanmedian(baseline["speed_per_hr"]))
    if not np.isfinite(recent_med) or not np.isfinite(base_med) or recent_med <= 1e-12 or base_med <= 1e-12:
        return 1.0

    # If efficiency improved (recent > baseline) -> time factor < 1
    ratio = base_med / recent_med  # >1 means you got more efficient -> faster
    # Dampen + clip to keep conservative
    factor = float(np.clip(ratio ** 0.6, 0.90, 1.10))
    return factor


# -----------------------------
# Personal bests
# -----------------------------
def _fastest_segment_in_stream(
    dist_m: np.ndarray, time_s: np.ndarray, target_m: float
) -> Optional[float]:
    """
    Two-pointer sliding window: find the minimum pace (min/km) to cover
    `target_m` consecutive metres anywhere within a GPS stream.

    This catches embedded PRs — e.g. a 5 K best hidden inside a 10 K race.
    Requires 90 % of target distance to be present in the stream.
    Pace is interpolated to exactly target_m so it is directly comparable
    with whole-activity pace values.
    """
    n = len(dist_m)
    total_dist = float(dist_m[-1]) - float(dist_m[0])
    if n < 2 or total_dist < target_m * 0.9:
        return None
    best_pace: Optional[float] = None
    j = 0
    for i in range(n):
        # Advance j until we've covered at least target_m from i
        while j < n - 1 and (dist_m[j] - dist_m[i]) < target_m:
            j += 1
        covered = float(dist_m[j] - dist_m[i])
        if covered < target_m * 0.9:
            continue
        elapsed_s = float(time_s[j] - time_s[i])
        if elapsed_s <= 0:
            continue
        # Interpolate elapsed time to exactly target_m
        elapsed_adj = elapsed_s * (target_m / covered)
        pace = (elapsed_adj / 60.0) / (target_m / 1000.0)  # min/km
        # Sanity guard: 1:45/km (world-record-ish) to 12:00/km (brisk walk)
        if 1.75 < pace < 12.0:
            if best_pace is None or pace < best_pace:
                best_pace = pace
    return best_pace


def compute_personal_bests(
    activities: pd.DataFrame,
    streams_by_id: Optional[Dict[int, Any]] = None,
    recent_days: Optional[int] = None,
    top_n: int = 3,
) -> dict:
    """
    Compute personal bests for standard race distances using the median of the
    top-N fastest efforts per distance — more robust than a single best time.

    Methodology
    -----------
    For each target distance:
    1. Collect the best pace per unique activity from both whole-activity bands
       and GPS-stream sliding-window search (catches embedded efforts, e.g. a
       5 K best inside a 10 K race).
    2. One pace per activity — the faster of the two methods — so the same run
       can never inflate the sample.
    3. Take the top-N by pace, compute the median.  If fewer than N efforts are
       available, use what exists (graceful degradation).
    4. Report n_efforts so the UI can label "median of 3 × 5K" vs "single 5K".

    Parameters
    ----------
    activities    : full activity DataFrame
    streams_by_id : per-second GPS streams (enables embedded-PR detection)
    recent_days   : if set, only consider activities from the last N days
    top_n         : number of top efforts to median (default 3)
    """
    df = activities[pd.notna(activities["distance_km"]) & pd.notna(activities["pace_min_per_km"])].copy()
    df = df[df["duration_min"] > 3].copy()

    if recent_days is not None:
        _cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=recent_days)
        _dt = pd.to_datetime(df["start_dt_local"]).dt.tz_localize(None)
        df = df[_dt >= _cutoff].copy()

    bests: dict = {}

    # Canonical distances and activity-level distance bands
    # S4 FIX: tighter bands reduce contamination from runs that are clearly not
    # race-distance efforts (e.g. a 6.8 km easy run polluting the 5K pool).
    targets = {
        "best_5k":       (5_000.0,  4.5,   5.5),
        "best_10k":      (10_000.0, 9.0,  11.0),
        "best_hm":       (21_097.5, 20.0, 22.5),
        "best_marathon": (42_195.0, 40.0, 44.0),
    }

    # Build activity lookups for stream search
    _id_to_date: Dict[int, Any] = {}
    _id_to_dist: Dict[int, float] = {}
    if "id" in df.columns and "start_dt_local" in df.columns:
        for _, row in df.iterrows():
            _id_to_date[int(row["id"])] = row["start_dt_local"]
            _id_to_dist[int(row["id"])] = float(row.get("distance_km", 0))

    # S6: when streams are loaded, skip the activity-level fallback so that
    # warmup/cooldown kilometres don't deflate the pace estimate.
    _streams_loaded = bool(streams_by_id)

    for key, (target_m, lo_km, hi_km) in targets.items():
        # ── Per-activity best pace pool ──────────────────────────────────
        # {act_id: (pace_min_per_km, date, source)}
        _pool: Dict[int, tuple] = {}

        # 1. Activity-level: whole-run falls within distance band.
        #    Skipped when GPS streams are available — the sliding-window search
        #    (step 2) is more accurate because it strips warmup/cooldown km.
        if not _streams_loaded:
            sub = df[(df["distance_km"] >= lo_km) & (df["distance_km"] <= hi_km)]
            for _, row in sub.iterrows():
                act_id = int(row["id"])
                pace   = float(row["pace_min_per_km"])
                if act_id not in _pool or pace < _pool[act_id][0]:
                    _pool[act_id] = (pace, row["start_dt_local"], "activity")

        # 2. Stream-level: sliding-window search for fastest segment
        if streams_by_id:
            for act_id, streams in streams_by_id.items():
                if act_id not in _id_to_date:
                    continue
                if _id_to_dist.get(act_id, 0) < (target_m / 1000.0) * 0.9:
                    continue
                if not isinstance(streams, dict):
                    continue
                dist_obj = streams.get("distance", {})
                time_obj = streams.get("time", {})
                if not (isinstance(dist_obj, dict) and isinstance(time_obj, dict)):
                    continue
                dist_data = dist_obj.get("data", [])
                time_data = time_obj.get("data", [])
                if len(dist_data) < 10 or len(time_data) < 10:
                    continue
                dist_arr = np.array(dist_data, dtype=float)
                time_arr = np.array(time_data, dtype=float)
                n = min(len(dist_arr), len(time_arr))
                stream_pace = _fastest_segment_in_stream(
                    dist_arr[:n], time_arr[:n], target_m
                )
                if stream_pace is None:
                    continue
                # Keep the faster of activity-level or stream-level for this run
                if act_id not in _pool or stream_pace < _pool[act_id][0]:
                    _pool[act_id] = (stream_pace, _id_to_date[act_id], "stream")

        if not _pool:
            continue

        # ── Top-N median ─────────────────────────────────────────────────
        sorted_efforts = sorted(_pool.values(), key=lambda x: x[0])[:top_n]
        paces       = [e[0] for e in sorted_efforts]
        median_pace = float(np.median(paces))
        n_efforts   = len(paces)
        # Date + source come from the single fastest effort (for display)
        best_pace, best_date, best_source = sorted_efforts[0]
        # All effort dates — shown in UI so user knows which runs were used (S3)
        effort_dates = [e[1] for e in sorted_efforts]

        bests[key] = {
            "pace_min_per_km":      median_pace,   # median — used for VDOT
            "pace_min_per_km_best": best_pace,      # outright fastest — for display
            "distance_km":          target_m / 1000.0,
            "date":                 best_date,
            "source":               best_source,
            "n_efforts":            n_efforts,
            "effort_dates":         effort_dates,  # S3: dates of all top-N efforts used
        }

    # ── Longest run ──────────────────────────────────────────────────────
    if len(df) > 0:
        idx = df["distance_km"].idxmax()
        r = df.loc[idx]
        bests["longest_run"] = {
            "distance_km": float(r["distance_km"]),
            "date":        r["start_dt_local"],
        }
    return bests


# -----------------------------
# HR zone breakdown
# -----------------------------
def make_hr_zones(z1_max: float, z2_max: float, z3_max: float, z4_max: float):
    """Build HR zone list from user-defined boundaries (as fractions of max HR)."""
    return [
        ("Z1 Recovery",  0.00,  z1_max),
        ("Z2 Aerobic",   z1_max, z2_max),
        ("Z3 Tempo",     z2_max, z3_max),
        ("Z4 Threshold", z3_max, z4_max),
        ("Z5 VO\u2082max",   z4_max, 9.99),
    ]


def compute_hr_zones(
    df_range: pd.DataFrame,
    streams_by_id: Dict[int, Dict[str, Any]],
    max_hr: int,
    hr_zones=None,
) -> pd.DataFrame:
    """Compute minutes in each HR zone using per-second stream data where available."""
    if hr_zones is None:
        hr_zones = _DEFAULT_HR_ZONES
    totals = {z[0]: 0.0 for z in hr_zones}

    for _, row in df_range.iterrows():
        aid = int(row["id"])
        streams = streams_by_id.get(aid)

        if isinstance(streams, dict):
            hr_obj = streams.get("heartrate", {})
            t_obj  = streams.get("time", {})
            if (isinstance(hr_obj, dict) and isinstance(hr_obj.get("data"), list)
                    and isinstance(t_obj, dict) and isinstance(t_obj.get("data"), list)):
                hr_arr = np.array(hr_obj["data"], dtype=float)
                t_arr  = np.array(t_obj["data"],  dtype=float)
                n = min(len(hr_arr), len(t_arr))
                hr_arr, t_arr = hr_arr[:n], t_arr[:n]
                dt = np.diff(t_arr, prepend=t_arr[0])
                dt[0] = 1.0
                intensity = hr_arr / float(max_hr)
                for zname, zlo, zhi in hr_zones:
                    totals[zname] += float(np.nansum(dt[(intensity >= zlo) & (intensity < zhi)]))
                continue

        # Fallback: activity-level avg_hr
        if pd.notna(row.get("avg_hr")) and pd.notna(row.get("duration_min")):
            intensity = float(row["avg_hr"]) / float(max_hr)
            dur_s = float(row["duration_min"]) * 60.0
            for zname, zlo, zhi in hr_zones:
                if zlo <= intensity < zhi:
                    totals[zname] += dur_s
                    break

    return pd.DataFrame([{"Zone": k, "Minutes": v / 60.0} for k, v in totals.items()])


# -----------------------------
# Jack Daniels training paces from VDOT
# -----------------------------
def estimate_training_paces(vdot: float, use_miles: bool = False) -> list:
    """
    Derive Jack Daniels' five training zones from a VDOT estimate.
    Each zone corresponds to a % of VO2max. Velocity is back-calculated
    from the Daniels/Gilbert quadratic: VO2 = -4.60 + 0.182258v + 0.000104v²
    Returns a list of dicts for display.
    Source: Daniels J (2005). Daniels' Running Formula, 2nd ed.
    """
    def _pace_from_frac(frac: float) -> float:
        vo2 = frac * vdot
        a, b, c = 0.000104, 0.182258, -4.60 - vo2
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return np.nan
        v = (-b + np.sqrt(disc)) / (2 * a)  # m/min
        return (1000.0 / v) if v > 0 else np.nan  # min/km

    zones = [
        ("Easy",       0.70, "Conversational. Base building & recovery. Most of your running.",      "65–75% VO₂max"),
        ("Marathon",   0.80, "Aerobic threshold. Goal race pace for the marathon.",                   "75–84% VO₂max"),
        ("Threshold",  0.86, "Comfortably hard. Sustained 20–60 min. Raises lactate threshold.",     "83–88% VO₂max"),
        ("Interval",   0.98, "VO₂max pace. 3–5 min reps with equal recovery.",                       "95–100% VO₂max"),
        ("Repetition", 1.05, "Speed & economy. 200–400 m reps with full recovery.",                  ">100% VO₂max"),
    ]
    rows = []
    for name, frac, purpose, intensity in zones:
        pace_km = _pace_from_frac(frac)
        if use_miles:
            pace_disp = pace_km / KM_TO_MILES if np.isfinite(pace_km) else np.nan
            unit = "min/mi"
        else:
            pace_disp = pace_km
            unit = "min/km"
        if np.isfinite(pace_disp):
            m = int(pace_disp)
            s = int(round((pace_disp - m) * 60))
            if s == 60:
                m += 1; s = 0
            pace_str = f"{m}:{s:02d}/{unit}"
        else:
            pace_str = "—"
        rows.append({"Zone": name, "Pace": pace_str, "Intensity": intensity, "Purpose": purpose})
    return rows


# -----------------------------
# Grade-adjusted pace (Minetti)
# -----------------------------
def compute_activity_gap(streams: dict) -> Optional[float]:
    """
    Compute overall grade-adjusted pace (GAP) for a single activity (min/km).
    Uses the Minetti metabolic cost polynomial on the grade_smooth stream.
    GAP = time-weighted average of (actual_pace × C_flat / C_grade).
    Returns None when grade_smooth stream is unavailable.
    """
    grade_arr = _safe_array(streams, "grade_smooth")
    v_arr     = _safe_array(streams, "velocity_smooth")
    t_arr     = _safe_array(streams, "time")
    if grade_arr is None or v_arr is None or t_arr is None:
        return None

    n = min(len(grade_arr), len(v_arr), len(t_arr))
    grade = np.clip(grade_arr[:n] / 100.0, -0.40, 0.40)
    v     = v_arr[:n]
    t     = t_arr[:n]

    c_g    = 280.5*grade**5 - 58.7*grade**4 - 76.8*grade**3 + 51.9*grade**2 + 19.6*grade + 2.5
    c_flat = 2.5
    gap_factor  = np.where(c_g > 0.5, c_flat / c_g, np.nan)
    pace_min_km = np.where(v > 0, (1000.0 / v) / 60.0, np.nan)
    gap_pace    = pace_min_km * gap_factor

    dt = np.diff(t, prepend=t[0])
    dt[0] = 1.0
    valid = np.isfinite(gap_pace) & (gap_pace > 2.0) & (gap_pace < 20.0)
    if not np.any(valid):
        return None
    return float(np.sum(gap_pace[valid] * dt[valid]) / np.sum(dt[valid]))


# -----------------------------
# VO2max estimate (Jack Daniels VDOT)
# -----------------------------
def estimate_vo2max(bests: dict) -> Optional[float]:
    """Estimate VDOT from best recorded efforts using Jack Daniels' formula.
    Priority: 5K > 10K > Half Marathon > Marathon.
    For recreational athletes, shorter-distance efforts better reflect aerobic ceiling
    without contamination from glycogen depletion or pacing errors (common at marathon).
    """
    priority = [("best_5k", 5.0), ("best_10k", 10.0), ("best_hm", 21.0975), ("best_marathon", 42.195)]
    for key, dist_km in priority:
        if key not in bests:
            continue
        b = bests[key]
        time_sec = b["pace_min_per_km"] * dist_km * 60.0
        time_min = time_sec / 60.0
        v = (dist_km * 1000.0) / time_min  # m/min
        vo2 = -4.60 + 0.182258 * v + 0.000104 * v ** 2
        pct = 0.8 + 0.1894393 * np.exp(-0.012778 * time_min) + 0.2989558 * np.exp(-0.1932605 * time_min)
        vdot = vo2 / pct
        if np.isfinite(vdot) and vdot > 10:
            return round(float(vdot), 1)
    return None


def estimate_vo2max_range(bests: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute VDOT from ALL qualifying distances (5K, 10K, HM, Marathon) and return
    (vdot_low, vdot_high) as the range across those estimates.
    Returns (None, None) when fewer than 2 distances are available — single-effort
    estimates carry no meaningful inter-distance spread to report.
    """
    priority = [("best_marathon", 42.195), ("best_hm", 21.0975), ("best_10k", 10.0), ("best_5k", 5.0)]
    vdots = []
    for key, dist_km in priority:
        if key not in bests:
            continue
        b = bests[key]
        time_sec = b["pace_min_per_km"] * dist_km * 60.0
        time_min = time_sec / 60.0
        v = (dist_km * 1000.0) / time_min
        vo2 = -4.60 + 0.182258 * v + 0.000104 * v ** 2
        pct = 0.8 + 0.1894393 * np.exp(-0.012778 * time_min) + 0.2989558 * np.exp(-0.1932605 * time_min)
        vdot = vo2 / pct
        if np.isfinite(vdot) and vdot > 10:
            vdots.append(round(float(vdot), 1))
    if len(vdots) < 2:
        return None, None
    return round(min(vdots), 1), round(max(vdots), 1)


# -----------------------------
# Submaximal HR-based VO2max
# -----------------------------
def estimate_vo2max_submaximal(
    activities: pd.DataFrame,
    max_hr: int,
    rest_hr: int,
    recent_days: int = 90,
) -> dict:
    """
    Estimate VO2max from submaximal steady-state training runs — no race required.

    Method
    ------
    For each qualifying aerobic run:

        v (m/min)   = distance_m / duration_min
        VO2_run     = 0.2 × v + 3.5          (ACSM flat-ground running O2 cost)
        HRr         = (avg_hr − rest_hr) / (max_hr − rest_hr)   (Karvonen reserve)
        VO2max_i    = VO2_run / HRr           (since VO2 ≈ HRr × VO2max at steady state)

    The median of all per-run estimates is returned as the point estimate.
    The 25th–75th percentile forms the confidence interval.

    Qualifying criteria (steady-state aerobic filter)
    --------------------------------------------------
    • avg_hr present
    • HRr in [0.50, 0.90]  — above easy jog, below near-maximal
      (lower bound raised from 0.40 → 0.50: Swain et al. (1994) only validated
       the linear %HRR↔%VO2max relationship above ~50% HRmax; below this the
       relationship breaks down and estimates skew artificially high)
    • duration ≥ 20 min and distance ≥ 3 km  — steady-state assumption
    • VO2max_i in [20, 90]  — physiological sanity

    Sources
    -------
    • ACSM's Guidelines for Exercise Testing and Prescription, 11th ed.
    • Swain DP et al. (1994). Target HRs for the development of cardiorespiratory
      fitness. Medicine & Science in Sports & Exercise, 26(1), 112–116.
    • Karvonen MJ et al. (1957). The effects of training on heart rate.
      Annales Medicinae Experimentalis et Biologiae Fenniae, 35(3), 307–315.

    Returns dict with keys:
        vo2max        — median estimate (ml/kg/min), or None if insufficient data
        vo2max_low    — 25th percentile
        vo2max_high   — 75th percentile
        n_runs        — number of qualifying runs used
        method        — "submaximal_hr"
    """
    result: dict = {
        "vo2max": None, "vo2max_low": None, "vo2max_high": None,
        "n_runs": 0, "method": "submaximal_hr",
    }

    df = activities.copy()

    # Date filter
    _cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=recent_days)
    _dt = pd.to_datetime(df["start_dt_local"]).dt.tz_localize(None)
    df = df[_dt >= _cutoff].copy()

    # Require HR, duration, distance
    df = df[pd.notna(df["avg_hr"]) & (df["avg_hr"] > 0)].copy()
    df = df[df["duration_min"] >= 20].copy()
    df = df[df["distance_km"] >= 3.0].copy()

    if df.empty:
        return result

    # HR reserve fraction (Karvonen)
    _hr_range = max(float(max_hr) - float(rest_hr), 1.0)
    df["HRr"] = ((df["avg_hr"] - float(rest_hr)) / _hr_range).clip(0.0, 1.0)

    # Steady-state aerobic zone only
    df = df[(df["HRr"] >= 0.50) & (df["HRr"] <= 0.90)].copy()  # C4: raised from 0.40 per Swain et al. (1994)

    if df.empty:
        return result

    # Speed and ACSM VO2
    df["speed_m_per_min"] = df["distance_km"] * 1000.0 / df["duration_min"]
    df["vo2_run"] = 0.2 * df["speed_m_per_min"] + 3.5

    # Per-run VO2max estimate
    df["vo2max_i"] = df["vo2_run"] / df["HRr"]

    # Physiological sanity filter
    df = df[(df["vo2max_i"] >= 20) & (df["vo2max_i"] <= 90)].copy()

    if df.empty:
        return result

    vals = df["vo2max_i"].values
    result.update({
        "vo2max":      round(float(np.median(vals)), 1),
        "vo2max_low":  round(float(np.percentile(vals, 25)), 1),
        "vo2max_high": round(float(np.percentile(vals, 75)), 1),
        "n_runs":      int(len(vals)),
        "method":      "submaximal_hr",
    })
    return result


# -----------------------------
# Consistency & streaks
# -----------------------------
def compute_consistency(activities: pd.DataFrame) -> dict:
    acts = activities.copy().sort_values("start_dt_local")
    acts["week_start"] = acts["start_dt_local"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly_counts = acts.groupby("week_start").size().sort_index()

    # Consecutive week streak (>=1 run per week)
    weeks = weekly_counts.index.tolist()
    week_streak = 0
    for i in range(len(weeks) - 1, -1, -1):
        if i == len(weeks) - 1:
            week_streak = 1
        elif (weeks[i + 1] - weeks[i]).days <= 8:
            week_streak += 1
        else:
            break

    recent = weekly_counts.tail(12)
    pct_3plus = float((recent >= 3).mean() * 100) if len(recent) > 0 else 0.0

    # Longest single-week distance
    acts["week_start2"] = acts["start_dt_local"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly_km = acts.groupby("week_start2")["distance_km"].sum()
    best_week_km = float(weekly_km.max()) if len(weekly_km) > 0 else 0.0

    return {
        "week_streak": week_streak,
        "pct_consistent_weeks": pct_3plus,
        "total_runs": len(acts),
        "total_km": float(acts["distance_km"].sum()),
        "best_week_km": best_week_km,
    }


# -----------------------------
# Training calendar heatmap
# -----------------------------
def build_calendar_heatmap(activities: pd.DataFrame, n_weeks: int = 53, use_miles: bool = False):
    import plotly.graph_objects as go

    acts = activities.copy()
    acts["date"] = acts["start_dt_local"].dt.normalize()

    # Dominant run type per day (highest priority wins)
    has_type = "run_type" in acts.columns
    if has_type:
        acts["type_priority"] = acts["run_type"].map(RUN_TYPE_PRIORITY).fillna(0)
        dominant = acts.loc[acts.groupby("date")["type_priority"].idxmax(), ["date", "run_type"]].set_index("date")["run_type"]

    _has_load = "load_hr" in acts.columns
    _agg_dict = {"distance_km": ("distance_km", "sum"), "n_runs": ("id", "count")}
    if _has_load:
        _agg_dict["load_hr"] = ("load_hr", "sum")
    daily = acts.groupby("date").agg(**_agg_dict).reset_index()

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(weeks=n_weeks)
    all_dates = pd.date_range(start_date, end_date, freq="D")
    cal = pd.DataFrame({"date": all_dates})
    cal = cal.merge(daily, on="date", how="left").fillna({"distance_km": 0.0, "n_runs": 0})
    if has_type:
        cal["run_type"] = cal["date"].map(dominant).fillna("rest")

    cal["week_col"] = ((cal["date"] - start_date).dt.days // 7)
    cal["dow"] = cal["date"].dt.dayofweek

    def _hover(r):
        d_str = r["date"].strftime("%a %b %-d, %Y")
        if r["distance_km"] > 0:
            tag = f" \u00b7 {r['run_type']}" if has_type and r["run_type"] != "rest" else ""
            dist_str = _dist_fmt(r["distance_km"], use_miles)
            _load_str = f"\nLoad: {r['load_hr']:.0f} TRIMP" if _has_load and pd.notna(r.get("load_hr")) and r.get("load_hr", 0) > 0 else ""
            return f"{d_str}<br>{dist_str} \u00b7 {int(r['n_runs'])} run{'s' if r['n_runs'] != 1 else ''}{tag}{_load_str}"
        return f"{d_str}<br>Rest"

    cal["hover"] = cal.apply(_hover, axis=1)

    # Color: type-based hue, brightness scaled by TRIMP load within each type band.
    # z = type_num (integer) + load_fraction (0..0.99): higher load = more saturated colour.
    _type_to_num = {"rest": 0, "General": 1, "Easy": 2, "Long Run": 3, "Tempo": 4, "Workout": 5, "Race": 6}

    # Load fraction (0 = low load, 0.99 = peak load within the selected window)
    if _has_load:
        if "load_hr" not in cal.columns:
            cal["load_hr"] = 0.0
        cal["load_hr"] = cal["load_hr"].fillna(0)
        _p95 = cal["load_hr"].quantile(0.95)
        if _p95 > 0:
            cal["load_frac"] = (cal["load_hr"] / _p95).clip(0.15, 0.99)
        else:
            cal["load_frac"] = 0.6
    else:
        cal["load_frac"] = 0.7  # fixed brightness when no load data

    # Fractional part only applies on active days; rest days stay at 0
    cal["load_frac"] = cal["load_frac"].where(cal["distance_km"] > 0, 0.0)

    if has_type:
        cal["z_val"] = cal["run_type"].map(_type_to_num).fillna(0) + cal["load_frac"]
    else:
        # Fallback: pure distance-based brightness (no type colours)
        _max_d = cal["distance_km"].quantile(0.95)
        cal["z_val"] = (cal["distance_km"] / max(_max_d, 1)).clip(0, 6)

    z    = cal.pivot(index="dow", columns="week_col", values="z_val").values
    text = cal.pivot(index="dow", columns="week_col", values="hover").values

    # Continuous-within-band colorscale: each type band fades from dim → saturated.
    # zmax = 6.99 so each integer band spans exactly 1.0 unit.
    # Base (saturated) and dim (low load) colours per type:
    base_colors = ["#111111", "#aec7e8", "#6baed6", "#2ca02c", "#fd8d3c", "#d62728", "#9467bd"]
    dim_colors  = ["#111111", "#1e2c3a", "#0d2230", "#082b08", "#3b2000", "#2d0707", "#1a0d2e"]
    zmax = 6.99
    discrete_cs = [[0.0, base_colors[0]], [0.99 / zmax, base_colors[0]]]  # rest band: always dark
    for i in range(1, 7):
        z_lo = (i + 0.0) / zmax
        z_hi = (i + 0.99) / zmax
        discrete_cs.append([z_lo,  dim_colors[i]])
        discrete_cs.append([z_hi,  base_colors[i]])
    discrete_cs.append([1.0, base_colors[-1]])

    week_starts = cal.groupby("week_col")["date"].min().sort_index()
    x_labels, prev_month = [], None
    for d in week_starts:
        if d.month != prev_month:
            x_labels.append(d.strftime("%b"))
            prev_month = d.month
        else:
            x_labels.append("")

    fig = go.Figure(go.Heatmap(
        z=z, text=text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=discrete_cs,
        zmin=0, zmax=zmax,
        showscale=False, xgap=3, ygap=3,
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=40, r=10, t=24, b=10),
        xaxis=dict(tickvals=list(range(len(x_labels))), ticktext=x_labels, side="top",
                   showgrid=False, zeroline=False),
        yaxis=dict(tickvals=[0, 1, 2, 3, 4, 5, 6],
                   ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                   autorange="reversed", showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# -----------------------------
# Personalized insight banner
# -----------------------------
def generate_insight(daily_full: pd.DataFrame, activities: pd.DataFrame) -> Tuple[str, str]:
    """Returns (message, level) where level is 'success' | 'warning' | 'info'."""
    if len(daily_full) == 0 or len(activities) == 0:
        return "Load your Strava data to see personalized insights here.", "info"

    latest = daily_full.sort_values("date_ts").iloc[-1]
    tsb        = float(latest["tsb"])  if pd.notna(latest.get("tsb"))  else 0.0
    acwr       = float(latest["acwr"]) if pd.notna(latest.get("acwr")) else 1.0
    days_since = max(0, (pd.Timestamp.now() - activities["start_dt_local"].max()).days)

    # Load / fatigue signals -- highest priority
    if acwr > 1.5:
        return (f"Load is elevated (ACWR {acwr:.2f}). An easy or rest day now will let this training adaptation land.", "warning")
    if acwr > 1.3:
        return (f"ACWR slightly high ({acwr:.2f}) \u2014 building fast. Prioritise sleep and nutrition this week.", "warning")
    if tsb < -75:
        return (f"Heavy fatigue (TSB {tsb:.0f}). A 5\u20137 day recovery block will unlock your next fitness jump.", "warning")

    # Run type balance signal (last 4 weeks)
    if "run_type" in activities.columns:
        recent_acts = activities[
            activities["start_dt_local"] >= (pd.Timestamp.now() - pd.Timedelta(weeks=4))
        ]
        if len(recent_acts) >= 4:
            type_counts = recent_acts["run_type"].value_counts()
            hard_count  = sum(type_counts.get(t, 0) for t in ["Tempo", "Workout", "Race"])
            easy_count  = sum(type_counts.get(t, 0) for t in ["Easy", "Long Run", "General"])
            total       = len(recent_acts)
            hard_pct    = hard_count / total * 100
            if hard_pct > 40:
                return (f"{hard_pct:.0f}% of your last 4 weeks were quality sessions. "
                        "Add more easy runs to protect adaptation \u2014 aim for 80% easy.", "warning")
            if hard_count == 0 and total >= 5:
                return ("No tempo or workout runs in the last 4 weeks. "
                        "A quality session this week will help build race-specific fitness.", "info")

    # Freshness signals
    if tsb > 50:
        return (f"You're fresh and race-ready (TSB +{tsb:.0f}). A quality session or tune-up race will use this well.", "success")
    if tsb > 10:
        return f"Good form (TSB +{tsb:.0f}). Training hard today will be well absorbed.", "success"

    # Inactivity signals
    if days_since >= 7:
        return f"No run in {days_since} days. An easy aerobic session will restart the adaptation signal.", "info"
    if days_since >= 3:
        return f"{days_since} days since your last run \u2014 a steady aerobic session today keeps momentum going.", "info"

    return "Training is consistent. Keep building.", "success"


# -----------------------------
# Cadence analysis
# -----------------------------
def compute_cadence_stats(df_range: pd.DataFrame, streams_by_id: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, row in df_range.iterrows():
        aid = int(row["id"])
        streams = streams_by_id.get(aid, {})
        cad_obj = streams.get("cadence", {})
        if isinstance(cad_obj, dict) and isinstance(cad_obj.get("data"), list):
            cad = np.array(cad_obj["data"], dtype=float)
            cad = cad[np.isfinite(cad) & (cad > 50)] * 2  # single-leg -> total SPM
            if len(cad) > 30:
                rows.append({
                    "date": row["start_dt_local"],
                    "name": str(row.get("name", "")),
                    "distance_km": float(row["distance_km"]),
                    "avg_cadence": float(np.mean(cad)),
                    "pct_above_170": float(np.mean(cad >= 170) * 100),
                    "pct_above_180": float(np.mean(cad >= 180) * 100),
                })
    return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()


def compute_risk_penalty(daily: pd.DataFrame) -> float:
    """
    Small penalty based on latest daily risk score (portfolio-friendly heuristic).
    0-55 => no penalty; 55-100 => up to ~6% slower.
    """
    if daily is None or len(daily) == 0 or "risk_score" not in daily.columns:
        return 1.0
    latest = daily.sort_values("date_ts").iloc[-1]
    rs = float(latest.get("risk_score", np.nan))
    if not np.isfinite(rs):
        return 1.0
    if rs <= 55:
        return 1.0
    return float(1.0 + min(0.06, (rs - 55.0) / 45.0 * 0.06))


# -----------------------------
# PMC forward projection
# -----------------------------
def forward_project_pmc(
    daily: pd.DataFrame,
    race_date: pd.Timestamp,
    taper_weeks: int,
    pre_taper_daily_load: float,
    load_build_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Project CTL / ATL / TSB forward to race_date under assumed training + taper.

    Parameters
    ----------
    daily               : historical daily aggregates — seeds the initial CTL and ATL.
    race_date           : target race date.
    taper_weeks         : weeks of taper immediately before race_date.
    pre_taper_daily_load: assumed average daily TRIMP load during the build phase.
    load_build_pct      : weekly % increase during the build phase (0 = maintain,
                          5 = +5 %/week, etc.). Capped at 1.5× to stay conservative.

    Returns a DataFrame with columns: date, ctl, atl, tsb, acwr, daily_load.
    Returns an empty DataFrame if race_date is not in the future or daily is empty.

    Science note — EWMA time constants match Banister (1975):
        CTL alpha = 2 / (28 + 1) ≈ 0.069  (28-day fitness)
        ATL alpha = 2 / (7 + 1)  = 0.25   (7-day fatigue)
    Taper modelled as exponential volume reduction: load × 0.6^(taper_week + 1),
    which yields ≈40 % reduction per week — consistent with Bosquet et al. (2007).
    """
    if len(daily) == 0:
        return pd.DataFrame()

    last = daily.sort_values("date_ts").iloc[-1]
    ctl = float(last.get("chronic_load", 0) or 0)
    atl = float(last.get("acute_load", 0) or 0)

    today = pd.Timestamp.now().normalize()
    race_date = pd.Timestamp(race_date).normalize()
    if race_date <= today:
        return pd.DataFrame()

    alpha_ctl = 2.0 / (28.0 + 1.0)
    alpha_atl = 2.0 / (7.0 + 1.0)
    taper_start = race_date - pd.Timedelta(weeks=int(taper_weeks))

    rows = []
    current = today + pd.Timedelta(days=1)
    while current <= race_date:
        week_num = (current - today).days // 7
        if current >= taper_start:
            taper_week = (current - taper_start).days // 7
            daily_load = pre_taper_daily_load * (0.6 ** (taper_week + 1))
        else:
            build_factor = min(1.5, (1.0 + load_build_pct / 100.0) ** week_num)
            daily_load = pre_taper_daily_load * build_factor

        ctl = ctl * (1.0 - alpha_ctl) + daily_load * alpha_ctl
        atl = atl * (1.0 - alpha_atl) + daily_load * alpha_atl
        tsb = ctl - atl
        acwr = atl / ctl if ctl > 1e-6 else np.nan

        rows.append({
            "date": current,
            "ctl": round(ctl, 2),
            "atl": round(atl, 2),
            "tsb": round(tsb, 2),
            "acwr": round(float(acwr), 3) if np.isfinite(acwr) else np.nan,
            "daily_load": round(daily_load, 2),
        })
        current += pd.Timedelta(days=1)

    return pd.DataFrame(rows)


# -----------------------------
# Gear stats helper
# -----------------------------
def compute_gear_stats(activities: pd.DataFrame, gear_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Compute total km per gear_id from all activities."""
    if "gear_id" not in activities.columns:
        return pd.DataFrame()
    d = activities[pd.notna(activities["gear_id"]) & (activities["gear_id"] != "")].copy()
    if len(d) == 0:
        return pd.DataFrame()
    d["gear_id"] = d["gear_id"].astype(str)
    stats = (
        d.groupby("gear_id")
        .agg(
            total_km=("distance_km", "sum"),
            runs=("id", "count"),
            last_used=("start_dt_local", "max"),
            first_used=("start_dt_local", "min"),
        )
        .reset_index()
    )
    if gear_names:
        stats["name"] = stats["gear_id"].map(lambda g: gear_names.get(g, g))
    else:
        stats["name"] = stats["gear_id"]
    stats = stats.sort_values("total_km", ascending=False)
    return stats
