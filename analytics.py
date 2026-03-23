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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start, end = date_range
    # Use ALL activities for EWMA so chronic load (28d) has proper history even when
    # the selected date range is short. We filter the returned data to date_range at the end.
    df = activities.copy()

    df["hr_intensity"] = df["avg_hr"] / float(max_hr)
    df["load_hr"] = df["duration_min"] * df["hr_intensity"]

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

    half = total_time * 0.5
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
    """
    d = df_range.copy()
    d = d[pd.notna(d["avg_hr"]) & pd.notna(d["avg_speed_mps"])].copy()
    if len(d) == 0:
        return pd.DataFrame()

    d["hr_intensity"] = d["avg_hr"] / float(max_hr)
    d["speed_per_hr"] = d["avg_speed_mps"] / d["avg_hr"]
    d = d.sort_values("start_dt_local").reset_index(drop=True)

    # Rolling 20th percentile as baseline
    def roll_q(x: pd.Series, q: float) -> pd.Series:
        return x.rolling(window=12, min_periods=6).quantile(q)

    d["eff_q20"] = roll_q(d["speed_per_hr"], 0.20)
    d["compromised"] = (d["speed_per_hr"] < d["eff_q20"]).astype(int)
    d["eff_delta"] = d["speed_per_hr"] - d["eff_q20"]

    return d


# -----------------------------
# Tab 5 computations (race prediction)
# -----------------------------
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
    max_ratio: float = 4.0,
    top_n: int = 5,
) -> Tuple[Optional[float], Optional[pd.Series]]:
    """
    Predict race time using the Riegel power-law model:
        T2 = T1 × (D2 / D1)^exponent

    Professional-grade implementation:
    - min_km / max_km filter removes very short or unrealistically long runs.
    - max_ratio cap: source distance must be ≥ target_km / max_ratio to limit
      extrapolation error (e.g. no predicting a marathon from a 3 km run).
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
def compute_personal_bests(activities: pd.DataFrame) -> dict:
    df = activities[pd.notna(activities["distance_km"]) & pd.notna(activities["pace_min_per_km"])].copy()
    df = df[df["duration_min"] > 3].copy()
    bests: dict = {}
    ranges = {
        "best_5k":      (4.0,  7.0),
        "best_10k":     (8.0,  12.0),
        "best_hm":      (18.0, 23.0),
        "best_marathon":(38.0, 44.0),
    }
    for key, (lo_km, hi_km) in ranges.items():
        sub = df[(df["distance_km"] >= lo_km) & (df["distance_km"] <= hi_km)]
        if len(sub) > 0:
            idx = sub["pace_min_per_km"].idxmin()
            r = sub.loc[idx]
            bests[key] = {
                "pace_min_per_km": float(r["pace_min_per_km"]),
                "distance_km": float(r["distance_km"]),
                "date": r["start_dt_local"],
            }
    if len(df) > 0:
        idx = df["distance_km"].idxmax()
        r = df.loc[idx]
        bests["longest_run"] = {
            "distance_km": float(r["distance_km"]),
            "date": r["start_dt_local"],
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
# VO2max estimate (Jack Daniels VDOT)
# -----------------------------
def estimate_vo2max(bests: dict) -> Optional[float]:
    """Estimate VDOT from best recorded efforts using Jack Daniels' formula."""
    priority = [("best_marathon", 42.195), ("best_hm", 21.0975), ("best_10k", 10.0), ("best_5k", 5.0)]
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

    daily = acts.groupby("date").agg(
        distance_km=("distance_km", "sum"),
        n_runs=("id", "count"),
    ).reset_index()

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
            return f"{d_str}<br>{dist_str} \u00b7 {int(r['n_runs'])} run{'s' if r['n_runs'] != 1 else ''}{tag}"
        return f"{d_str}<br>Rest"

    cal["hover"] = cal.apply(_hover, axis=1)

    # Color: type-based when run, dark background when rest
    _type_to_num = {"rest": 0, "General": 1, "Easy": 2, "Long Run": 3, "Tempo": 4, "Workout": 5, "Race": 6}
    if has_type:
        cal["z_val"] = cal["run_type"].map(_type_to_num).fillna(0)
    else:
        cal["z_val"] = cal["distance_km"].clip(upper=30)

    z    = cal.pivot(index="dow", columns="week_col", values="z_val").values
    text = cal.pivot(index="dow", columns="week_col", values="hover").values

    # Discrete colorscale: 0=rest, 1=General, 2=Easy, 3=Long Run, 4=Tempo, 5=Workout, 6=Race
    # zmin=0, zmax=6 -> z=i maps to i/6 in [0,1]. Build hard step transitions.
    colors = ["#111111", "#aec7e8", "#6baed6", "#2ca02c", "#fd8d3c", "#d62728", "#9467bd"]
    n = len(colors)  # 7 values spanning z=0..6
    discrete_cs = [[0.0, colors[0]]]
    for i in range(1, n):
        t = i / (n - 1)
        discrete_cs.append([t - 1e-9, colors[i - 1]])  # hold previous colour right up to boundary
        discrete_cs.append([t, colors[i]])               # snap to new colour
    discrete_cs.append([1.0, colors[-1]])

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
        zmin=0, zmax=6,
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
    if tsb < -30:
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
    if tsb > 20:
        return (f"You're fresh and race-ready (TSB +{tsb:.0f}). A quality session or tune-up race will use this well.", "success")
    if tsb > 5:
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
