import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Allure",
    page_icon="🏃",
    layout="wide",
)

# ── Cookie controller (must come before any other st calls) ───────────
# Don't instantiate CookieController during OAuth callback — its JS rerun
# interrupts the token exchange mid-execution (RerunException before tokens are saved).
_in_oauth_callback = "code" in st.query_params
try:
    from streamlit_cookies_controller import CookieController as _CookieController
    _cookies = None if _in_oauth_callback else _CookieController()
    _COOKIES_ENABLED = not _in_oauth_callback
except Exception:
    _cookies = None
    _COOKIES_ENABLED = False

ACTIVITIES_PATH = "data/strava_runs_detailed.json"
STREAMS_PATH = "data/strava_runs_streams.json"
STRAVA_CACHE_PATH = "data/strava_oauth_cache.json"

RACE_PRESETS_KM: Dict[str, float] = {
    "5K": 5.0,
    "10K": 10.0,
    "Half Marathon": 21.0975,
    "Marathon": 42.195,
    "Custom (km)": np.nan,
}

# Defaults for "race effort" heart-rate band (as % of max HR)
RACE_EFFORT_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "5K": (0.90, 0.98),
    "10K": (0.88, 0.95),
    "Half Marathon": (0.84, 0.92),
    "Marathon": (0.78, 0.88),
    "Custom (km)": (0.84, 0.92),
}

# Default long-run threshold as % of race distance
LONG_RUN_DEFAULTS: Dict[str, float] = {
    "5K": 0.70,
    "10K": 0.70,
    "Half Marathon": 0.60,
    "Marathon": 0.55,
    "Custom (km)": 0.60,
}

WORKOUT_TYPE_MAP = {
    0: "Run",
    1: "Race",
    2: "Long Run",
    3: "Workout",
    None: "Unspecified",
}

# -----------------------------
# Run type classification
# -----------------------------
RUN_TYPE_COLORS: Dict[str, str] = {
    "Easy":     "#6baed6",  # blue
    "Long Run": "#2ca02c",  # green
    "General":  "#aec7e8",  # light blue-grey
    "Tempo":    "#fd8d3c",  # orange
    "Workout":  "#d62728",  # red
    "Race":     "#9467bd",  # purple
}

# Priority for "dominant type" per day (highest wins)
RUN_TYPE_PRIORITY: Dict[str, int] = {
    "General": 0, "Easy": 1, "Long Run": 2, "Tempo": 3, "Workout": 4, "Race": 5,
}

_EASY_KW     = {"easy", "recovery", "jog", "slow", "base", "aerobic", "ez", "shakeout", "recover"}
_WORKOUT_KW  = {"interval", "intervals", "tempo", "fartlek", "rep", "reps", "workout",
                "speed", "track", "progression", "threshold", "vo2", "hills", "strides",
                "quality", "hard", "fast", "sprint", "lactate", "cruise"}
_LONG_KW     = {"long", "lsd", "endurance", "long run"}
_RACE_KW     = {"race", "marathon", "parkrun", "10k", "5k", "half"}


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
# Loaders
# -----------------------------
def parse_activities_raw(raw: list) -> pd.DataFrame:
    """Convert a list of raw Strava activity dicts into the processed DataFrame."""
    df = pd.json_normalize(raw)

    keep = [
        "id",
        "name",
        "sport_type",
        "type",
        "start_date_local",
        "distance",
        "moving_time",
        "elapsed_time",
        "total_elevation_gain",
        "workout_type",
        "has_heartrate",
        "average_heartrate",
        "max_heartrate",
        "average_speed",
        "max_speed",
        "device_name",
        "manual",
        "trainer",
        "gear_id",
        "start_latlng",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    df["start_dt_local"] = pd.to_datetime(df["start_date_local"], errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["start_dt_local"])
    df["date"] = df["start_dt_local"].dt.date

    df["distance_km"] = df["distance"] / 1000.0
    df["duration_min"] = df["moving_time"] / 60.0

    df["avg_hr"] = pd.to_numeric(df["average_heartrate"], errors="coerce")
    df["max_hr_activity"] = pd.to_numeric(df["max_heartrate"], errors="coerce")
    df["avg_speed_mps"] = pd.to_numeric(df["average_speed"], errors="coerce")
    df["max_speed_mps"] = pd.to_numeric(df["max_speed"], errors="coerce")

    df["pace_sec_per_km"] = np.where(df["avg_speed_mps"] > 0, 1000.0 / df["avg_speed_mps"], np.nan)
    df["pace_min_per_km"] = df["pace_sec_per_km"] / 60.0

    def workout_label(x):
        if pd.isna(x):
            return "Unspecified"
        try:
            return WORKOUT_TYPE_MAP.get(int(x), "Unspecified")
        except Exception:
            return "Unspecified"

    df["workout_label"] = df["workout_type"].apply(workout_label)

    # Extract start lat/lng from start_latlng array field
    df["start_lat"] = df["start_latlng"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) >= 2 else np.nan
    )
    df["start_lng"] = df["start_latlng"].apply(
        lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else np.nan
    )

    return df


@st.cache_data(show_spinner=False)
def load_activities(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)
    return parse_activities_raw(raw)


@st.cache_data(show_spinner=False)
def load_streams(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r") as f:
        raw = json.load(f)

    out: Dict[int, Dict[str, Any]] = {}
    for item in raw:
        aid = item.get("activity_id")
        streams = item.get("streams", {})
        if isinstance(aid, int) and isinstance(streams, dict):
            out[aid] = streams
    return out

def streams_to_df(streams_obj: dict) -> pd.DataFrame:
    """Convert a single activity 'streams' dict into a tidy dataframe."""
    cols = {}
    for k, v in (streams_obj or {}).items():
        if isinstance(v, dict) and "data" in v:
            cols[k] = v["data"]

    df = pd.DataFrame(cols)

    # latlng is a list of [lat, lng] pairs -> split into two columns if present
    if "latlng" in df.columns:
        latlng = df["latlng"]
        df["lat"] = latlng.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) == 2 else np.nan)
        df["lng"] = latlng.apply(lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) == 2 else np.nan)
        df = df.drop(columns=["latlng"])

    # Derived helpers
    if "distance" in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance"], errors="coerce") / 1000.0

    if "velocity_smooth" in df.columns:
        v = pd.to_numeric(df["velocity_smooth"], errors="coerce")

        v = v.where(v > 0)

        df["velocity_smooth"] = v
        df["pace_min_per_km"] = (1000.0 / v) / 60.0

        # Drop inf/-inf just in case
        df["pace_min_per_km"] = df["pace_min_per_km"].replace([np.inf, -np.inf], np.nan)

        # If pace higher than 20, set to >20
        df["pace_min_per_km"] = df["pace_min_per_km"].where(df["pace_min_per_km"] < 20, 20)

    return df

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
        return ("N/A", "⚪️")
    if acwr < 0.8:
        return ("Low", "🔵")
    if acwr <= 1.3:
        return ("Good", "🟢")
    if acwr <= 1.7:
        return ("High", "🟠")
    return ("Very high", "🔴")


KM_TO_MILES = 0.621371


def _format_pace(min_per_km: float, use_miles: bool = False) -> str:
    """Format pace as M:SS/km or M:SS/mi."""
    if min_per_km is None or (isinstance(min_per_km, float) and np.isnan(min_per_km)):
        return "—"
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

    # Composite daily risk score (0-100-ish):
    # - High ACWR increases risk
    # - Few rest days increases risk
    # - High acute load relative to own history increases risk
    acwr = d["acwr"].copy()
    acwr_clipped = acwr.clip(lower=0.0, upper=3.0)
    acwr_component = (acwr_clipped - 1.0).clip(lower=0.0)  # only penalize > 1.0
    acute_component = zscore(d["acute_load"]).clip(lower=0.0)  # only penalize above mean
    rest_component = (3.0 - d["rest_days_last7"]).clip(lower=0.0) / 3.0  # penalize <3 rest days/week

    # Weighted sum of three components (weights sum to 1.0).
    # Divisors normalize each component to a ~0-1 scale:
    #   acwr_component  peaks around 1.0 when ACWR ≈ 2.0  (/ 1.0 = identity, already 0-1 after clip)
    #   acute_component is a z-score, clipped positive;  / 2.0 scales a 2-sigma spike to ~1.0
    #   rest_component  is already 0-1 by construction
    d["risk_score"] = 100.0 * (0.45 * acwr_component.fillna(0.0) / 1.0 + 0.35 * acute_component.fillna(0.0) / 2.0 + 0.20 * rest_component.fillna(0.0))
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
        return "—"
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
    min_km: float = 2.0,
    max_km: float = 1000.0,
) -> Tuple[Optional[float], Optional[pd.Series]]:
    """
    Predict race time using a Riegel-style power law:
        T2 = T1 * (D2 / D1)^exponent
    We take the BEST (minimum) predicted time among eligible runs as a performance-based estimate.
    Returns (pred_seconds, source_run_row).
    """
    if runs is None or len(runs) == 0 or target_km <= 0:
        return None, None

    df = runs.copy()
    df = df[pd.notna(df["distance_km"]) & pd.notna(df["duration_min"])].copy()
    df = df[(df["distance_km"] >= min_km) & (df["distance_km"] <= max_km) & (df["duration_min"] > 3)].copy()
    if len(df) == 0:
        return None, None

    df["time_sec"] = df["duration_min"] * 60.0
    df["pred_sec"] = df["time_sec"] * (float(target_km) / df["distance_km"]) ** float(exponent)

    best_idx = df["pred_sec"].idxmin()
    best_row = df.loc[best_idx]
    return float(best_row["pred_sec"]), best_row


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
_DEFAULT_HR_ZONES = [
    ("Z1 Recovery",  0.00, 0.60),
    ("Z2 Aerobic",   0.60, 0.75),
    ("Z3 Tempo",     0.75, 0.87),
    ("Z4 Threshold", 0.87, 0.93),
    ("Z5 VO₂max",   0.93, 9.99),
]


def make_hr_zones(z1_max: float, z2_max: float, z3_max: float, z4_max: float):
    """Build HR zone list from user-defined boundaries (as fractions of max HR)."""
    return [
        ("Z1 Recovery",  0.00,  z1_max),
        ("Z2 Aerobic",   z1_max, z2_max),
        ("Z3 Tempo",     z2_max, z3_max),
        ("Z4 Threshold", z3_max, z4_max),
        ("Z5 VO₂max",   z4_max, 9.99),
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

    # Consecutive week streak (≥1 run per week)
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
def build_calendar_heatmap(activities: pd.DataFrame, n_weeks: int = 53, use_miles: bool = False) -> go.Figure:
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
            tag = f" · {r['run_type']}" if has_type and r["run_type"] != "rest" else ""
            dist_str = _dist_fmt(r["distance_km"], use_miles)
            return f"{d_str}<br>{dist_str} · {int(r['n_runs'])} run{'s' if r['n_runs'] != 1 else ''}{tag}"
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
    # zmin=0, zmax=6 → z=i maps to i/6 in [0,1]. Build hard step transitions.
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

    # Load / fatigue signals — highest priority
    if acwr > 1.5:
        return (f"Load is elevated (ACWR {acwr:.2f}). An easy or rest day now will let this training adaptation land.", "warning")
    if acwr > 1.3:
        return (f"ACWR slightly high ({acwr:.2f}) — building fast. Prioritise sleep and nutrition this week.", "warning")
    if tsb < -30:
        return (f"Heavy fatigue (TSB {tsb:.0f}). A 5–7 day recovery block will unlock your next fitness jump.", "warning")

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
                        "Add more easy runs to protect adaptation — aim for 80% easy.", "warning")
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
        return f"{days_since} days since your last run — a steady aerobic session today keeps momentum going.", "info"

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
            cad = cad[np.isfinite(cad) & (cad > 50)] * 2  # single-leg → total SPM
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
# Strava OAuth helpers
# -----------------------------
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"


def get_strava_auth_url(client_id: str, redirect_uri: str) -> str:
    from urllib.parse import urlencode
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": "read,activity:read_all",
    }
    return STRAVA_AUTH_URL + "?" + urlencode(params)


def exchange_strava_code(client_id, client_secret, code, redirect_uri: str = "") -> dict:
    resp = requests.post(STRAVA_TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "code": code, "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    })
    return resp.json()


def refresh_strava_token(client_id, client_secret, refresh_token) -> dict:
    resp = requests.post(STRAVA_TOKEN_URL, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    })
    return resp.json()


def get_valid_token(client_id, client_secret) -> Optional[str]:
    """Return a valid access token, refreshing if needed. Returns None if not authenticated."""
    tokens = st.session_state.get("strava_tokens")
    if not tokens or "access_token" not in tokens:
        return None
    expires_at = tokens.get("expires_at", 0)
    # If expires_at is 0 or missing, trust the token we just received (don't refresh)
    if expires_at and time.time() > expires_at - 300:
        new_tokens = refresh_strava_token(client_id, client_secret, tokens["refresh_token"])
        if "access_token" in new_tokens:
            st.session_state["strava_tokens"] = new_tokens
            return new_tokens["access_token"]
        # Refresh failed but we still have access_token — try it anyway
        return tokens.get("access_token")
    return tokens["access_token"]


def _strava_get(url: str, access_token: str, params: dict = None, max_retries: int = 4) -> requests.Response:
    """GET with exponential back-off on 429 (rate limit) and transient 5xx errors."""
    headers = {"Authorization": f"Bearer {access_token}"}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            wait = min(wait, 60)
            time.sleep(wait)
            continue
        if resp.status_code >= 500 and attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            continue
        return resp
    return resp  # return last response even if still failing


def fetch_all_activities_api(access_token: str) -> list:
    """Fetch all activities from Strava API (paginated, rate-limit safe)."""
    all_acts = []
    page = 1
    while True:
        resp = _strava_get(
            f"{STRAVA_API_BASE}/athlete/activities", access_token,
            params={"per_page": 200, "page": page},
        )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_acts.extend(batch)
        if len(batch) < 200:
            break
        page += 1
    return all_acts


def fetch_activity_streams_api(activity_id: int, access_token: str) -> dict:
    """Fetch streams for a single activity (rate-limit safe)."""
    resp = _strava_get(
        f"{STRAVA_API_BASE}/activities/{activity_id}/streams", access_token,
        params={"keys": "time,heartrate,velocity_smooth,cadence,altitude,latlng", "key_by_type": "true"},
    )
    return resp.json() if resp.status_code == 200 else {}


def fetch_gear_api(gear_id: str, access_token: str) -> dict:
    """Fetch gear (shoe) details."""
    resp = requests.get(
        f"{STRAVA_API_BASE}/gear/{gear_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    return resp.json() if resp.status_code == 200 else {}


# ── Disk cache helpers ────────────────────────────────────────────────
def load_strava_disk_cache() -> Optional[Dict]:
    """Load cached activities + streams from disk. Returns None if missing."""
    if not os.path.exists(STRAVA_CACHE_PATH):
        return None
    try:
        with open(STRAVA_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_strava_disk_cache(activities_raw: list, streams_dict: Dict) -> None:
    """Persist activities and streams to disk so they survive session reloads."""
    os.makedirs(os.path.dirname(STRAVA_CACHE_PATH), exist_ok=True)
    payload = {
        "fetched_at": datetime.utcnow().isoformat(),
        "activities": activities_raw,
        "streams": {str(k): v for k, v in streams_dict.items()},
    }
    with open(STRAVA_CACHE_PATH, "w") as f:
        json.dump(payload, f)


# ── Supabase helpers ──────────────────────────────────────────────────
# Run in Supabase SQL editor:
# CREATE TABLE athletes (athlete_id BIGINT PRIMARY KEY, display_name TEXT, refresh_token TEXT, fetched_at TIMESTAMPTZ DEFAULT NOW(), preferences JSONB DEFAULT '{}');
# If you already created the table: ALTER TABLE athletes ADD COLUMN IF NOT EXISTS preferences JSONB DEFAULT '{}';
# CREATE TABLE activities (athlete_id BIGINT, activity_id BIGINT, data JSONB, PRIMARY KEY (athlete_id, activity_id));
# CREATE TABLE streams (athlete_id BIGINT, activity_id BIGINT, data JSONB, PRIMARY KEY (athlete_id, activity_id));
# CREATE INDEX idx_activities_athlete ON activities(athlete_id);
# CREATE INDEX idx_streams_athlete ON streams(athlete_id);
_SUPABASE_ENABLED = False
_SUPABASE_ERROR = ""
try:
    from supabase import create_client as _sb_create_client
    _sb_url = st.secrets.get("supabase", {}).get("url", "")
    _sb_key = st.secrets.get("supabase", {}).get("key", "")
    _SUPABASE_ENABLED = bool(_sb_url and _sb_key)
    if not _sb_url:
        _SUPABASE_ERROR = "Missing supabase.url in secrets"
    elif not _sb_key:
        _SUPABASE_ERROR = "Missing supabase.key in secrets"
except ImportError:
    _SUPABASE_ERROR = "supabase package not installed"
except Exception as _e:
    _SUPABASE_ERROR = str(_e)


@st.cache_resource
def _get_supabase():
    from supabase import create_client
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])


def sb_save_athlete(athlete_id: int, display_name: str, refresh_token: str) -> Optional[str]:
    """Returns error string on failure, None on success."""
    try:
        _get_supabase().table("athletes").upsert({
            "athlete_id": athlete_id,
            "display_name": display_name,
            "refresh_token": refresh_token,
            "fetched_at": datetime.utcnow().isoformat(),
        }).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_athlete(athlete_id: int) -> Optional[dict]:
    try:
        resp = _get_supabase().table("athletes").select("*").eq("athlete_id", athlete_id).maybe_single().execute()
        return resp.data
    except Exception:
        return None


def sb_save_activities(athlete_id: int, activities_raw: list) -> Optional[str]:
    """Returns error string on failure, None on success."""
    try:
        rows = [{"athlete_id": athlete_id, "activity_id": int(a["id"]), "data": a} for a in activities_raw]
        for i in range(0, len(rows), 500):
            _get_supabase().table("activities").upsert(rows[i:i+500]).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_activities(athlete_id: int) -> Optional[list]:
    try:
        resp = _get_supabase().table("activities").select("data").eq("athlete_id", athlete_id).execute()
        if resp.data:
            return [row["data"] for row in resp.data]
    except Exception:
        pass
    return None


def sb_save_preferences(athlete_id: int, prefs: dict) -> Optional[str]:
    """Save sidebar preferences to athletes table. Returns error string or None."""
    try:
        _get_supabase().table("athletes").update({"preferences": prefs}).eq("athlete_id", athlete_id).execute()
        return None
    except Exception as e:
        return str(e)


def sb_load_preferences(athlete_id: int) -> Optional[dict]:
    """Load saved sidebar preferences. Returns dict or None."""
    try:
        resp = (_get_supabase().table("athletes")
                .select("preferences").eq("athlete_id", athlete_id)
                .maybe_single().execute())
        if resp.data and resp.data.get("preferences"):
            return resp.data["preferences"]
    except Exception:
        pass
    return None


def sb_save_streams(athlete_id: int, streams: Dict[int, dict]) -> None:
    try:
        rows = [{"athlete_id": athlete_id, "activity_id": int(aid), "data": data}
                for aid, data in streams.items()]
        for i in range(0, len(rows), 200):
            _get_supabase().table("streams").upsert(rows[i:i+200]).execute()
    except Exception:
        pass


def sb_load_streams(athlete_id: int) -> Dict[int, dict]:
    try:
        resp = _get_supabase().table("streams").select("activity_id,data").eq("athlete_id", athlete_id).execute()
        return {int(row["activity_id"]): row["data"] for row in (resp.data or [])}
    except Exception:
        return {}


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


# -----------------------------
# Weather helper
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600 * 24)
def fetch_weather_for_activities(activities_json: str) -> pd.DataFrame:
    """
    Fetch historical temperature for activities using Open-Meteo archive API.
    activities_json: JSON string of list of dicts with keys: id, date, start_lat, start_lng, start_dt_local
    Returns DataFrame with columns: id, temp_c
    """
    activities_list = json.loads(activities_json)

    # Group by (lat_rounded, lng_rounded, month) to batch requests
    groups: Dict = defaultdict(list)
    for a in activities_list:
        if pd.isna(a.get("start_lat")) or pd.isna(a.get("start_lng")):
            continue
        lat = round(float(a["start_lat"]), 1)
        lng = round(float(a["start_lng"]), 1)
        date_str = str(a["date"])[:10]
        groups[(lat, lng, date_str[:7])].append(a)  # group by month

    results: Dict = {}  # id -> {temp_c}

    for (lat, lng, month_str), acts in groups.items():
        dates = sorted(set(str(a["date"])[:10] for a in acts))
        if not dates:
            continue
        start_d = dates[0]
        end_d = dates[-1]
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat, "longitude": lng,
                "start_date": start_d, "end_date": end_d,
                "hourly": "temperature_2m",
                "timezone": "auto",
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            hourly_times = data.get("hourly", {}).get("time", [])
            hourly_temps = data.get("hourly", {}).get("temperature_2m", [])
            # Build lookup: date+hour -> temp
            time_temp: Dict = {}
            for t_str, temp in zip(hourly_times, hourly_temps):
                time_temp[t_str] = temp

            for a in acts:
                dt_str = str(a["start_dt_local"])[:13]  # "2024-03-15T10"
                temp = time_temp.get(dt_str)
                if temp is None:
                    # Try within ±2 hours
                    for h_offset in [1, -1, 2, -2]:
                        try:
                            dt_obj = datetime.fromisoformat(dt_str)
                            alt_str = (dt_obj + timedelta(hours=h_offset)).strftime("%Y-%m-%dT%H")
                            temp = time_temp.get(alt_str)
                            if temp is not None:
                                break
                        except Exception:
                            pass
                if temp is not None:
                    results[a["id"]] = {"temp_c": float(temp)}
            time.sleep(0.05)  # be polite to the API
        except Exception:
            continue

    if not results:
        return pd.DataFrame(columns=["id", "temp_c"])
    return pd.DataFrame([{"id": k, **v} for k, v in results.items()])


# -----------------------------
# Sidebar controls (shared)
# -----------------------------
st.markdown("""
<style>
/* Hide Streamlit branding only */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Tighter content padding */
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1440px;}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 14px 18px;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {gap: 2px; border-bottom: 2px solid rgba(255,255,255,0.08);}
.stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: 8px 18px; font-weight: 500;}
.stTabs [aria-selected="true"] {background: rgba(255,255,255,0.07);}

/* Dividers */
hr {border-color: rgba(255,255,255,0.07) !important;}

/* Section headings */
h3 {margin-top: 0.25rem !important;}
</style>
""", unsafe_allow_html=True)

# ── Strava OAuth setup ────────────────────────────────────────────────
_strava_secrets: Dict = {}
try:
    _strava_secrets = st.secrets.get("strava", {})
except Exception:
    pass
_STRAVA_CLIENT_ID = _strava_secrets.get("client_id", "")
_STRAVA_CLIENT_SECRET = _strava_secrets.get("client_secret", "")
_OAUTH_ENABLED = bool(_STRAVA_CLIENT_ID and _STRAVA_CLIENT_SECRET)

# Handle OAuth callback (code in URL params)
_qp = st.query_params


if "code" in _qp and _OAUTH_ENABLED and "strava_tokens" not in st.session_state:
    _auth_code = _qp["code"]
    if st.session_state.get("_exchanged_code") != _auth_code:
        st.session_state["_exchanged_code"] = _auth_code
        _redirect_uri_cb = _strava_secrets.get("redirect_uri", "http://localhost:8501")
        try:
            _tokens = exchange_strava_code(
                _STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET, _auth_code, _redirect_uri_cb
            )
        except Exception as _exc:
            _tokens = {"error": str(_exc)}
        st.session_state["_last_token_response"] = _tokens
        if "access_token" in _tokens:
            st.session_state["strava_tokens"] = _tokens
            _ath = _tokens.get("athlete", {})
            _ath_id = _ath.get("id")
            if _ath_id:
                st.session_state["strava_athlete_id"] = int(_ath_id)
                st.session_state["strava_athlete_name"] = (
                    f"{_ath.get('firstname', '')} {_ath.get('lastname', '')}".strip()
                )
                # Pre-load saved preferences so sidebar uses them immediately
                if _SUPABASE_ENABLED and "_prefs" not in st.session_state:
                    _prefs_now = sb_load_preferences(int(_ath_id))
                    if _prefs_now:
                        st.session_state["_prefs"] = _prefs_now
            st.query_params.clear()
            st.rerun()
        else:
            st.session_state.pop("_exchanged_code", None)
    else:
        # Stuck: code already attempted but no tokens — show the response
        _last = st.session_state.get("_last_token_response", {})
        st.error(f"Strava auth failed: {_last}")
        if st.button("Try again"):
            st.session_state.pop("_exchanged_code", None)
            st.session_state.pop("_last_token_response", None)
            st.rerun()

# Silent re-auth from cookie (returning users)
if _OAUTH_ENABLED and "strava_tokens" not in st.session_state and _COOKIES_ENABLED and _cookies is not None:
    _cookie_refresh = _cookies.get("strava_refresh_token")
    _cookie_athlete_id = _cookies.get("strava_athlete_id")
    if _cookie_refresh and _cookie_athlete_id:
        _new_tokens = refresh_strava_token(_STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET, _cookie_refresh)
        if "access_token" in _new_tokens:
            st.session_state["strava_tokens"] = _new_tokens
            st.session_state["strava_athlete_id"] = int(_cookie_athlete_id)
            # Fetch athlete name + preferences from Supabase
            if _SUPABASE_ENABLED:
                _ath_row = sb_load_athlete(int(_cookie_athlete_id))
                if _ath_row:
                    st.session_state["strava_athlete_name"] = _ath_row.get("display_name", "")
                    if _ath_row.get("preferences") and "_prefs" not in st.session_state:
                        st.session_state["_prefs"] = _ath_row["preferences"]
                    # Rotate refresh token in Supabase
                    sb_save_athlete(int(_cookie_athlete_id), _ath_row.get("display_name", ""),
                                    _new_tokens["refresh_token"])
            # Rotate refresh token in cookie (no Secure flag — works on HTTP + HTTPS)
            if _cookies is not None:
                _cookies.set("strava_refresh_token", _new_tokens["refresh_token"])
                _cookies.set("strava_athlete_id", str(_cookie_athlete_id))
            st.rerun()
        else:
            # Cookie is stale — clear it
            _cookies.remove("strava_refresh_token")
            _cookies.remove("strava_athlete_id")

st.title("🏃 Endurance Analytics Dashboard")

with st.sidebar:
    st.header("Controls")

    # Load saved preferences (populated after first auth)
    _prefs = st.session_state.get("_prefs", {})

    max_hr = st.number_input(
        "Max HR (bpm)",
        min_value=120,
        max_value=230,
        value=int(_prefs.get("max_hr", 180)),
        step=1,
        help="Used to compute intensity = avgHR / maxHR.",
    )

    with st.expander("HR Zone Boundaries (% of max HR)"):
        st.caption("Drag to adjust where each zone begins and ends. These boundaries drive both run classification and the zone breakdown chart.")
        hr_z1 = st.slider("Z1/Z2 boundary", 50, 75, int(_prefs.get("hr_z1", 60)), step=1,
                          help="Below this = Z1 Recovery") / 100.0
        hr_z2 = st.slider("Z2/Z3 boundary (Easy threshold)", 65, 88, int(_prefs.get("hr_z2", 80)), step=1,
                          help="Runs with median HR below this are classified as Easy") / 100.0
        hr_z3 = st.slider("Z3/Z4 boundary (Tempo threshold)", 78, 95, int(_prefs.get("hr_z3", 87)), step=1,
                          help="Runs with median HR above this are classified as Tempo") / 100.0
        hr_z4 = st.slider("Z4/Z5 boundary", 85, 100, int(_prefs.get("hr_z4", 93)), step=1,
                          help="Above this = Z5 VO₂max") / 100.0
        # Clamp to prevent inversions
        hr_z1 = min(hr_z1, hr_z2 - 0.01)
        hr_z3 = max(hr_z3, hr_z2 + 0.01)
        hr_z4 = max(hr_z4, hr_z3 + 0.01)
        _hr_zones = make_hr_zones(hr_z1, hr_z2, hr_z3, hr_z4)

    _race_keys = list(RACE_PRESETS_KM.keys())
    _saved_race = _prefs.get("race_choice", "Half Marathon")
    _race_idx = _race_keys.index(_saved_race) if _saved_race in _race_keys else 2
    race_choice = st.selectbox("Target race distance", _race_keys, index=_race_idx)
    if race_choice == "Custom (km)":
        race_km = st.number_input("Custom race distance (km)", min_value=1.0, max_value=200.0,
                                  value=float(_prefs.get("race_km", 21.0975)), step=0.5)
    else:
        race_km = float(RACE_PRESETS_KM[race_choice])

    lo_def, hi_def = RACE_EFFORT_DEFAULTS.get(race_choice, (0.84, 0.92))
    st.subheader("Race-effort HR band")
    effort_band = st.slider(
        "HR intensity range (% of max HR)",
        min_value=0.50,
        max_value=1.00,
        value=(float(_prefs.get("lo_hr", lo_def)), float(_prefs.get("hi_hr", hi_def))),
        step=0.01,
        help="Used in Tab 2 to track efficiency at race-relevant intensity.",
    )

    lr_def = float(LONG_RUN_DEFAULTS.get(race_choice, 0.60))
    st.subheader("Long-run threshold")
    long_run_ratio_thresh = st.slider(
        "Minimum long-run distance (% of race distance)",
        min_value=0.30,
        max_value=1.00,
        value=float(_prefs.get("long_run_ratio_thresh", lr_def)),
        step=0.05,
        help="Tab 3 considers runs longer than this threshold as 'long runs' for fatigue modeling.",
    )

    st.subheader("Readiness windows")
    readiness_window_days = st.slider(
        "Readiness lookback (days)",
        min_value=14,
        max_value=90,
        value=int(_prefs.get("readiness_window_days", 42)),
        step=7,
        help="Tab 4 uses this window to summarize recent risk/readiness.",
    )

    st.subheader("Prediction window")
    prediction_lookback_days = st.slider(
        "Race prediction lookback (days)",
        min_value=30,
        max_value=365,
        value=int(_prefs.get("prediction_lookback_days", 180)),
        step=15,
        help="Tab 5 uses this window to predict race time.",
    )

    st.divider()
    st.subheader("Filters")
    only_runs = st.checkbox("Only running activities", value=bool(_prefs.get("only_runs", True)))
    exclude_manual = st.checkbox("Exclude manual activities", value=bool(_prefs.get("exclude_manual", True)))
    exclude_trainer = st.checkbox("Exclude trainer/treadmill", value=bool(_prefs.get("exclude_trainer", False)))

    st.divider()
    use_miles = st.toggle("Show distances in miles 🇺🇸", value=bool(_prefs.get("use_miles", False)))
    show_streams_tab = st.checkbox("🔧 Raw streams explorer", value=False,
                                   help="Adds a Raw Streams tab to inspect per-second GPS, HR, and cadence data for individual activities.")
    # Date range is added below after activities are loaded

    # Save Settings button (only shown when authenticated)
    if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
        st.divider()
        if st.button("💾 Save Settings", use_container_width=True,
                     help="Save current sidebar values so they reload next time"):
            _prefs_to_save = {
                "max_hr": int(max_hr),
                "hr_z1": int(hr_z1 * 100), "hr_z2": int(hr_z2 * 100),
                "hr_z3": int(hr_z3 * 100), "hr_z4": int(hr_z4 * 100),
                "race_choice": race_choice,
                "race_km": float(race_km),
                "lo_hr": float(effort_band[0]), "hi_hr": float(effort_band[1]),
                "long_run_ratio_thresh": float(long_run_ratio_thresh),
                "readiness_window_days": int(readiness_window_days),
                "prediction_lookback_days": int(prediction_lookback_days),
                "only_runs": bool(only_runs),
                "exclude_manual": bool(exclude_manual),
                "exclude_trainer": bool(exclude_trainer),
                "use_miles": bool(use_miles),
            }
            _save_aid = st.session_state.get("strava_athlete_id")
            if _SUPABASE_ENABLED and _save_aid:
                _pref_err = sb_save_preferences(_save_aid, _prefs_to_save)
                if _pref_err:
                    st.error(f"Save failed: {_pref_err}")
                else:
                    st.session_state["_prefs"] = _prefs_to_save
                    st.success("Settings saved ✓")
            else:
                st.session_state["_prefs"] = _prefs_to_save
                st.success("Settings saved locally ✓")

    # Strava connection controls (shown when authenticated)
    if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
        st.divider()
        _athlete_id = st.session_state.get("strava_athlete_id")
        _athlete_display = st.session_state.get("strava_athlete_name", "")
        if _SUPABASE_ENABLED and _athlete_id:
            _ath_row = sb_load_athlete(_athlete_id)
            if _ath_row:
                _athlete_display = _ath_row.get("display_name", _athlete_display)
        if _athlete_display:
            st.caption(f"🔗 Connected as **{_athlete_display}**")

        # Last synced timestamp
        _fetched_at = st.session_state.get("strava_fetched_at", "")
        if _fetched_at == "supabase":
            # Use the athletes table fetched_at for the real sync time
            _real_ts = (_ath_row or {}).get("fetched_at", "") if _SUPABASE_ENABLED and _athlete_id else ""
            if _real_ts:
                try:
                    _age = datetime.utcnow() - datetime.fromisoformat(_real_ts.replace("Z", "+00:00").split("+")[0])
                    _age_str = (f"{int(_age.total_seconds() // 3600)}h ago"
                                if _age.total_seconds() >= 3600
                                else f"{int(_age.total_seconds() // 60)}m ago")
                    st.caption(f"Last synced: {_age_str}")
                except Exception:
                    st.caption("Last synced: from cloud")
            else:
                st.caption("Last synced: from cloud")
        elif _fetched_at:
            try:
                _age = datetime.utcnow() - datetime.fromisoformat(_fetched_at)
                _age_str = (f"{int(_age.total_seconds() // 3600)}h ago"
                            if _age.total_seconds() >= 3600
                            else f"{int(_age.total_seconds() // 60)}m ago")
                st.caption(f"Last synced: {_age_str}")
            except Exception:
                pass

        _col1, _col2 = st.columns(2)
        if _col1.button("Refresh data", help="Re-fetch all activities from Strava"):
            _refresh_athlete_id = st.session_state.get("strava_athlete_id")
            for _k in ["strava_activities", "strava_streams", "gear_details", "strava_fetched_at"]:
                st.session_state.pop(_k, None)
            if os.path.exists(STRAVA_CACHE_PATH):
                os.remove(STRAVA_CACHE_PATH)
            # Clear Supabase activities so they re-fetch from API
            if _SUPABASE_ENABLED and _refresh_athlete_id:
                try:
                    _get_supabase().table("activities").delete().eq("athlete_id", _refresh_athlete_id).execute()
                    _get_supabase().table("streams").delete().eq("athlete_id", _refresh_athlete_id).execute()
                except Exception:
                    pass
            st.rerun()
        if _col2.button("Disconnect"):
            for _k in ["strava_tokens", "strava_activities", "strava_streams", "gear_details",
                        "strava_fetched_at", "_auth_persisted", "_prefs"]:
                st.session_state.pop(_k, None)
            if _COOKIES_ENABLED and _cookies is not None:
                try:
                    _cookies.remove("strava_refresh_token")
                    _cookies.remove("strava_athlete_id")
                except Exception:
                    pass
            st.rerun()

# -----------------------------
# Load & filter activities
# -----------------------------
if _OAUTH_ENABLED and "strava_tokens" not in st.session_state:
    _redirect_uri = _strava_secrets.get("redirect_uri", "http://localhost:8501")
    _auth_url = get_strava_auth_url(_STRAVA_CLIENT_ID, _redirect_uri)

    # ── Landing page ─────────────────────────────────────────────────
    st.markdown("### Your Strava data. Actually analysed.")
    st.markdown(
        "This dashboard turns your Strava activities into the kind of insights "
        "you'd normally need TrainingPeaks or a coach to get — for free."
    )

    st.markdown("&nbsp;")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**📈 Training Load**\nSee your fitness, fatigue and form trend over time — and know when you're ready to race or need rest.")
    c2.markdown("**🎯 Race Prediction**\nGet a personalised finish time prediction with confidence range, based on your actual training data.")
    c3.markdown("**❤️ HR & Pace Analysis**\nUnderstand your heart rate zones, cadence, pace trends and how your training balance compares to the 80/20 model.")

    st.markdown("&nbsp;")

    st.markdown(
        f'<a href="{_auth_url}" target="_self" style="text-decoration:none;">'
        '<button style="background:#FC4C02;color:white;padding:14px 36px;border:none;'
        'border-radius:6px;font-size:17px;font-weight:700;cursor:pointer;letter-spacing:0.3px;">'
        '🔗 Connect with Strava</button></a>',
        unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")
    st.caption(
        "🔒 **Read-only access** — this app never modifies your Strava data. "
        "Your activities are loaded once and stored securely. "
        "You can disconnect at any time from the sidebar."
    )
    st.stop()
elif _OAUTH_ENABLED and "strava_tokens" in st.session_state:
    _token = get_valid_token(_STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET)
    if _token is None:
        st.error("Could not get a valid Strava token. Please reconnect.")
        st.stop()

    _athlete_id = st.session_state.get("strava_athlete_id")
    _athlete_name = st.session_state.get("strava_athlete_name", "")

    # Persist cookies + Supabase on first stable authenticated render
    if not st.session_state.get("_auth_persisted") and _athlete_id:
        _tokens_now = st.session_state["strava_tokens"]
        # Set cookies without Secure flag so they work on HTTP (localhost) and HTTPS
        if _COOKIES_ENABLED and _cookies is not None:
            _cookies.set("strava_refresh_token", _tokens_now["refresh_token"])
            _cookies.set("strava_athlete_id", str(_athlete_id))
        if _SUPABASE_ENABLED:
            _err = sb_save_athlete(_athlete_id, _athlete_name, _tokens_now["refresh_token"])
            if _err:
                st.session_state["_sb_save_error"] = f"athlete: {_err}"
        # Load preferences if not already loaded
        if _SUPABASE_ENABLED and "_prefs" not in st.session_state:
            _prefs_loaded = sb_load_preferences(_athlete_id)
            if _prefs_loaded:
                st.session_state["_prefs"] = _prefs_loaded
        st.session_state["_auth_persisted"] = True

    # Load activities: Supabase → disk cache → Strava API (in priority order)
    if "strava_activities" not in st.session_state:
        _loaded = False

        # 1. Try Supabase
        if _SUPABASE_ENABLED and _athlete_id:
            _sb_acts = sb_load_activities(_athlete_id)
            if _sb_acts:
                st.session_state["strava_activities"] = _sb_acts
                st.session_state["strava_fetched_at"] = "supabase"
                _loaded = True

        # 2. Try local disk cache (for local dev)
        if not _loaded:
            _disk = load_strava_disk_cache()
            if _disk:
                _raw_from_disk = _disk["activities"]
                st.session_state["strava_activities"] = _raw_from_disk
                st.session_state["strava_fetched_at"] = _disk.get("fetched_at", "")
                _loaded = True
                # Migrate disk cache into Supabase
                if _SUPABASE_ENABLED and _athlete_id:
                    _err = sb_save_activities(_athlete_id, _raw_from_disk)
                    if _err:
                        st.session_state["_sb_save_error"] = f"activities (from disk): {_err}"

        # 3. Fetch from Strava API
        if not _loaded:
            with st.spinner("Fetching your activities from Strava\u2026"):
                _raw = fetch_all_activities_api(_token)
                st.session_state["strava_activities"] = _raw
                st.session_state["strava_fetched_at"] = datetime.utcnow().isoformat()
                if _SUPABASE_ENABLED and _athlete_id:
                    _err = sb_save_activities(_athlete_id, _raw)
                    if _err:
                        st.session_state["_sb_save_error"] = f"activities: {_err}"
                save_strava_disk_cache(_raw, {})

    # Load streams: Supabase → disk cache → empty (fetched lazily per date range)
    if "strava_streams" not in st.session_state:
        if _SUPABASE_ENABLED and _athlete_id:
            st.session_state["strava_streams"] = sb_load_streams(_athlete_id)
        else:
            _disk = load_strava_disk_cache()
            st.session_state["strava_streams"] = (
                {int(k): v for k, v in _disk.get("streams", {}).items()} if _disk else {}
            )

    activities = parse_activities_raw(st.session_state["strava_activities"])
    streams_by_id: Dict[int, Dict] = st.session_state.get("strava_streams", {})
else:
    # Static file fallback (local dev without OAuth configured)
    try:
        activities = load_activities(ACTIVITIES_PATH)
    except FileNotFoundError:
        st.error(
            "⚠️ Strava connection not configured. "
            "Add your Strava API credentials to `.streamlit/secrets.toml` to connect your account."
        )
        st.stop()

    try:
        streams_by_id = load_streams(STREAMS_PATH)
    except FileNotFoundError:
        streams_by_id = {}

if only_runs:
    activities = activities[activities["type"].astype(str).str.lower() == "run"]

if exclude_manual:
    activities = activities[~activities["manual"].fillna(False)]

if exclude_trainer:
    activities = activities[~activities["trainer"].fillna(False)]

if len(activities) == 0:
    st.warning("No activities available after filters. Adjust sidebar filters.")
    st.stop()

min_dt = activities["start_dt_local"].min().to_pydatetime()
max_dt = activities["start_dt_local"].max().to_pydatetime()
default_start = max(min_dt, max_dt - pd.Timedelta(days=90))
default_end = max_dt

with st.sidebar:
    st.divider()
    st.subheader("Date range")
    date_range = st.date_input(
        "Date range",
        value=(default_start.date(), default_end.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        label_visibility="collapsed",
    )
    st.caption(
        "💡 Detailed stream data (HR, cadence, pace) is fetched and saved "
        "for activities in this range. Widen the range to unlock analysis "
        "for older runs — each run is only fetched once."
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start.date(), default_end.date()

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Classify every activity — long_run_km depends on sidebar, recomputes when controls change
_long_run_km_for_classify = float(LONG_RUN_DEFAULTS.get(race_choice, 0.60)) * float(race_km)
activities = activities.copy()
activities["run_type"] = classify_all_runs(
    activities, streams_by_id, _long_run_km_for_classify, int(max_hr),
    easy_thresh=hr_z2, tempo_thresh=hr_z3,
)

mask = (activities["start_dt_local"] >= start_ts) & (activities["start_dt_local"] <= end_ts)
df_range = activities.loc[mask].copy()

if len(df_range) == 0:
    st.info("No activities in the selected range.")
    st.stop()

# ── Fetch streams for activities in the selected range (OAuth mode) ───
if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
    _token_for_streams = get_valid_token(_STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET)
    _ids_in_range = [int(i) for i in df_range["id"].tolist()]
    _missing_ids = [i for i in _ids_in_range if i not in streams_by_id]
    if _missing_ids and _token_for_streams:
        _prog = st.progress(0, text=f"Fetching streams for {len(_missing_ids)} activities...")
        for _idx, _aid in enumerate(_missing_ids):
            _streams = fetch_activity_streams_api(_aid, _token_for_streams)
            if _streams:
                streams_by_id[_aid] = _streams
            _prog.progress((_idx + 1) / len(_missing_ids),
                           text=f"Fetching streams… {_idx + 1}/{len(_missing_ids)}")
            time.sleep(0.05)  # stay within rate limits
        _prog.empty()
        st.session_state["strava_streams"] = streams_by_id
        # Persist new streams
        if _SUPABASE_ENABLED and st.session_state.get("strava_athlete_id"):
            sb_save_streams(st.session_state["strava_athlete_id"],
                            {_aid: streams_by_id[_aid] for _aid in _missing_ids if _aid in streams_by_id})
        save_strava_disk_cache(st.session_state["strava_activities"], streams_by_id)

# ── Weather enrichment ────────────────────────────────────────────────
_weather_df = pd.DataFrame(columns=["id", "temp_c"])
_acts_with_loc = activities[
    pd.notna(activities.get("start_lat", pd.Series(dtype=float)))
    & pd.notna(activities.get("start_lng", pd.Series(dtype=float)))
] if "start_lat" in activities.columns and "start_lng" in activities.columns else pd.DataFrame()
if len(_acts_with_loc) > 0:
    try:
        _weather_input = _acts_with_loc[["id", "date", "start_lat", "start_lng", "start_dt_local"]].copy()
        _weather_input["start_dt_local"] = _weather_input["start_dt_local"].astype(str)
        _weather_input["date"] = _weather_input["date"].astype(str)
        _weather_df = fetch_weather_for_activities(_weather_input.to_json(orient="records"))
    except Exception:
        pass

if len(_weather_df) > 0:
    activities = activities.merge(_weather_df, on="id", how="left")
    # Re-apply mask after merge
    mask = (activities["start_dt_local"] >= start_ts) & (activities["start_dt_local"] <= end_ts)
    df_range = activities.loc[mask].copy()

# Precompute shared data used across tabs
daily_all, weekly_all = build_daily_weekly(activities, max_hr=max_hr, date_range=(start_ts, end_ts))
bests           = compute_personal_bests(activities)
consistency     = compute_consistency(activities)
vo2max_est      = estimate_vo2max(bests)
cadence_df      = compute_cadence_stats(df_range, streams_by_id)

# ── Insight banner (always visible above tabs) ───────────────────────
insight_msg, insight_level = generate_insight(daily_all, activities)
if insight_level == "success":
    st.success(f"**Today's insight** — {insight_msg}")
elif insight_level == "warning":
    st.warning(f"**Today's insight** — {insight_msg}")
else:
    st.info(f"**Today's insight** — {insight_msg}")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab1, tab2, tab3, tab4, tab5, tab_gear, explanation_tab, tab_streams = st.tabs([
    "Overview",
    "Training Load",
    "Pace & Efficiency",
    "Long Runs",
    "Recovery & Risk",
    "Race Predictor",
    "Gear",
    "Guide",
    "Raw Streams",
])
# =====================================================================
# TAB OVERVIEW
# =====================================================================
with tab_overview:
    # ── All-time KPIs ────────────────────────────────────────────────
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Total runs", f"{consistency['total_runs']:,}")
    o2.metric("Total distance", _dist_fmt(consistency['total_km'], use_miles, decimals=0))
    o3.metric("Week streak", f"{consistency['week_streak']} wks",
              help="Consecutive weeks with at least one run.")
    o4.metric("Consistent weeks (last 12)", f"{consistency['pct_consistent_weeks']:.0f}%",
              help="Weeks with ≥3 runs in the last 12 weeks.")
    if vo2max_est is not None:
        o5.metric("Estimated VO₂max", f"{vo2max_est:.1f} ml/kg/min",
                  help="VDOT estimate via Jack Daniels' formula from your best recorded effort. Not a lab test.")
    else:
        o5.metric("Estimated VO₂max", "—", "Need a 5K–marathon effort")

    st.divider()

    # ── Data completeness indicator ───────────────────────────────────
    st.subheader("Data quality — selected period")
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
        icon = "✓" if pct >= 80 else ("△" if pct >= 50 else "✗")
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
    st.subheader("Training calendar — last 12 months")
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
                           f"{_dist_fmt(b['distance_km'], use_miles)} · {dt_str}")
        else:
            col.metric(label, "—", "No qualifying run yet")

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
        lambda m: f"{int(m // 60)}h {int(m % 60):02d}m" if pd.notna(m) and m > 0 else "—"
    )
    # Avg pace = total time / total distance
    _avg_pace_mkm = (_wlog_grp["duration_min"] / _wlog_grp["distance_km"]).replace([np.inf, -np.inf], np.nan)
    _wlog_disp[f"Avg pace ({_p_unit(use_miles)})"] = _avg_pace_mkm.apply(
        lambda x: _format_pace(x, use_miles) if pd.notna(x) else "—"
    )
    _wlog_disp[f"Long run ({_d_unit(use_miles)})"] = _wlog_grp["long_run_km"].apply(
        lambda x: _dist_fmt(x, use_miles) if pd.notna(x) else "—"
    )
    _wlog_disp["Quality"] = _wlog_grp["quality"].apply(lambda x: f"{int(x)} 🔥" if x > 0 else "—")
    _wlog_disp["Avg HR"] = _wlog_grp["avg_hr"].apply(
        lambda x: f"{int(round(x))} bpm" if pd.notna(x) and x > 0 else "—"
    )
    if "total_elevation_gain" in df_range.columns:
        _wlog_disp["Elev gain (m)"] = _wlog_grp["elev_gain"].apply(
            lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "—"
        )

    st.dataframe(_wlog_disp, hide_index=True, use_container_width=True)

    st.divider()

    # ── VO2max context ────────────────────────────────────────────────
    if vo2max_est is not None:
        st.subheader("Estimated VO₂max")
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            # Contextual classification (Firstbeat / ACSM guidelines for men 30-39 as rough reference)
            if vo2max_est >= 60:
                v_class, v_color = "Elite", "#52e88a"
            elif vo2max_est >= 52:
                v_class, v_color = "Excellent", "#3dba6e"
            elif vo2max_est >= 45:
                v_class, v_color = "Good", "#fd8d3c"
            elif vo2max_est >= 38:
                v_class, v_color = "Average", "#fdae6b"
            else:
                v_class, v_color = "Below average", "#d62728"
            st.markdown(
                f"<div style='font-size:3rem;font-weight:700;color:{v_color}'>{vo2max_est:.1f}</div>"
                f"<div style='font-size:1rem;color:{v_color}'>{v_class}</div>"
                f"<div style='font-size:0.8rem;color:grey;margin-top:4px'>ml · kg⁻¹ · min⁻¹ (VDOT estimate)</div>",
                unsafe_allow_html=True,
            )
        with v_col2:
            st.caption(
                "VDOT is derived from your best recorded effort using Jack Daniels' formula. "
                "It reflects race-pace fitness — not a lab VO₂max. A 1-point increase typically means "
                "~1–2% faster race times. Improve it by adding easy aerobic volume and one quality session per week."
            )
            # Reference table
            ref = pd.DataFrame({
                "Level": ["Beginner", "Average", "Good", "Excellent", "Elite"],
                "VO₂max (ml/kg/min)": ["< 35", "35–44", "45–51", "52–59", "60+"],
                "~5K time": ["> 30 min", "25–30 min", "20–25 min", "17–20 min", "< 17 min"],
            })
            st.dataframe(ref, hide_index=True, use_container_width=True)

# =====================================================================
# TAB 0 - Streams explorer
# =====================================================================

with tab_streams:
    if not show_streams_tab:
        st.info("Enable **🔧 Raw streams explorer** in the sidebar to use this tab.")
    else:
        st.subheader("Strava Streams Explorer")
        st.caption("Inspect per-second/per-sample streams (pace, HR, cadence, grade, etc.) for individual activities.")

        if not isinstance(streams_by_id, dict) or len(streams_by_id) == 0:
            st.warning("No streams found in streams file.")
        else:
            # Build a nice label: date — name — distance
            df_act = df_range.copy()
            df_act["date_str"] = df_act["start_dt_local"].dt.strftime("%Y-%m-%d")
            df_act["label"] = (df_act["date_str"].astype(str) + " — " +
                               df_act["name"].fillna("").astype(str).str.slice(0, 50) + " — " +
                               df_act["distance_km"].fillna(0).map(lambda x: _dist_fmt(x, use_miles)))

            # Keep only activities that have streams available
            df_act["has_streams"] = df_act["id"].astype(int).isin(streams_by_id.keys())
            df_streamable = df_act[df_act["has_streams"]].sort_values("start_dt_local", ascending=False)

            if len(df_streamable) == 0:
                st.info("No activities with streams available in the selected date range. Try widening the date range.")
            else:
                # Select activity by label but map to id
                label_to_id = dict(zip(df_streamable["label"], df_streamable["id"].astype(int)))
                selected_label = st.selectbox("Activity", list(label_to_id.keys()))
                activity_id = label_to_id[selected_label]

                streams = streams_by_id.get(int(activity_id), {})
                df = streams_to_df(streams)

                # Choose x-axis
                x_options = []
                for col in ["time", "distance", "distance_km"]:
                    if col in df.columns:
                        x_options.append(col)

                if not x_options:
                    st.error("No 'time' or 'distance' stream found for this activity.")
                else:
                    default_x = "distance_km" if "distance_km" in x_options else x_options[0]
                    x_axis = st.radio("X axis", x_options, horizontal=True, index=x_options.index(default_x))

                    # y-axis candidates: numeric columns excluding x and lat/lng
                    exclude = set(["lat", "lng"]) | set(x_options)
                    y_candidates = [c for c in df.columns if c not in exclude]

                    numeric_cols = []
                    for c in y_candidates:
                        s = pd.to_numeric(df[c], errors="coerce")
                        if s.notna().any():
                            numeric_cols.append(c)

                    if not numeric_cols:
                        st.error("No numeric metrics available to plot for this activity.")
                    else:
                        default_metric = "pace_min_per_km" if "pace_min_per_km" in numeric_cols else ("heartrate" if "heartrate" in numeric_cols else numeric_cols[0])
                        metric = st.selectbox("Metric (Y axis)", numeric_cols, index=numeric_cols.index(default_metric))

                        smooth_window = st.slider("Rolling mean window", 1, 200, 1, help="Window size for rolling mean calculation.")
                        plot_df = df.copy()
                        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
                        plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce")

                        if smooth_window > 1:
                            plot_df[metric] = plot_df[metric].rolling(smooth_window, min_periods=1).mean()

                        plot_df = plot_df.dropna(subset=[x_axis, metric])

                        fig = px.line(
                            plot_df,
                            x=x_axis,
                            y=metric,
                            title=f"Activity {activity_id}: {metric} vs {x_axis}",
                        )
                        fig.update_layout(height=600)

                        # Invert pace axis (lower is better)
                        if metric == "pace_min_per_km":
                            fig.update_yaxes(autorange="reversed")

                        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 1 - Training Load
# =====================================================================
with tab1:
    daily, weekly = daily_all, weekly_all

    if len(daily) == 0 or len(weekly) == 0:
        st.info("Not enough data in the selected range to compute training load.")
        st.markdown(
            "**To see this tab:** Make sure your date range includes at least **4 weeks** of runs with "
            "heart rate data recorded. Expand your date range in the sidebar or sync more activities."
        )
    else:
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
                  help="load = duration_min × (avgHR / maxHR). Only runs with HR data contribute.")
        k3.metric("ACWR", f"{latest_day['acwr']:.2f}" if pd.notna(latest_day["acwr"]) else "N/A",
                  f"{acwr_emoji} {acwr_label}",
                  help="Acute:Chronic Workload Ratio. 0.8–1.3 is the balanced zone.")
        k4.metric("Long run % of race", f"{latest_week['long_run_ratio']*100:.0f}%")

        st.caption("HR load requires average HR data on each run. Distance and duration charts use all runs.")

        st.divider()

        # ── Performance Management Chart (PMC) ───────────────────────
        st.subheader("Performance Management Chart")
        st.caption(
            "**Fitness** (CTL, 28d EWMA) builds slowly. **Fatigue** (ATL, 7d EWMA) spikes fast. "
            "**Form / TSB** = Fitness − Fatigue. Positive TSB = fresh; negative = fatigued. "
            "Race-ready zone: TSB between +5 and +25."
        )

        pmc_fig = go.Figure()

        # TSB form-zone bands
        pmc_fig.add_hrect(y0=5, y1=25, fillcolor="rgba(82,232,138,0.08)", line_width=0,
                          annotation_text="🏁 Race-ready", annotation_position="top left",
                          annotation=dict(font_size=11, font_color="rgba(82,232,138,0.7)"),
                          yref="y2")
        pmc_fig.add_hrect(y0=-30, y1=5, fillcolor="rgba(253,141,60,0.05)", line_width=0,
                          annotation_text="⚡ Productive training", annotation_position="top left",
                          annotation=dict(font_size=11, font_color="rgba(253,141,60,0.6)"),
                          yref="y2")
        pmc_fig.add_hrect(y0=-100, y1=-30, fillcolor="rgba(214,39,40,0.07)", line_width=0,
                          annotation_text="😓 High fatigue — recover", annotation_position="top left",
                          annotation=dict(font_size=11, font_color="rgba(214,39,40,0.6)"),
                          yref="y2")
        pmc_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", line_width=1,
                          yref="y2")

        # CTL, ATL on left axis
        pmc_fig.add_trace(go.Scatter(
            x=daily["date_ts"], y=daily["chronic_load"], mode="lines",
            name="Fitness / CTL (28d)", line=dict(color="#3dba6e", width=2.5),
        ))
        pmc_fig.add_trace(go.Scatter(
            x=daily["date_ts"], y=daily["acute_load"], mode="lines",
            name="Fatigue / ATL (7d)", line=dict(color="#fd8d3c", width=2, dash="dot"),
        ))

        # TSB on right axis
        pmc_fig.add_trace(go.Scatter(
            x=daily["date_ts"], y=daily["tsb"], mode="lines",
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
        _today_row = daily.sort_values("date_ts").iloc[-1]
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
            st.success(f"Current form (TSB): **+{latest_tsb:.0f}** — You're fresh. Good window to race or do a breakthrough session.")
        elif latest_tsb > 5:
            st.success(f"Current form (TSB): **+{latest_tsb:.0f}** — Good form. Training will be well absorbed.")
        elif latest_tsb > -10:
            st.info(f"Current form (TSB): **{latest_tsb:.0f}** — Productive training zone. Slightly fatigued but adapting.")
        elif latest_tsb > -30:
            st.warning(f"Current form (TSB): **{latest_tsb:.0f}** — Carrying fatigue. Monitor recovery closely.")
        else:
            st.error(f"Current form (TSB): **{latest_tsb:.0f}** — Heavy fatigue accumulated. Consider a recovery block.")

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
                    st.success("Good polarization — mostly aerobic base building.")
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
            df_range["week_start_col"] = df_range["start_dt_local"].dt.to_period("W").apply(
                lambda p: p.start_time
            )
            weekly_by_type = (
                df_range.groupby(["week_start_col", "run_type"])["distance_km"]
                .sum()
                .reset_index()
                .rename(columns={"week_start_col": "week_start"})
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
            "Elite endurance athletes spend ~80% of their time in Z1–Z2."
        )
        zone_df = compute_hr_zones(df_range, streams_by_id, int(max_hr), hr_zones=_hr_zones)
        zone_df = zone_df[zone_df["Minutes"] > 0]
        if len(zone_df) == 0:
            st.info("No HR data available in the selected range to compute zones.")
        else:
            zone_colors = ["#6baed6", "#74c476", "#fd8d3c", "#e6550d", "#d62728"]
            zone_descriptions = {
                "Z1 Recovery":  "Very easy — active recovery, cool-downs",
                "Z2 Aerobic":   "Comfortable — conversational pace, fat-burning base",
                "Z3 Tempo":     "Moderate — marathon to half-marathon effort",
                "Z4 Threshold": "Hard — 10K race effort, lactate threshold",
                "Z5 VO₂max":   "Maximum — short intervals, 5K race effort",
            }
            zone_df["Description"] = zone_df["Zone"].map(zone_descriptions).fillna("")
            zone_df["Hours"] = (zone_df["Minutes"] / 60).round(1)
            zone_df["Pct"] = (zone_df["Minutes"] / zone_df["Minutes"].sum() * 100).round(1)

            _zone_feelings = {
                "Z1 Recovery": "Recovery", "Z2 Aerobic": "Easy / Base",
                "Z3 Tempo": "Comfortably hard", "Z4 Threshold": "Race effort",
                "Z5 VO₂max": "Max intensity",
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
                st.success(f"**{z1z2_pct:.0f}%** of training in Z1–Z2 ✓ — well within the 80/20 polarized model. Z2 alone: {z2_pct:.0f}%.")
            elif z1z2_pct >= 60:
                st.warning(f"**{z1z2_pct:.0f}%** in Z1–Z2 — slightly high intensity. Try adding more easy runs to reach 75–80%.")
            else:
                st.error(f"Only **{z1z2_pct:.0f}%** in Z1–Z2 — training hard. Add easy days to avoid accumulated fatigue.")

            # Zone reference table
            with st.expander("Zone reference guide"):
                _zone_bpm_rows = []
                for (zname, lo_frac, hi_frac) in _hr_zones:
                    lo_bpm = int(lo_frac * max_hr)
                    hi_bpm = int(min(hi_frac, 1.0) * max_hr)
                    _zone_bpm_rows.append({
                        "Zone": zname,
                        "BPM range": f"{lo_bpm}–{hi_bpm}" if hi_frac < 2 else f">{lo_bpm}",
                        "Feel": zone_descriptions.get(zname, ""),
                        "Training purpose": {
                            "Z1 Recovery": "Active recovery, blood flow, easy distance",
                            "Z2 Aerobic": "Aerobic base, mitochondrial density, fat adaptation",
                            "Z3 Tempo": "Lactate clearance, marathon fitness, 'comfortably hard'",
                            "Z4 Threshold": "Raise lactate threshold, 10K–HM race pace",
                            "Z5 VO₂max": "Maximal oxygen uptake, short sharp intervals",
                        }.get(zname, ""),
                    })
                st.dataframe(pd.DataFrame(_zone_bpm_rows), hide_index=True, use_container_width=True)

        st.divider()

        # ── Advanced load metrics (collapsed by default) ─────────────
        with st.expander("Advanced metrics (for coaches) — monotony & strain"):
            st.caption("Monotony = mean daily load ÷ std. Strain = weekly load × monotony. High monotony with high load = little variation in a big week — watch recovery.")
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
                    title="Training strain (load × monotony)",
                    labels={"week_start": "Week", "strain": "Strain"},
                )
                fig_strain.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_strain, use_container_width=True)

# =====================================================================
# TAB 2 - Pace & Efficiency
# =====================================================================
with tab2:
    st.subheader("Pace, effort & efficiency")

    d2 = df_range.copy()
    d2["hr_intensity"] = d2["avg_hr"] / float(max_hr)
    d2 = d2[pd.notna(d2["avg_hr"]) & pd.notna(d2["pace_min_per_km"])]

    if len(d2) == 0:
        st.info("No runs with both average HR and average speed in the selected range.")
    else:

        lo, hi = effort_band
        d2["in_race_effort_band"] = (d2["hr_intensity"] >= lo) & (d2["hr_intensity"] <= hi)
        # Compute display-unit columns once so all charts below stay consistent
        _pace_factor = (1.0 / KM_TO_MILES) if use_miles else 1.0
        d2["pace_disp"]     = d2["pace_min_per_km"] * _pace_factor
        d2["distance_disp"] = d2["distance_km"] * (KM_TO_MILES if use_miles else 1.0)
        _pace_lbl = f"Pace ({_p_unit(use_miles)})"
        _dist_lbl = f"Distance ({_d_unit(use_miles)})"

        d2["speed_per_hr"] = d2["avg_speed_mps"] / d2["avg_hr"]
        d2["hr_per_km"] = d2["avg_hr"] * d2["pace_min_per_km"]

        d2 = d2.sort_values("start_dt_local")
        race_eff = d2[d2["in_race_effort_band"]].copy()

        top = st.columns(4)
        top[0].metric("Runs in range", f"{len(d2)}")
        top[1].metric("In race-effort band", f"{len(race_eff)}", help=f"HR band = {lo:.0%}–{hi:.0%} of max HR")
        top[2].metric("Median pace", _format_pace(float(d2["pace_min_per_km"].median()), use_miles))
        top[3].metric("Median HR", f"{int(round(d2['avg_hr'].median()))} bpm")

        if len(race_eff) == 0:
            st.warning(
                f"No runs fall in your race-effort HR band ({lo:.0%}–{hi:.0%} of {max_hr} bpm max HR). "
                "The efficiency trend below will use all runs instead. "
                "Widen the band in the sidebar to capture more efforts."
            )
        else:
            st.caption(f"Race-effort band: {lo:.0%}–{hi:.0%} of max HR. Adjustable in the sidebar.")

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
                trendline_color_override="white",
                hover_data=["name", "start_dt_local", "distance_disp", "duration_min", "total_elevation_gain", "hr_intensity"],
                labels={"avg_hr": "Average HR (bpm)", "pace_disp": _pace_lbl,
                        "run_type": "Type", "distance_disp": _dist_lbl},
                title="Pace vs HR — coloured by run type",
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

        st.subheader("Efficiency trend (race-effort runs)")
        trend_df = race_eff.copy() if len(race_eff) >= 3 else d2.copy()
        if len(race_eff) < 3:
            st.info("Not enough race-effort runs for a stable trend yet (need ~3+). Showing all runs instead.")

        trend_df["speed_per_hr_roll"] = trend_df["speed_per_hr"].rolling(window=5, min_periods=1).median()
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
                        text="→ Pace stable over the period", xref="paper", yref="paper",
                        x=0.01, y=0.04, showarrow=False,
                        font=dict(size=11, color="rgba(200,200,200,0.7)"),
                    )
                elif _pace_delta < 0:
                    fig_pace.add_annotation(
                        text=f"↗ {_delta_sec:.0f}s/{_trend_unit} faster over the period",
                        xref="paper", yref="paper", x=0.01, y=0.04, showarrow=False,
                        font=dict(size=11, color="rgba(61,186,110,0.9)"),
                    )
                else:
                    fig_pace.add_annotation(
                        text=f"↘ {_delta_sec:.0f}s/{_trend_unit} slower over the period",
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
                          help="Steps per minute (both feet). 170–180 spm is the commonly cited target range.")
            cad_c2.metric("% runs ≥ 170 spm", f"{(cadence_df['avg_cadence'] >= 170).mean()*100:.0f}%")
            cad_c3.metric("% runs ≥ 180 spm", f"{(cadence_df['avg_cadence'] >= 180).mean()*100:.0f}%")

            fig_cad = go.Figure()
            fig_cad.add_hrect(
                y0=170, y1=180, fillcolor="rgba(61,186,110,0.15)", line_width=0,
                annotation_text="✓ Optimal range (170–180 spm)",
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
                labels={"temp_c": "Temperature (°C)", "pace_disp": _pace_lbl},
                title="Pace vs temperature",
                trendline="ols",
            )
            fig_weather.update_yaxes(autorange="reversed")
            fig_weather.update_layout(yaxis_tickformat=".2f")
            st.plotly_chart(fig_weather, use_container_width=True)
            st.caption("Lower pace = faster. A downward trendline suggests you run faster in cooler conditions.")

# =====================================================================
# TAB 3 - Long-Run Fatigue
# =====================================================================
with tab3:
    st.subheader("Long-run durability")

    long_run_min_km = float(long_run_ratio_thresh) * float(race_km)
    st.caption(f"Analysing runs ≥ {_dist_fmt(long_run_min_km, use_miles)} (your threshold × race distance). Adjust in the sidebar.")

    fatigue = build_fatigue_table(df_range, streams_by_id, long_run_min_km=long_run_min_km)

    if len(fatigue) == 0:
        st.info("No long runs with stream data in the selected range. Lower the long-run threshold or widen the date range.")
    else:
        med_pf  = np.nanmedian(fatigue["pace_fade_pct"])
        med_hrd = np.nanmedian(fatigue["hr_drift_pct"])
        med_dec = np.nanmedian(fatigue["decoupling"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Long runs analysed", f"{len(fatigue)}")
        k2.metric("Median pace fade", f"{med_pf*100:.1f}%",
                  help="<5% = good · 5–10% = moderate · >10% = significant fade")
        k3.metric("Median HR drift",  f"{med_hrd*100:.1f}%" if np.isfinite(med_hrd) else "N/A",
                  help="How much HR rose in the second half vs first at similar pace. Higher = more cardiovascular stress.")
        k4.metric("Median aerobic efficiency loss", f"{med_dec*100:.1f}%" if np.isfinite(med_dec) else "N/A",
                  help="Drop in speed÷HR from first to second half. Negative = efficiency declined. Closer to 0 = better durability.")

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
                             annotation_text="5% — moderate", annotation_position="top left")
            fig_pf.add_hline(y=0.10, line_dash="dash", line_color="red",
                             annotation_text="10% — significant", annotation_position="top left")
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
            return f"{dt_str} — {r['distance_km']:.1f} km — {str(r.get('name',''))[:40]}"

        options = { _label_row(r): int(r["id"]) for _, r in fatigue.iterrows() }
        selected_label = st.selectbox("Select a run", list(options.keys()))
        selected_id = options[selected_label]

        s = streams_by_id.get(int(selected_id), {})
        run_df = build_within_run_df(s)

        if run_df is None or len(run_df) == 0:
            st.warning("Selected run is missing required streams (distance/time/velocity).")
            st.stop()

        # ── Grade-adjusted pace ──────────────────────────────────────
        raw_s = s  # streams dict for selected run
        grade_arr = _safe_array(raw_s, "grade_smooth")
        if grade_arr is not None and len(grade_arr) == len(run_df):
            g = grade_arr / 100.0  # percent → fraction
            # Minetti et al. metabolic cost: C(g) = 280.5g⁵ - 58.7g⁴ - 76.8g³ + 51.9g² + 19.6g + 2.5
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
            st.caption("GAP (grade-adjusted pace) normalises for elevation using the Minetti metabolic cost formula — it shows what your flat equivalent effort was.")

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

# =====================================================================
# TAB 4 - Readiness & Risk
# =====================================================================
with tab4:
    st.subheader("Readiness, recovery & injury-risk proxies")

    daily, weekly = daily_all.copy(), weekly_all.copy()
    if len(daily) == 0 or len(weekly) == 0:
        st.info("Not enough data in the selected range to compute readiness.")

    else:
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
            readiness = ("Low risk / likely fresh", "🟢")
        elif risk_score < 55:
            readiness = ("Moderate risk", "🟠")
        else:
            readiness = ("High risk", "🔴")

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overreach/injury risk", f"{risk_score:.0f}/100", f"{readiness[1]} {readiness[0]}",
                  help="Composite proxy based on ACWR, recent load spikes, and rest days. Not medical advice.")
        c2.metric("ACWR", f"{acwr_val:.2f}" if pd.notna(acwr_val) else "N/A",
                  help="0.8–1.3 = balanced. Above 1.5 = elevated overreach risk.")
        c3.metric("Rest days (last 7)", f"{int(latest['rest_days_last7'])}",
                  help="Days with near-zero HR load. Aim for at least 1–2 per week.")
        c4.metric("Acute load (7d EWMA)", f"{latest['acute_load']:.1f}")

        st.caption("These are *proxies* — not medical advice. Use them as signals, not verdicts.")

        # Risk score trend
        fig_risk = px.area(
            d_focus,
            x="date_ts",
            y="risk_score",
            title="Composite risk score (recent window)",
            labels={"date_ts": "Date", "risk_score": "Risk score (0–100)"},
        )
        fig_risk.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_risk, use_container_width=True)

        # Stacked flags chart — shows which flags fired each day
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
                title="Alert flags per day — hover to see which flags fired",
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
        st.caption("Flags runs where your speed-per-HR was below your rolling 20th-percentile baseline — a signal of fatigue, illness, or heat stress.")
        comp = compute_compromised_runs(df_range, max_hr=max_hr)
        if len(comp) == 0 or comp["eff_q20"].isna().all():
            st.info("Not enough HR-bearing runs to detect outliers yet (need ~6+).")
        else:
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
                    labels={"start_dt_local": "Date", "speed_per_hr": "Speed ÷ HR (m/s per bpm)"},
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
                st.dataframe(display.sort_values("start_dt_local", ascending=False), use_container_width=True)

# =====================================================================
# TAB 5 - Race prediction
# =====================================================================
with tab5:
    st.subheader("Race predictor")

    model_end = end_ts
    model_start = model_end - pd.Timedelta(days=int(prediction_lookback_days))

    model_runs = activities[(activities["start_dt_local"] >= model_start) & (activities["start_dt_local"] <= model_end)].copy()
    model_runs = model_runs[model_runs["type"].astype(str).str.lower() == "run"].copy()

    # Prefer effort runs for prediction (Long Run, Tempo, Workout, Race) — easy runs are too slow
    # to give accurate Riegel extrapolations. Fall back to all runs if not enough effort runs.
    _effort_types = {"Long Run", "Tempo", "Workout", "Race", "General"}
    if "run_type" in model_runs.columns:
        effort_runs = model_runs[model_runs["run_type"].isin(_effort_types)]
        if len(effort_runs) >= 3:
            model_runs = effort_runs
            st.caption(
                f"Prediction uses **{len(model_runs)} effort runs** (Long Run / Tempo / Workout / Race) "
                f"from the last {int(prediction_lookback_days)} days. Easy runs are excluded to avoid "
                "underestimating your race fitness."
            )
        else:
            st.caption(
                f"Using all {len(model_runs)} runs — fewer than 3 effort runs found in the window. "
                "Log tempo, interval, or long runs to improve prediction accuracy."
            )

    st.caption(
        f"Using your last {int(prediction_lookback_days)} days of runs "
        f"({model_start.date()} → {model_end.date()})."
    )

    colA, colB = st.columns([1.2, 1])
    with colA:
        effort_preset = st.selectbox(
            "Prediction mode",
            ["Balanced (default)", "Optimistic", "Conservative"],
            help="Controls how much performance is assumed to drop off at longer distances. Balanced = standard Riegel (1.06). Optimistic = less drop-off (1.03). Conservative = more drop-off (1.09).",
        )
        exp = {"Balanced (default)": 1.06, "Optimistic": 1.03, "Conservative": 1.09}[effort_preset]
    with colB:
        _min_dist_default = 3.0 / KM_TO_MILES if use_miles else 3.0
        _min_dist_max = 20.0 / KM_TO_MILES if use_miles else 20.0
        min_dist_input = st.number_input(
            f"Ignore runs shorter than ({_d_unit(use_miles)})",
            min_value=0.5, max_value=round(_min_dist_max, 1),
            value=round(_min_dist_default, 1), step=0.5,
            help="Short runs (intervals, warm-ups) are excluded from the prediction to reduce noise.",
        )
        min_dist = min_dist_input * KM_TO_MILES if use_miles else min_dist_input

    # Baseline: best equivalent performance
    baseline_sec, source = predict_race_time_riegel(
        model_runs, target_km=float(race_km), exponent=float(exp), min_km=float(min_dist),
    )

    if baseline_sec is None or source is None:
        st.info("Not enough runs in this window to predict. Try increasing the lookback window or lowering the minimum distance.")
    else:
        eff_factor = compute_efficiency_adjustment(
            model_runs, max_hr=int(max_hr), effort_band=effort_band,
            lookback_days=int(max(30, prediction_lookback_days // 3)), end_ts=model_end,
        )
        daily_risk, weekly_risk = compute_risk_table(daily_all, weekly_all)
        risk_factor = compute_risk_penalty(daily_risk)

        pred_sec = baseline_sec * eff_factor * risk_factor

        low_sec, _  = predict_race_time_riegel(model_runs, float(race_km), exponent=max(1.02, exp - 0.03), min_km=float(min_dist))
        high_sec, _ = predict_race_time_riegel(model_runs, float(race_km), exponent=min(1.12, exp + 0.03), min_km=float(min_dist))
        if low_sec is not None and high_sec is not None:
            pred_lo = low_sec * eff_factor * risk_factor
            pred_hi = high_sec * eff_factor * risk_factor
        else:
            pred_lo, pred_hi = None, None

        pace_sec_per_km = pred_sec / float(race_km)
        pace_min = pace_sec_per_km / 60.0

        # Plain-English efficiency and risk descriptions
        eff_delta_sec_km = (eff_factor - 1.0) * pace_sec_per_km
        if abs(eff_factor - 1.0) < 0.005:
            eff_label = "Stable →"
        elif eff_factor < 1.0:
            eff_label = f"Improving fitness (−{abs(eff_delta_sec_km):.0f}s/km faster)"
        else:
            eff_label = f"Declining fitness (+{eff_delta_sec_km:.0f}s/km slower)"

        risk_delta_sec_km = (risk_factor - 1.0) * pace_sec_per_km
        risk_label = "No adjustment" if risk_factor <= 1.001 else f"High load (+{risk_delta_sec_km:.0f}s/km)"

        # Best actual result at this distance (if any)
        best_actual = bests.get(
            {"5.0": "best_5k", "10.0": "best_10k", "21.0975": "best_hm", "42.195": "best_marathon"}.get(
                f"{race_km:.4f}", ""
            )
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted finish time", format_hms(pred_sec))
        k2.metric(f"Predicted pace", _format_pace(pace_min, use_miles))
        k3.metric("Fitness trend adjustment", eff_label)
        k4.metric("Load adjustment", risk_label)

        if pred_lo is not None and pred_hi is not None and np.isfinite(pred_lo) and np.isfinite(pred_hi):
            st.info(
                f"🎯 Predicted finish: **{format_hms(pred_sec)}** &nbsp;·&nbsp; "
                f"Confidence range: **{format_hms(pred_lo)} → {format_hms(pred_hi)}** "
                f"(±{abs(pred_hi - pred_lo)/60:.0f} min). "
                "Based on Riegel's power-law model — accuracy improves with more race-effort runs."
            )

        if best_actual is not None:
            best_actual_sec = best_actual["pace_min_per_km"] * race_km * 60.0
            diff = pred_sec - best_actual_sec
            sign = "+" if diff >= 0 else "−"
            st.info(
                f"Your best recorded effort at this distance was **{_format_pace(best_actual['pace_min_per_km'], use_miles)}** "
                f"({pd.to_datetime(best_actual['date']).strftime('%b %Y')}) — "
                f"equivalent to {format_hms(best_actual_sec)}. "
                f"Prediction is {sign}{abs(diff/60):.0f} min relative to that."
            )

        st.divider()
        with st.expander("🔍 Show source runs & model data"):
            st.write(
                f"**Best source run:** {pd.to_datetime(source['start_dt_local']).strftime('%Y-%m-%d')} "
                f"— {float(source['distance_km']):.1f} km in {format_hms(float(source['duration_min'])*60.0)}"
            )

            # Show the top equivalent performances to be transparent
            df = model_runs[pd.notna(model_runs["distance_km"]) & pd.notna(model_runs["duration_min"])].copy()
            df = df[(df["distance_km"] >= float(min_dist)) & (df["duration_min"] > 3)].copy()
            df["time_sec"] = df["duration_min"] * 60.0
            df["equiv_target_sec"] = df["time_sec"] * (float(race_km) / df["distance_km"]) ** float(exp)
            df["equiv_time"] = df["equiv_target_sec"].apply(format_hms)
            df = df.sort_values("equiv_target_sec").head(12)

            fig = px.bar(
                df,
                x="equiv_time",
                y=df["start_dt_local"].dt.strftime("%Y-%m-%d"),
                orientation="h",
                title="Top runs by equivalent race performance (lower is better)",
                labels={"equiv_time": "Equivalent finish time", "y": "Run date"},
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

            show = df[["start_dt_local", "name", "distance_km", "duration_min", "equiv_time"]].copy()
            st.dataframe(show.sort_values("equiv_time"), use_container_width=True)

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
        fig_splits.update_yaxes(autorange="reversed", title=_pace_lbl if "pace_lbl" in dir() else f"Pace ({_p_unit(use_miles)})")
        fig_splits.update_layout(
            height=360, barmode="group", xaxis_title=_d_unit(use_miles),
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_splits, use_container_width=True)

        with st.expander("Show full split table"):
            st.dataframe(splits_df, hide_index=True, use_container_width=True)

# =====================================================================
# TAB GEAR - Gear & Shoe Tracker
# =====================================================================
with tab_gear:
    st.header("Gear & Shoe Tracker")

    # Fetch gear names if in OAuth mode
    gear_names_map: Dict[str, str] = {}
    if _OAUTH_ENABLED and "strava_tokens" in st.session_state:
        _token_gear = get_valid_token(_STRAVA_CLIENT_ID, _STRAVA_CLIENT_SECRET)
        if _token_gear and "gear_details" not in st.session_state:
            unique_gear_ids = activities["gear_id"].dropna().unique().tolist() if "gear_id" in activities.columns else []
            _gear_cache: Dict[str, str] = {}
            for gid in unique_gear_ids:
                gid_str = str(gid)
                if gid_str not in ("nan", "", "None"):
                    details = fetch_gear_api(gid_str, _token_gear)
                    if "name" in details:
                        _gear_cache[gid_str] = details["name"]
            st.session_state["gear_details"] = _gear_cache
        gear_names_map = st.session_state.get("gear_details", {})

    gear_df = compute_gear_stats(activities, gear_names_map)

    if len(gear_df) == 0:
        st.info("No gear data found. Gear info comes from Strava — make sure you've logged shoes on your activities.")
    else:
        # Warning thresholds
        RETIRE_KM = 800
        WARN_KM = 600
        _g_factor = KM_TO_MILES if use_miles else 1.0
        _g_unit = _d_unit(use_miles)
        RETIRE_D = RETIRE_KM * _g_factor
        WARN_D = WARN_KM * _g_factor

        # KPI cards per shoe
        _gear_cols = st.columns(min(len(gear_df), 4))
        for _gi, (_, _grow) in enumerate(gear_df.iterrows()):
            _km = _grow["total_km"]
            _d = _km * _g_factor
            if _km >= RETIRE_KM:
                _delta_str = "Replace now"
                _delta_color = "inverse"
            elif _km >= WARN_KM:
                _delta_str = f"{RETIRE_D - _d:.0f} {_g_unit} left"
                _delta_color = "inverse"
            else:
                _delta_str = f"{RETIRE_D - _d:.0f} {_g_unit} remaining"
                _delta_color = "normal"
            _last_used_str = (
                _grow["last_used"].strftime("%b %d, %Y")
                if pd.notna(_grow["last_used"])
                else "N/A"
            )
            _gear_cols[_gi % 4].metric(
                label=_grow["name"],
                value=f"{_d:,.0f} {_g_unit}",
                delta=_delta_str,
                delta_color=_delta_color,
                help=f"{_grow['runs']} runs · Last used: {_last_used_str}",
            )

        st.divider()

        # Bar chart: mileage per shoe with color coding
        gear_df["status"] = gear_df["total_km"].apply(
            lambda x: "Replace now" if x >= RETIRE_KM else ("Getting worn" if x >= WARN_KM else "Good")
        )
        gear_df["total_disp"] = gear_df["total_km"] * _g_factor
        _color_map_gear = {"Good": "#2ca02c", "Getting worn": "#fd8d3c", "Replace now": "#d62728"}
        fig_gear = px.bar(
            gear_df, x="name", y="total_disp", color="status",
            color_discrete_map=_color_map_gear,
            labels={"name": "Shoe / Gear", "total_disp": f"Total {_g_unit}"},
            title=f"Total mileage per shoe ({_g_unit})",
        )
        fig_gear.add_hline(y=WARN_D, line_dash="dot", line_color="#fd8d3c",
                           annotation_text=f"Warn ({WARN_D:.0f} {_g_unit})", annotation_position="top right")
        fig_gear.add_hline(y=RETIRE_D, line_dash="dash", line_color="#d62728",
                           annotation_text=f"Retire ({RETIRE_D:.0f} {_g_unit})", annotation_position="top right")
        fig_gear.update_layout(showlegend=True, xaxis_tickangle=-20)
        st.plotly_chart(fig_gear, use_container_width=True)

        # Monthly mileage per shoe (recent period)
        if "gear_id" in df_range.columns:
            d_gear_range = df_range[pd.notna(df_range["gear_id"]) & (df_range["gear_id"] != "")].copy()
            if len(d_gear_range) > 0:
                d_gear_range["gear_id"] = d_gear_range["gear_id"].astype(str)
                if gear_names_map:
                    d_gear_range["shoe"] = d_gear_range["gear_id"].map(lambda g: gear_names_map.get(g, g))
                else:
                    d_gear_range["shoe"] = d_gear_range["gear_id"]
                d_gear_range["month"] = d_gear_range["start_dt_local"].dt.to_period("M").astype(str)
                monthly_gear = d_gear_range.groupby(["month", "shoe"])["distance_km"].sum().reset_index()
                monthly_gear["distance_disp"] = monthly_gear["distance_km"] * _g_factor
                fig_monthly_gear = px.bar(
                    monthly_gear, x="month", y="distance_disp", color="shoe",
                    labels={"month": "Month", "distance_disp": _g_unit, "shoe": "Shoe"},
                    title=f"Monthly {_g_unit} per shoe (selected period)",
                    barmode="stack",
                )
                st.plotly_chart(fig_monthly_gear, use_container_width=True)

        # Raw table
        with st.expander("Full gear table"):
            _show_cols = ["name", "total_disp", "runs", "first_used", "last_used"]
            st.dataframe(
                gear_df[_show_cols].rename(columns={
                    "name": "Shoe", "total_disp": f"Total {_g_unit}", "runs": "Runs",
                    "first_used": "First use", "last_used": "Last use",
                }),
                use_container_width=True,
            )


with explanation_tab:
    st.subheader("📚 Metrics Guide (How to read this dashboard)")
    st.caption(
        "These are explainable, HR-based training analytics. Use this tab to understand what each metric means, how it’s computed, and how to interpret it."
    )

    def _bullets(items):
        return "\n".join([f"- {x}" for x in items])

    # ---------- Card: Raw Streams ----------
    with st.expander("Raw Streams — per-second sensor data", expanded=False):
        st.markdown(
            """
**What this tab is for**  
This is the “raw truth” layer: per-sample signals (often 1 Hz) from Strava streams. Use it to validate anything you see in later tabs.

**Key streams you’ll see**
- **time (s)**: elapsed seconds since start.
- **distance / distance_km**: cumulative distance.
- **velocity_smooth (m/s)**: GPS-smoothed speed.
- **pace_min_per_km**: derived from speed; lower is faster.
- **heartrate (bpm)**: sensor readings when available.
- **cadence / grade_smooth / altitude / lat/lng**: present depending on device + Strava export.

**How derived fields are calculated**
- `distance_km = distance / 1000`
- `pace_min_per_km = (1000 / velocity_smooth) / 60`

**How to interpret**
- Look for **trends**, not noise: pacing drift, HR drift, surges, stop/start artifacts.
- **Stable pace + rising HR** → cardiovascular drift (heat, dehydration, fatigue).
- **Rising pace (slower) + rising HR** → fatigue or poor fueling/pacing.
- Pace is inverted on plots (lower is better).
"""
        )

    # ---------- Card: Training Load ----------
    with st.expander("Training Load — volume, intensity & fitness trend", expanded=False):
        st.markdown(
            """
**Goal**  
Quantify “how much stress” you’re absorbing and how quickly training is changing.

**Core metric: HR intensity**
- **What:** Relative effort.
- **Calc:** `hr_intensity = avg_hr / max_hr`
- **Interpretation:** Higher = harder. Accuracy depends on realistic max HR.

**Daily HR load**
- **What:** Stress for the day (duration × intensity).
- **Calc:** `daily_load = duration_min × hr_intensity`
- **Interpretation:** 60 min @ 0.75 ≈ 45 min @ 1.00 in load terms.

**Acute load (7d EWMA)**
- **What:** Short-term fatigue proxy.
- **Calc:** EWMA of daily load over ~7 days.
- **Interpretation:** Rising fast = accumulating fatigue; falling = recovery.

**Chronic load (28d EWMA)**
- **What:** Longer-term fitness proxy.
- **Calc:** EWMA of daily load over ~28 days.
- **Interpretation:** Gradual rise = sustainable build.

**ACWR (Acute:Chronic Workload Ratio)**
- **What:** Change management proxy.
- **Calc:** `acwr = acute_load / chronic_load`
- **Interpretation:**  
  - ~0.8-1.3: generally “balanced”  
  - greater than 1.5: higher risk of overreaching (heuristic, not medical advice)

**Race-normalized volume**
- **Weekly distance / race distance**: `weekly_km / race_km`  
- **Long run % of race**: `longest_run_km / race_km`  
**Interpretation:** Helps compare readiness across race distances.

**Monotony & strain**
- **Monotony:** mean daily load / std(daily load) within the week  
- **Strain:** weekly load × monotony  
**Interpretation:** High monotony means low variation; high strain = big week + little variation → watch recovery.
"""
        )

    # ---------- Card: Pace & Efficiency ----------
    with st.expander("Pace & Efficiency — getting faster at the same effort", expanded=False):
        st.markdown(
            """
**Goal**  
Track fitness changes while controlling for effort.

**Pace vs HR scatter**
- **What:** Output (pace) vs physiological effort (HR).
- **Interpretation:** Over time, “better” usually shifts toward faster paces at similar HR.
- **Pitfalls:** Hills, wind, heat, and GPS error increase scatter.

**Race-effort HR band**
- **What:** Filter for comparable intensity (e.g., HM effort).
- **Calc:** `%maxHR` range picked in sidebar.
- **Interpretation:** Compare runs inside the band to avoid “easy day vs workout” bias.

**Efficiency index (speed / HR)**
- **What:** Output per cardiovascular cost.
- **Calc:** `speed_per_hr = avg_speed_mps / avg_hr`
- **Interpretation:** Higher = better economy/fitness at similar conditions.
- **Pitfalls:** Sensitive to heat, fatigue, dehydration, terrain.

**Rolling medians/trends**
- **What:** Smoothed view of noisy field data.
- **Interpretation:** Rising efficiency trend = improving; falling trend can signal fatigue/overreach.
"""
        )

    # ---------- Card: Long Runs ----------
    with st.expander("Long Runs — durability & fatigue within each run", expanded=False):
        st.markdown(
            """
**Goal**  
Measure durability: how well you hold pace/efficiency as the run progresses.

**Pace fade**
- **What:** How much slower you get later in the run.
- **Calc:** `(pace_second_half - pace_first_half) / pace_first_half`
- **Interpretation:**  
  - Small fade (<~3–5%) = strong pacing/durability  
  - Larger fade can indicate fatigue, pacing error, fueling issues

**HR drift**
- **What:** HR increase later in the run at similar pace.
- **Calc:** `(hr_second_half - hr_first_half) / hr_first_half`
- **Interpretation:** Elevated drift often correlates with heat, dehydration, or low aerobic base.

**Decoupling (efficiency loss)**
- **What:** Drop in (speed/HR) from first to second half.
- **Calc:** `( (speed/hr)_second / (speed/hr)_first ) - 1`
- **Interpretation:** Negative = efficiency decline; closer to 0 = better aerobic durability.

**Run inspector plot**
- Pace & HR vs distance helps you visually confirm whether fatigue was gradual, abrupt, or due to stops/terrain.
"""
        )

    # ---------- Card: Recovery & Risk ----------
    with st.expander("Recovery & Risk — overreach and injury-risk proxies", expanded=False):
        st.markdown(
            """
**Goal**  
Provide decision-support indicators for recovery and overuse risk.

**Rest days (last 7)**
- **What:** How much true downtime you’ve had.
- **Calc:** Count of days with ~0 HR load in last 7 days.
- **Interpretation:** Consistently low rest → watch cumulative fatigue.

**Daily risk score (0–100)**
- **What:** Composite proxy based on workload + recovery signals.
- **Inputs:** ACWR, acute load relative to baseline, low rest.
- **Interpretation:**  
  - Lower = likely fresher  
  - Higher = more caution (not medical advice)

**Flags**
- ACWR high / very high  
- Low rest  
- “Big day” (outlier daily load vs recent distribution)  
**Interpretation:** Multiple flags close together = consider backing off.

**Weekly patterns**
- **Load change vs 4-week avg:** highlights spikes (often where injuries happen).
- **Load vs monotony (bubble size = strain):** big bubbles at high monotony = stressful weeks with little variation.
"""
        )

    # ---------- Card: Race Predictor ----------
    with st.expander("Race Predictor — how the prediction is built", expanded=False):
        st.markdown(
            """
**Goal**  
Estimate race-day time using recent efforts, distance scaling, efficiency trend, and a small readiness penalty.

**Minimum effort distance**
- **What:** Excludes noisy short runs from prediction.
- **Interpretation:** Shorter runs can be too stochastic (GPS/HR ramp/intervals).

**Race-effort run selection**
- Uses the race-effort HR band to keep efforts comparable.

**Distance scaling (Riegel-style power law)**
- **Idea:** Predict time at distance D2 from effort at D1:
  `T2 = T1 × (D2 / D1)^k`
- **Interpretation:** Good for endurance extrapolation when using representative efforts.
- **Pitfalls:** Less reliable if the input run is not steady-state (intervals, lots of stops).

**Efficiency adjustment**
- Uses the trend in speed/HR to nudge prediction based on recent fitness direction.

**Risk penalty**
- Small conservative penalty if Tab 4 indicates higher load/risk (reduces overconfidence).

**How to interpret the result**
- Treat it as a conditional estimate: “If conditions match recent training and I’m not overreached, this is plausible.”
"""
        )