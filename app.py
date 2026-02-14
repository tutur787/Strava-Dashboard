import json
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Endurance Analytics Dashboard",
    page_icon="🏃",
    layout="wide",
)

ACTIVITIES_PATH = "data/strava_runs_detailed.json"
STREAMS_PATH = "data/strava_runs_streams.json"

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
    np.nan: "Unspecified",
}


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_activities(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)

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
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    df["start_dt_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")
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

    return df


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
    mask = (activities["start_dt_local"] >= start) & (activities["start_dt_local"] <= end)
    df = activities.loc[mask].copy()

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

    return daily_full, weekly


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


def _format_min_per_km(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    m = int(x)
    s = int(round((x - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}/km"


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
# Sidebar controls (shared)
# -----------------------------
st.title("🏃 Endurance Analytics Dashboard")

with st.sidebar:
    st.header("Controls")

    max_hr = st.number_input(
        "Max HR (bpm)",
        min_value=120,
        max_value=230,
        value=180,
        step=1,
        help="Used to compute intensity = avgHR / maxHR.",
    )

    race_choice = st.selectbox("Target race distance", list(RACE_PRESETS_KM.keys()), index=2)
    if race_choice == "Custom (km)":
        race_km = st.number_input("Custom race distance (km)", min_value=1.0, max_value=200.0, value=21.0975, step=0.5)
    else:
        race_km = float(RACE_PRESETS_KM[race_choice])

    lo_def, hi_def = RACE_EFFORT_DEFAULTS.get(race_choice, (0.84, 0.92))
    st.subheader("Race-effort HR band")
    effort_band = st.slider(
        "HR intensity range (% of max HR)",
        min_value=0.50,
        max_value=1.00,
        value=(float(lo_def), float(hi_def)),
        step=0.01,
        help="Used in Tab 2 to track efficiency at race-relevant intensity.",
    )

    lr_def = float(LONG_RUN_DEFAULTS.get(race_choice, 0.60))
    st.subheader("Long-run threshold")
    long_run_ratio_thresh = st.slider(
        "Minimum long-run distance (% of race distance)",
        min_value=0.30,
        max_value=1.00,
        value=float(lr_def),
        step=0.05,
        help="Tab 3 considers runs longer than this threshold as 'long runs' for fatigue modeling.",
    )

    st.subheader("Readiness windows")
    readiness_window_days = st.slider(
        "Readiness lookback (days)",
        min_value=14,
        max_value=90,
        value=42,
        step=7,
        help="Tab 4 uses this window to summarize recent risk/readiness.",
    )

    st.subheader("Prediction window")
    prediction_lookback_days = st.slider(
        "Race prediction lookback (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=15,
        help="Tab 5 uses this window to predict race time.",
    )

    st.divider()
    st.subheader("Filters")
    only_runs = st.checkbox("Only running activities", value=True)
    exclude_manual = st.checkbox("Exclude manual activities", value=True)
    exclude_trainer = st.checkbox("Exclude trainer/treadmill", value=False)

# -----------------------------
# Load & filter activities
# -----------------------------
activities = load_activities(ACTIVITIES_PATH)
streams_by_id = load_streams(STREAMS_PATH)

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

date_range = st.date_input(
    "Date range",
    value=(default_start.date(), default_end.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start.date(), default_end.date()

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

mask = (activities["start_dt_local"] >= start_ts) & (activities["start_dt_local"] <= end_ts)
df_range = activities.loc[mask].copy()

if len(df_range) == 0:
    st.info("No activities in the selected range.")
    st.stop()

# Precompute daily/weekly for shared tabs (avoid recompute)
daily_all, weekly_all = build_daily_weekly(activities, max_hr=max_hr, date_range=(start_ts, end_ts))

# -----------------------------
# Tabs
# -----------------------------
tab_streams, tab1, tab2, tab3, tab4, tab5, explanation_tab = st.tabs([
    "Tab 0 — Streams explorer",
    "Tab 1 — Training Load",
    "Tab 2 — Pace & Efficiency",
    "Tab 3 — Long-Run Fatigue",
    "Tab 4 — Readiness & Risk",
    "Tab 5 — Race prediction",
    "Tab 6 — Explanations & Interpretations",
])
# =====================================================================
# TAB 0 - Streams explorer
# =====================================================================

with tab_streams:
    st.subheader("Strava Streams Explorer")
    st.caption("Inspect per-second/per-sample streams (pace, HR, cadence, grade, etc.) for individual activities.")

    # Ensure we have streams
    if not isinstance(streams_by_id, dict) or len(streams_by_id) == 0:
        st.warning("No streams found in streams file.")
    else:
        # Build a nice label: date — name — distance
        df_act = df_range.copy()
        df_act["date_str"] = df_act["start_dt_local"].dt.strftime("%Y-%m-%d")
        df_act["label"] = df_act["date_str"].astype(str) + " — " + df_act["name"].fillna("").astype(str).str.slice(0, 50) + " — " + df_act["distance_km"].fillna(0).map(lambda x: f"{x:.1f} km")

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
        st.info("Not enough data in the selected range to compute load.")
    else:

        weekly = weekly.copy()
        weekly["weekly_distance_ratio"] = weekly["weekly_distance_km"] / race_km
        weekly["long_run_ratio"] = weekly["long_run_km"] / race_km

        latest_week = weekly.sort_values("week_start").iloc[-1]
        latest_day = daily.sort_values("date_ts").iloc[-1]
        acwr_label, acwr_emoji = acwr_band(float(latest_day["acwr"]) if pd.notna(latest_day["acwr"]) else np.nan)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Weekly distance", f"{latest_week['weekly_distance_km']:.1f} km")
        k2.metric("Weekly HR load", f"{latest_week['weekly_load_hr']:.1f}", help="load = duration_min × (avgHR / maxHR)")
        k3.metric("ACWR", f"{latest_day['acwr']:.2f}" if pd.notna(latest_day["acwr"]) else "N/A", f"{acwr_emoji} {acwr_label}")
        k4.metric("Long run % of race", f"{latest_week['long_run_ratio']*100:.0f}%")

        st.caption("Load is computed only when average HR is present; distance/time still count toward volume charts.")

        st.subheader("Load trend (daily): Fitness vs fatigue proxy")
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=daily["date_ts"], y=daily["acute_load"], mode="lines", name="Acute load (7d EWMA)"))
        fig_load.add_trace(go.Scatter(x=daily["date_ts"], y=daily["chronic_load"], mode="lines", name="Chronic load (28d EWMA)"))
        fig_load.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Date",
            yaxis_title="HR Load (a.u.)",
        )
        st.plotly_chart(fig_load, use_container_width=True)

        st.subheader("Weekly progression")
        c1, c2 = st.columns([2, 1])

        with c1:
            fig_weekly_dist = px.bar(
                weekly,
                x="week_start",
                y="weekly_distance_km",
                title="Weekly distance (km)",
                labels={"week_start": "Week start", "weekly_distance_km": "km"},
            )
            fig_weekly_dist.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_weekly_dist, use_container_width=True)

        with c2:
            fig_ratios = go.Figure()
            fig_ratios.add_trace(go.Bar(x=weekly["week_start"], y=weekly["weekly_distance_ratio"], name="Weekly / race distance"))
            fig_ratios.add_trace(go.Bar(x=weekly["week_start"], y=weekly["long_run_ratio"], name="Long run / race distance"))
            fig_ratios.update_layout(
                barmode="group",
                height=340,
                title="Race-normalized volume",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Week start",
                yaxis_title="Ratio (× race distance)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_ratios, use_container_width=True)

        st.subheader("Stability & risk proxies (weekly)")
        r1, r2 = st.columns(2)
        with r1:
            fig_mono = px.line(
                weekly,
                x="week_start",
                y="monotony",
                title="Training monotony (mean daily load / std)",
                labels={"week_start": "Week start", "monotony": "Monotony"},
            )
            fig_mono.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_mono, use_container_width=True)

        with r2:
            fig_strain = px.line(
                weekly,
                x="week_start",
                y="strain",
                title="Training strain (weekly load × monotony)",
                labels={"week_start": "Week start", "strain": "Strain"},
            )
            fig_strain.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
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

        d2["speed_per_hr"] = d2["avg_speed_mps"] / d2["avg_hr"]
        d2["hr_per_km"] = d2["avg_hr"] * d2["pace_min_per_km"]

        d2 = d2.sort_values("start_dt_local")
        race_eff = d2[d2["in_race_effort_band"]].copy()

        top = st.columns(4)
        top[0].metric("Runs in range", f"{len(d2)}")
        top[1].metric("Runs in race-effort band", f"{len(race_eff)}", help=f"Band = {lo:.0%}–{hi:.0%} of Max HR")
        top[2].metric("Median pace", _format_min_per_km(float(d2["pace_min_per_km"].median())))
        top[3].metric("Median HR", f"{int(round(d2['avg_hr'].median()))} bpm")

        st.caption("The race-effort HR band is adjustable in the sidebar. It helps track efficiency at comparable intensity.")

        c1, c2 = st.columns([2, 1])

        with c1:
            fig_scatter = px.scatter(
                d2,
                x="avg_hr",
                y="pace_min_per_km",
                color="distance_km",
                hover_data=["name", "start_dt_local", "distance_km", "duration_min", "total_elevation_gain", "hr_intensity"],
                labels={"avg_hr": "Average HR (bpm)", "pace_min_per_km": "Pace (min/km)", "distance_km": "Distance (km)"},
                title="Pace vs HR (colored by distance)",
            )
            fig_scatter.update_yaxes(autorange="reversed")
            fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            fig_dist = px.histogram(
                d2,
                x="pace_min_per_km",
                nbins=25,
                title="Pace distribution",
                labels={"pace_min_per_km": "Pace (min/km)", "count": "Runs"},
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

        t1, t2c = st.columns(2)

        with t1:
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["speed_per_hr"], mode="markers", name="Speed/HR"))
            fig_eff.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["speed_per_hr_roll"], mode="lines", name="Rolling median (5 runs)"))
            fig_eff.update_layout(
                height=320,
                title="Efficiency index (speed ÷ HR)",
                xaxis_title="Date",
                yaxis_title="m/s per bpm",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        with t2c:
            fig_pace = go.Figure()
            fig_pace.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["pace_min_per_km"], mode="markers", name="Pace"))
            fig_pace.add_trace(go.Scatter(x=trend_df["start_dt_local"], y=trend_df["pace_roll"], mode="lines", name="Rolling median (5 runs)"))
            fig_pace.update_layout(
                height=320,
                title="Pace trend at comparable effort",
                xaxis_title="Date",
                yaxis_title="min/km",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig_pace.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_pace, use_container_width=True)

# =====================================================================
# TAB 3 - Long-Run Fatigue
# =====================================================================
with tab3:
    st.subheader("Long-run fatigue modeling (within-run)")

    long_run_min_km = float(long_run_ratio_thresh) * float(race_km)
    st.caption(f"Long runs are defined here as runs ≥ {long_run_min_km:.1f} km (based on your threshold × race distance).")

    fatigue = build_fatigue_table(df_range, streams_by_id, long_run_min_km=long_run_min_km)

    if len(fatigue) == 0:
        st.info("No long runs with available streams in the selected date range. Try lowering the long-run threshold or widening the date range.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Long runs (with streams)", f"{len(fatigue)}")
        k2.metric("Median pace fade", f"{np.nanmedian(fatigue['pace_fade_pct'])*100:.1f}%")
        k3.metric("Median HR drift", f"{np.nanmedian(fatigue['hr_drift_pct'])*100:.1f}%" if np.isfinite(np.nanmedian(fatigue['hr_drift_pct'])) else "N/A")
        k4.metric("Median decoupling", f"{np.nanmedian(fatigue['decoupling'])*100:.1f}%" if np.isfinite(np.nanmedian(fatigue['decoupling'])) else "N/A")

        st.subheader("Fatigue trends over time")
        t1, t2 = st.columns(2)
        with t1:
            fig_pf = px.line(
                fatigue,
                x="start_dt_local",
                y="pace_fade_pct",
                markers=True,
                title="Pace fade (second half vs first half)",
                labels={"start_dt_local": "Date", "pace_fade_pct": "Pace fade (fraction)"},
            )
            fig_pf.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_pf, use_container_width=True)

        with t2:
            fig_hr = px.line(
                fatigue,
                x="start_dt_local",
                y="hr_drift_pct",
                markers=True,
                title="HR drift (second half vs first half)",
                labels={"start_dt_local": "Date", "hr_drift_pct": "HR drift (fraction)"},
            )
            fig_hr.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=run_df["distance_km"], y=run_df["pace_smooth"], mode="lines", name="Pace (smoothed)"))
        fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Distance (km)",
            yaxis_title="Pace (min/km)",
        )
        fig.update_yaxes(autorange="reversed")

        if np.any(np.isfinite(run_df["hr_smooth"])):
            fig.add_trace(go.Scatter(x=run_df["distance_km"], y=run_df["hr_smooth"], mode="lines", name="HR (smoothed)", yaxis="y2"))
            fig.update_layout(
                yaxis2=dict(title="Heart rate (bpm)", overlaying="y", side="right", showgrid=False)
            )

        st.plotly_chart(fig, use_container_width=True)

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
        c1.metric("Today risk score", f"{risk_score:.0f}/100", f"{readiness[1]} {readiness[0]}")
        c2.metric("ACWR (latest)", f"{acwr_val:.2f}" if pd.notna(acwr_val) else "N/A")
        c3.metric("Rest days (last 7)", f"{int(latest['rest_days_last7'])}")
        c4.metric("Acute load (7d EWMA)", f"{latest['acute_load']:.1f}")

        st.caption("These are *proxies*, not medical advice. They’re meant for portfolio-grade decision support with transparent assumptions.")

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

        # Flags timeline (count of flags per day)
        d_focus["flags_total"] = d_focus[["flag_acwr_high","flag_acwr_very_high","flag_low_rest","flag_big_day"]].sum(axis=1)
        fig_flags = px.bar(
            d_focus,
            x="date_ts",
            y="flags_total",
            title="Alert flags per day (rule-based)",
            labels={"date_ts":"Date", "flags_total":"# flags"},
        )
        fig_flags.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
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

        # Compromised run detector (optional)
        st.subheader("Compromised run detector (efficiency outliers)")
        comp = compute_compromised_runs(df_range, max_hr=max_hr)
        if len(comp) == 0 or comp["eff_q20"].isna().all():
            st.info("Not enough HR-bearing runs to detect outliers yet.")
        else:
            comp_recent = comp[comp["start_dt_local"] >= start_focus].copy()
            rate = 100.0 * comp_recent["compromised"].mean() if len(comp_recent) else np.nan

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Compromised runs (recent)", f"{int(comp_recent['compromised'].sum())}/{len(comp_recent)}", f"{rate:.0f}%")
                fig_comp = px.scatter(
                    comp_recent,
                    x="start_dt_local",
                    y="speed_per_hr",
                    color=comp_recent["compromised"].astype(str),
                    title="Efficiency over time (flagged outliers)",
                    labels={"start_dt_local":"Date", "speed_per_hr":"Speed/HR (m/s per bpm)", "color":"Compromised"},
                )
                fig_comp.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_comp, use_container_width=True)

            with c2:
                fig_delta = px.histogram(
                    comp_recent.dropna(subset=["eff_delta"]),
                    x="eff_delta",
                    nbins=25,
                    title="Efficiency delta vs rolling 20th percentile baseline",
                    labels={"eff_delta":"Speed/HR - baseline"},
                )
                fig_delta.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_delta, use_container_width=True)

            with st.expander("Show compromised-run table"):
                cols = ["start_dt_local","name","distance_km","duration_min","avg_hr","pace_min_per_km","speed_per_hr","eff_q20","eff_delta","compromised"]
                st.dataframe(comp_recent[cols].sort_values("start_dt_local", ascending=False), use_container_width=True)

# =====================================================================
# TAB 5 - Race prediction
# =====================================================================
with tab5:
    st.subheader("Race-day prediction (HR-based + performance history)")

    # Use a consistent data window for modeling (lookback relative to selected end date)
    model_end = end_ts
    model_start = model_end - pd.Timedelta(days=int(prediction_lookback_days))

    model_runs = activities[(activities["start_dt_local"] >= model_start) & (activities["start_dt_local"] <= model_end)].copy()
    model_runs = model_runs[model_runs["type"].astype(str).str.lower() == "run"].copy()

    st.caption(
        f"Using runs from the last {int(prediction_lookback_days)} days "
        f"({model_start.date()} → {model_end.date()}) to build a simple, explainable prediction."
    )

    colA, colB = st.columns([1.2, 1])
    with colA:
        exp = st.slider("Riegel exponent (fatigue scaling)", 1.02, 1.12, 1.06, 0.01,
                        help="Higher = performance drops off faster as distance increases. 1.06 is a common default.")
    with colB:
        min_dist = st.number_input("Minimum effort distance used (km)", min_value=1.0, max_value=20.0, value=3.0, step=0.5)

    # Baseline: best equivalent performance
    baseline_sec, source = predict_race_time_riegel(
        model_runs,
        target_km=float(race_km),
        exponent=float(exp),
        min_km=float(min_dist),
    )

    if baseline_sec is None or source is None:
        st.info("Not enough runs in this window to predict. Try increasing the lookback window or lowering the minimum distance.")
    else:
        # Adjustments
        eff_factor = compute_efficiency_adjustment(
            model_runs, max_hr=int(max_hr), effort_band=effort_band,
            lookback_days=int(max(30, prediction_lookback_days // 3)), end_ts=model_end
        )

        # Risk penalty from daily risk model (computed on the selected date range)
        daily_risk, weekly_risk = compute_risk_table(daily_all, weekly_all)
        risk_factor = compute_risk_penalty(daily_risk)

        pred_sec = baseline_sec * eff_factor * risk_factor

        # Interval using exponent range
        low_sec, _ = predict_race_time_riegel(model_runs, float(race_km), exponent=max(1.02, exp - 0.03), min_km=float(min_dist))
        high_sec, _ = predict_race_time_riegel(model_runs, float(race_km), exponent=min(1.12, exp + 0.03), min_km=float(min_dist))
        if low_sec is not None and high_sec is not None:
            lo = low_sec * eff_factor * risk_factor
            hi = high_sec * eff_factor * risk_factor
        else:
            lo, hi = None, None

        # Pace (min/km)
        pace_sec_per_km = pred_sec / float(race_km)
        pace_min = pace_sec_per_km / 60.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted finish time", format_hms(pred_sec))
        k2.metric("Predicted pace", f"{int(pace_min)}:{int(round((pace_min-int(pace_min))*60)):02d} /km")
        k3.metric("Efficiency adjustment", f"{eff_factor:.3f}×", help="Based on speed/HR trend within your race-effort HR band.")
        k4.metric("Risk penalty", f"{risk_factor:.3f}×", help="Small penalty based on latest composite risk score from Tab 4.")

        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            st.info(f"Prediction interval (sensitivity to exponent): **{format_hms(lo)} → {format_hms(hi)}**")

        st.subheader("What the model used")
        st.write(
            f"**Baseline source run:** {pd.to_datetime(source['start_dt_local']).strftime('%Y-%m-%d')} "
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

        with st.expander("Show prediction table"):
            show = df[["start_dt_local","name","distance_km","duration_min","equiv_time"]].copy()
            st.dataframe(show.sort_values("equiv_time"), use_container_width=True)

with explanation_tab:
    st.subheader("📚 Metrics Guide (How to read this dashboard)")
    st.caption(
        "These are explainable, HR-based training analytics. Use this tab to understand what each metric means, how it’s computed, and how to interpret it."
    )

    def _bullets(items):
        return "\n".join([f"- {x}" for x in items])

    # ---------- Card: Tab 0 ----------
    with st.expander("Tab 0 — Streams Explorer (raw per-sample data)", expanded=False):
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

    # ---------- Card: Tab 1 ----------
    with st.expander("Tab 1 — Training Load (volume + intensity over time)", expanded=False):
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

    # ---------- Card: Tab 2 ----------
    with st.expander("Tab 2 — Pace & Efficiency (getting faster at same effort)", expanded=False):
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

    # ---------- Card: Tab 3 ----------
    with st.expander("Tab 3 — Long-Run Fatigue (durability inside long runs)", expanded=False):
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

    # ---------- Card: Tab 4 ----------
    with st.expander("Tab 4 — Readiness & Risk (rule-based proxies)", expanded=False):
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

    # ---------- Card: Tab 5 ----------
    with st.expander("Tab 5 — Race Prediction (transparent, HR-based estimate)", expanded=False):
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