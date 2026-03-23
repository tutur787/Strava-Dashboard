"""
data_loader.py — load/parse/fetch/cache activities and streams.
"""
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import ACTIVITIES_PATH, STRAVA_CACHE_PATH, STREAMS_PATH, WORKOUT_TYPE_MAP


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


# ── Weather helper ────────────────────────────────────────────────────
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
                    # Try within +-2 hours
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
