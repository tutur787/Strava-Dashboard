"""
config.py — constants only, no Streamlit calls.
"""
from typing import Dict, Tuple

import numpy as np

# File paths
ACTIVITIES_PATH = "data/strava_runs_detailed.json"
STREAMS_PATH = "data/strava_runs_streams.json"
STRAVA_CACHE_PATH = "data/strava_oauth_cache.json"

# Race presets
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

# Run type classification
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

KM_TO_MILES = 0.621371

# Default HR zones (as fractions of max HR)
_DEFAULT_HR_ZONES = [
    ("Z1 Recovery",  0.00, 0.60),
    ("Z2 Aerobic",   0.60, 0.75),
    ("Z3 Tempo",     0.75, 0.87),
    ("Z4 Threshold", 0.87, 0.93),
    ("Z5 VO\u2082max",   0.93, 9.99),
]

# Strava OAuth URLs
STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"
