import json
import os
import time
from datetime import datetime, timezone, timedelta
from stravalib import Client

STREAM_TYPES = ["time", "latlng", "distance", "altitude", "velocity_smooth", "heartrate", "cadence", "watts"]

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def _parse_dt_utc(value):
    """
    Accepts:
      - 'YYYY-MM-DD'
      - 'YYYY-MM-DD HH:MM:SS+00:00'
      - datetime
    Returns: aware datetime in UTC
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        # allow bare date
        if len(s) == 10:
            dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)

def _latest_activity_datetime_utc(detailed_activities):
    latest = None
    for a in detailed_activities:
        dt = _parse_dt_utc(a.get("start_date"))  # start_date is UTC
        if dt and (latest is None or dt > latest):
            latest = dt
    return latest

def _streams_to_dict(streams):
    out = {}
    for stream_type, stream in streams.items():
        out[stream_type] = {
            "data": getattr(stream, "data", None),
            "original_size": getattr(stream, "original_size", None),
            "resolution": getattr(stream, "resolution", None),
            "series_type": getattr(stream, "series_type", None),
        }
    return out

def fetch_new_activities(
    client: Client,
    detailed_path: str = "data/strava_runs_detailed.json",
    streams_path: str = "data/strava_runs_streams.json",
    athlete_path: str = "data/athlete.json",
    stream_types=STREAM_TYPES,
    resolution: str = "high",
    series_type: str = "time",
    sleep_s: float = 0.3,
    start_from: str | None = None,     # e.g. "2025-09-01" to rebuild from scratch
    runs_only: bool = True,
):
    athlete = _load_json(athlete_path, default=None)
    detailed = _load_json(detailed_path, default=[])
    streams_all = _load_json(streams_path, default=[])

    if athlete is None:
        athlete = client.get_athlete()
        _save_json(athlete_path, athlete.model_dump())

    detailed_ids = {int(a["id"]) for a in detailed if "id" in a}
    streams_ids = {int(item["activity_id"]) for item in streams_all if "activity_id" in item}

    # Decide where to start
    if start_from is not None:
        after_dt = _parse_dt_utc(start_from)
    else:
        after_dt = _latest_activity_datetime_utc(detailed) or _parse_dt_utc(athlete.get("created_at"))

    # Small overlap buffer to avoid missing edge cases (timezone / identical timestamps)
    after_dt = after_dt - timedelta(minutes=2)

    print(f"Fetching activities from {after_dt}")

    new_summaries = list(client.get_activities(after=after_dt))

    print(f"Found {len(new_summaries)} activities")

    n_details_added = 0
    n_streams_added = 0

    for idx, summary in enumerate(new_summaries, start=0):
        # optional type filter
        if runs_only and str(summary.type) != "root=\'Run\'":
            print(f"here (not run) {summary.type}")
            continue

        activity_id = int(summary.id)

        print(f"Activity ID: {activity_id}")

        if activity_id not in detailed_ids:
            print("here (not in detailed_ids)")
            try:
                full = client.get_activity(activity_id)
                detailed.append(full.model_dump())
                detailed_ids.add(activity_id)
                n_details_added += 1
                print(f"[{idx}/{len(new_summaries)}] added detail {activity_id}")
            except Exception as e:
                print(f"[{idx}/{len(new_summaries)}] FAILED detail {activity_id}: {e}")
            time.sleep(sleep_s)

        if activity_id not in streams_ids:
            try:
                streams = client.get_activity_streams(
                    activity_id=activity_id,
                    types=list(stream_types),
                    resolution=resolution,
                    series_type=series_type,
                )
                streams_all.append({"activity_id": activity_id, "streams": _streams_to_dict(streams)})
                streams_ids.add(activity_id)
                n_streams_added += 1
                print(f"[{idx}/{len(new_summaries)}] added streams {activity_id}")
            except Exception as e:
                print(f"[{idx}/{len(new_summaries)}] FAILED streams {activity_id}: {e}")
            time.sleep(sleep_s)

    _save_json(detailed_path, detailed)
    _save_json(streams_path, streams_all)

    print(f"Done. Added {n_details_added} detailed activities and {n_streams_added} stream sets.")
    return n_details_added, n_streams_added