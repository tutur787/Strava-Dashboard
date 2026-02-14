# Endurance Analytics Dashboard

This project is a dashboard that allows you to analyze your endurance activities and predict your performance in races.

It is built with Streamlit and uses the Stravalib library to interact with the Strava API.

## Installation

1. Clone the repository
2. Install the dependencies `pip install -r requirements.txt`
3. Read the `Initialization of data` section to get started
4. Run the dashboard with `streamlit run app.py`
5. Open the browser and go to http://localhost:8501

## Initialization of data

This project pulls activity data from the Strava API and stores it locally as JSON files.  
Before fetching activities, you must authenticate with Strava and obtain a valid access token.

### Step 1: Create a Strava API application
1. Go to https://www.strava.com/settings/api
2. Create an application and note:
   - `CLIENT_ID`
   - `CLIENT_SECRET`

### Step 2: Authenticate and obtain tokens
Run the main script and choose **option 0 (authenticate)**.

You will be prompted for:
- `CLIENT_ID`
- `CLIENT_SECRET`

The script will:
1. Print an authorization URL
2. Open the URL in your browser and authorize the app
3. Redirect you to a URL containing a `code`
4. Prompt you to paste that `code` back into the terminal

You will receive:
- `ACCESS_TOKEN`
- `REFRESH_TOKEN`

Save both securely. The access token is required for data fetching.

### Step 3: Refresh an expired access token
Strava access tokens expire after 6 hours.  
To refresh a token, run the script and choose **option 1 (refresh token)**.

You will be prompted for:
- `CLIENT_ID`
- `CLIENT_SECRET`
- `REFRESH_TOKEN`

The script will return:
- a new `ACCESS_TOKEN`
- a new `REFRESH_TOKEN`

### Step 4: Initialize local data files
To fetch activity data and initialize local storage, run the script and choose **option 2 (fetch new activities)**.

On the first run:
- You will be prompted for an `ACCESS_TOKEN`
- You will be prompted for a start date (`YYYY-MM-DD`)
- All activities after that date will be downloaded

The following files will be created automatically if they do not exist:
- `data/athlete.json` — athlete profile metadata
- `data/strava_runs_detailed.json` — detailed activity objects
- `data/strava_runs_streams.json` — high-resolution activity streams

On subsequent runs:
- Only activities newer than the most recent stored activity are fetched
- Existing activities and streams are not duplicated

## How it works

This dashboard is an interactive endurance training analytics platform built with Streamlit, using Strava activity and stream data to analyze training load, performance, fatigue, readiness, and race outcomes in a transparent, explainable way.

The app is designed to be:
- Reusable across athletes and race distances (5K → Marathon)
- HR-based (no power meter required)
- Explainable (rule-based and analytical, not a black-box model)
- Portfolio-ready, showcasing data engineering, time-series analysis, and sports analytics concepts

All computations are derived from standard Strava fields such as distance, time, heart rate, speed, and per-second streams.

### Architecture
- Frontend: Streamlit
- Data source: Strava activity exports (JSON)
- `strava_runs_detailed.json` (per-activity summaries)
- `strava_runs_streams.json` (per-sample streams: HR, pace, distance, etc.)
- Caching: `@st.cache_data` for efficient recomputation
- Visualization: Plotly (interactive, zoomable charts)

User-controlled parameters (Max HR, race distance, HR bands, readiness window, etc.) are shared across all tabs to keep analyses consistent.

### Tabs overview

#### Tab 0 — Streams Explorer

An exploratory tool to inspect raw per-sample streams for individual activities.
- Select an activity by date – name – distance
- Convert Strava streams into a tidy dataframe
- Plot any numeric metric (HR, pace, cadence, grade, etc.)
- Choose time or distance as the x-axis
- Optional rolling smoothing
- Pace plots are automatically inverted (lower = faster)

This tab provides full transparency into the underlying data used by later models.

#### Tab 1 — Training Load

Quantifies training volume and intensity over time using HR-based load.
- Daily and weekly load (duration × HR intensity)
- Acute (7-day) vs chronic (28-day) load
- ACWR (Acute:Chronic Workload Ratio)
- Weekly distance and long-run normalization by race distance
- Training monotony and strain

This tab answers: “How much am I training, and how fast is it changing?”

#### Tab 2 — Pace & Efficiency

Evaluates running efficiency at comparable effort.
- Pace vs heart rate scatter
- Pace distribution
- Efficiency index (speed / HR)
- Trend analysis within a race-effort HR band
- HR-normalized pace comparisons

This isolates fitness changes from effort changes.

#### Tab 3 — Long-Run Fatigue

Uses per-sample streams to model within-run fatigue on long runs.
- Pace fade (second half vs first half)
- Heart-rate drift
- HR–pace decoupling
- Long-run trend tracking over time
- Interactive inspection of individual long runs

This tab focuses on endurance durability rather than raw speed.

#### Tab 4 — Readiness & Risk

Provides rule-based readiness and injury-risk proxies using recent training history.
- Composite daily risk score (0–100)
- Rest-day tracking
- ACWR alerts
- Weekly load spikes and monotony
- “Compromised run” detection based on efficiency outliers

These are decision-support indicators, not medical advice.

#### Tab 5 — Race Prediction

Generates an explainable race-day time prediction.
- Uses recent race-effort runs above a minimum distance
- Applies distance scaling (Riegel-style power law)
- Adjusts for efficiency trend
- Applies a small penalty when readiness risk is elevated
- Shows which runs influenced the prediction

No black-box ML — every assumption is visible and tunable.

#### Tab 6 — Explanations & Interpretations

Provides explanations and interpretations for the various metrics and calculations.
- Explanations for the various metrics and calculations
- Explanations for the various charts and plots
- Explanations for the various tables and data
- Explanations for the various models and algorithms
- Explanations for the various assumptions and limitations

### Why this project

This project demonstrates:
- Real-world data wrangling (nested JSON, time-series, streams)
- Feature engineering from physiological signals
- Analytical modeling under noisy conditions
- Clear, user-driven explainability
- Practical sports analytics design

It's intended both as a training tool for runners and a portfolio project for data science and sports analytics roles.
