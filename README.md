# Allure.run

A personal running analytics dashboard that connects to your Strava account and turns your training history into actionable insight. Built with Streamlit, powered by Supabase, visualised with Plotly.

---

## What it does

Allure analyses your complete running history across nine tabs:


| Tab                   | What it answers                                                               |
| --------------------- | ----------------------------------------------------------------------------- |
| **Overview**          | How fit am I right now? Personal bests, VO₂max estimate, consistency streaks  |
| **Training Load**     | Am I building or burning out? CTL, ATL, TSB, ACWR, HR zone breakdown          |
| **Pace & Efficiency** | Is my running economy improving? Speed/HR trend, GAP, pace distribution       |
| **Long Runs**         | How durable is my aerobic base? Aerobic decoupling, pace fade, HR drift       |
| **Recovery & Risk**   | Should I train hard today? Composite risk score, compromised run detection    |
| **Race Predictor**    | What can I race right now? Riegel-based prediction from best recent efforts   |
| **Gear**              | When do my shoes need replacing? Mileage tracking with retirement warnings    |
| **Guide**             | How does any of this work? Formula explanations with sources                  |
| **Raw Streams**       | What happened second-by-second on a specific run? Per-sample GPS, HR, cadence |


Every number on the dashboard is explained, sourced, and based on established exercise science — no black-box models.

---

## Key formulas

**Training load** — Full Banister TRIMP with heart-rate reserve:

```
HRr = (avg_hr − rest_hr) / (max_hr − rest_hr)
TRIMP = duration × HRr × 0.64 × e^(1.92 × HRr)   # Men
TRIMP = duration × HRr × 0.86 × e^(1.67 × HRr)   # Women
```

*Source: Banister et al. (1975), Morton et al. (1990)*

**Performance Management Chart** — Exponentially weighted moving averages of daily TRIMP:

- CTL (Fitness) = 28-day EWMA
- ATL (Fatigue) = 7-day EWMA
- TSB (Form) = CTL − ATL
- ACWR = ATL / CTL

**VO₂max / VDOT** — Jack Daniels' formula from best recorded effort:

```
VO₂ = −4.60 + 0.182258v + 0.000104v²
%VO₂max = 0.8 + 0.1894393·e^(−0.012778t) + 0.2989558·e^(−0.1932605t)
VDOT = VO₂ / %VO₂max
```

*Source: Daniels' Running Formula (2005)*

**Grade Adjusted Pace** — Minetti metabolic cost model:

```
C(g) = 280.5g⁵ − 58.7g⁴ − 76.8g³ + 51.9g² + 19.6g + 2.5
GAP = actual_pace × (C(0) / C(g))
```

*Source: Minetti et al. (2002), J Applied Physiology*

**Race prediction** — Riegel power law, median of top-5 qualifying efforts:

```
T₂ = T₁ × (D₂ / D₁)^1.06
```

*Source: Riegel (1981)*

**Aerobic decoupling** — Pa:HR metric (Friel):

```
decoupling = (speed/HR)_first_half / (speed/HR)_second_half − 1
< 5% = aerobic base held up well
```

---

## Architecture

```
app.py                  Orchestrator — auth, data loading, tab routing
├── ui/sidebar.py       All sidebar widgets, returns settings dict
├── ui/styles.py        Global CSS injection
├── analytics.py        All computations (TRIMP, VDOT, risk, GAP, etc.)
├── data_loader.py      Parse activities, disk cache, weather fetch
├── database.py         Supabase helpers (load/save athletes, activities, streams)
├── auth.py             Strava OAuth2 flow, token management
├── config.py           Constants — race presets, HR zones, file paths
└── tabs/
    ├── overview.py
    ├── training_load.py
    ├── pace.py
    ├── long_runs.py
    ├── recovery.py
    ├── race_predictor.py
    ├── gear.py
    ├── guide.py
    └── streams.py
```

**Stack:** Python · Streamlit · Supabase (PostgreSQL) · Plotly · Strava API

---

## Setup

### 1. Create a Strava API application

1. Go to [strava.com/settings/api](https://www.strava.com/settings/api)
2. Create an application
3. Set the **Authorisation Callback Domain** to `localhost` for local dev, or your deployed URL for production
4. Note your `Client ID` and `Client Secret`

### 2. Create a Supabase project

1. Create a free project at [supabase.com](https://supabase.com)
2. Run the following in the **SQL Editor**:

```sql
CREATE TABLE athletes (
    athlete_id  BIGINT PRIMARY KEY,
    display_name TEXT,
    refresh_token TEXT,
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    preferences JSONB DEFAULT '{}'
);

CREATE TABLE activities (
    athlete_id  BIGINT,
    activity_id BIGINT,
    data        JSONB,
    PRIMARY KEY (athlete_id, activity_id)
);

CREATE TABLE streams (
    athlete_id  BIGINT,
    activity_id BIGINT,
    data        JSONB,
    PRIMARY KEY (athlete_id, activity_id)
);

CREATE INDEX idx_activities_athlete ON activities(athlete_id);
CREATE INDEX idx_streams_athlete    ON streams(athlete_id);

-- Row Level Security (blocks direct anon-key access)
ALTER TABLE athletes  ENABLE ROW LEVEL SECURITY;
ALTER TABLE activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE streams    ENABLE ROW LEVEL SECURITY;
```

1. Find your **Project URL** and **Service Role Key** under Project Settings → API

### 3. Configure secrets

Create `.streamlit/secrets.toml` in the project root:

```toml
[strava]
client_id     = "your_strava_client_id"
client_secret = "your_strava_client_secret"
redirect_uri  = "http://localhost:8501"

[supabase]
url = "https://your-project.supabase.co"
key = "your_service_role_key"
```

> **Never commit this file.** It is already listed in `.gitignore`.

### 4. Install dependencies and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), click **Connect with Strava**, and authorise the app. Your activities will sync automatically.

---

## Sidebar settings


| Setting                  | Effect                                                        |
| ------------------------ | ------------------------------------------------------------- |
| **Max HR**               | Drives all HR zone calculations and TRIMP intensity           |
| **Resting HR**           | Used in heart-rate reserve for full Banister TRIMP            |
| **Biological sex**       | Selects Banister's sex-specific exponential coefficients      |
| **HR Zone Boundaries**   | Adjustable % thresholds; actual bpm shown live                |
| **Target race distance** | Sets the reference for long-run threshold and race prediction |
| **Race-effort HR band**  | Defines the intensity window used in Pace & Efficiency        |
| **Long-run threshold**   | Minimum distance (% of race) to qualify as a long run         |
| **Readiness window**     | Lookback period for Recovery & Risk tab                       |
| **Prediction lookback**  | How far back Race Predictor searches for qualifying efforts   |


Settings are saved per user to Supabase and restored automatically on next login.

---

## Multi-user support

The app is designed for small groups (up to ~20 users). Each user authenticates independently via Strava OAuth. Data is stored and queried per `athlete_id` — no user can access another's data. Row Level Security on Supabase tables blocks any direct API access via the anon key.

---

## Requirements

See `requirements.txt`. Key dependencies:

```
streamlit
supabase
plotly
pandas
numpy
requests
streamlit-cookies-controller
```

