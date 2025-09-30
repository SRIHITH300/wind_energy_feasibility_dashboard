# Wind Energy Feasibility Dashboard - Streamlit Prototype

## Run locally
1. Create venv:
   python -m venv .venv
   source .venv/bin/activate  # (Linux / macOS)
   .venv\Scripts\activate     # (Windows)

2. Install:
   pip install -r requirements.txt

3. Run:
   streamlit run streamlit_app.py

## Input data format (CSV)
- Two columns required: `timestamp`, `wind_speed`
- `timestamp` should be ISO-like (e.g., 2023-01-01 00:00:00)
- `wind_speed` in meters per second (m/s)
- Example row:
  2023-01-01 00:00:00, 5.6

You can upload your CSV or use the included sample dataset (generated synthetic hourly wind speeds).

## What the app does
- Shows exploratory charts (time series, histogram)
- Runs ARIMA and RandomForest forecasts
- Computes expected energy using a user-configurable turbine power curve
- Exports forecast & summary as CSV

## Notes
- This is a prototype intended for demonstration and early-stage site screening.
- For production use, integrate authenticated API calls (NOAA), perform data quality checks, and include uncertainty quantification and local terrain adjustments.

