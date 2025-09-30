# Wind Energy Feasibility Dashboard (polished version)
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import io
from datetime import timedelta

st.set_page_config(page_title="Wind Feasibility Dashboard", layout="wide")

st.title("Wind Energy Feasibility Dashboard")
st.caption("A tool for assessing wind project viability, featuring location selection, forecasting, energy modeling, and green planning skills.")

# Location selection and data source
st.header("Site Selection")
col_loc1, col_loc2 = st.columns([2,1])
with col_loc1:
    location = st.text_input("Enter site address/coordinates or describe the site:", value="Sample Site")
with col_loc2:
    m = folium.Map(location=[20.0, 78.0], zoom_start=5)
    map_data = st_folium(m, width=300, height=260)
    coords = None
    if map_data and map_data['last_clicked']:
        coords = map_data['last_clicked']
        st.write(f"Selected location: {coords}")

data_choice = st.radio("Choose wind data source:", ["Sample Data", "Upload CSV"], horizontal=True)
if data_choice == "Upload CSV":
    uploaded = st.file_uploader("Upload wind data CSV (timestamp, windspeed)", type="csv")
else:
    uploaded = None

# Sample data generation
def generate_sample_data(hours=336, base_speed=6.0):
    now = pd.Timestamp.now()
    rng = pd.date_range(end=now, periods=hours, freq="H")
    noise = np.random.normal(0, 1.5, size=len(rng))
    diurnal = np.sin(2 * np.pi * np.arange(len(rng)) / 24) * 0.8
    seasonal = np.sin(2 * np.pi * np.arange(len(rng)) / (24*30)) * 1.2
    speeds = np.clip(base_speed + diurnal + seasonal + noise, 0, None)
    return pd.DataFrame({"timestamp": rng, "windspeed": speeds})

if uploaded:
    df_raw = pd.read_csv(uploaded)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
else:
    hours_sample = st.number_input("Sample Data Hours", 24, 720, 336, 24)
    base_speed_sample = st.slider("Base mean wind speed (m/s)", 1.0, 12.0, 6.0)
    df_raw = generate_sample_data(hours=int(hours_sample), base_speed=float(base_speed_sample))

df_raw = df_raw.dropna(subset=["timestamp", "windspeed"])
st.write(f"Loaded {len(df_raw)} data points from {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")

# Exploratory analysis
st.header("Exploratory Analysis")
col_e1, col_e2 = st.columns([2,1])
with col_e1:
    st.write("Wind Speed Time Series")
    fig_ts = px.line(df_raw, x="timestamp", y="windspeed", title="Hourly Wind Speeds (m/s)")
    st.plotly_chart(fig_ts, use_container_width=True)
with col_e2:
    st.write("Histogram")
    fig_hist = px.histogram(df_raw, x="windspeed", nbins=30, title="Wind Speed Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# Statistical Forecasting Models
st.header("Wind Speed Forecasting")
forecast_horizon = st.number_input("Forecast horizon (hours)", 1, 168, 48)
arima_order = (st.number_input("ARIMA p", 0, 5, 2), st.number_input("ARIMA d", 0, 2, 0), st.number_input("ARIMA q", 0, 5, 2))
rf_lags = st.number_input("RF lag features", 3, 72, 24)
rf_trees = st.number_input("RF trees", 10, 500, 100, 10)

series = df_raw.set_index("timestamp")["windspeed"]

def fit_arima(series, steps, order):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.get_forecast(steps=steps)
    idx = pd.date_range(start=series.index[-1] + timedelta(hours=1), periods=steps, freq="H")
    return pd.Series(forecast.predicted_mean.values, index=idx)

def make_rf_features(series, lags):
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag{lag}"] = series.shift(lag)
    return df.dropna()

def fit_rf(series, steps, lags, n_estimators):
    df_feat = make_rf_features(series, lags)
    X = df_feat.drop("y", axis=1).values
    y = df_feat["y"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    last = X[-1]
    preds = []
    curr = last.copy()
    for _ in range(steps):
        preds.append(model.predict(curr.reshape(1, -1))[0])
        curr = np.roll(curr, 1)
        curr[0] = preds[-1]
    idx = pd.date_range(start=series.index[-1] + timedelta(hours=1), periods=steps, freq="H")
    return pd.Series(preds, index=idx)

arima_forecast = fit_arima(series, forecast_horizon, arima_order)
rf_forecast = fit_rf(series, forecast_horizon, rf_lags, rf_trees)

st.subheader("Forecast Plots")
df_forecasts = pd.DataFrame({
    "Observed": series.tail(48),
    "ARIMA": arima_forecast,
    "RF": rf_forecast
})
fig_fc = px.line(df_forecasts, title="Observed / Forecasted Wind Speeds")
st.plotly_chart(fig_fc, use_container_width=True)

# Turbine Power Curve Configuration
st.header("Turbine Power Curve & Energy Estimation")
turbine_capacity = st.number_input("Turbine rated capacity (kW)", 0.1, 1000.0, 100.0, 1.0)
curve_input = st.text_input("Power curve points (speed, kW); e.g. 3,1 5,7 10,60 15,135 25,0", value="3,1 5,7 10,60 15,135 25,0")
def parse_power_curve(text):
    points = [tuple(map(float, s.split(','))) for s in text.split() if ',' in s]
    d = dict(points)
    if 0.0 not in d:
        d[0.0] = 0.0
    return d
def interp_powercurve(curve):
    speeds = np.array(sorted(curve.keys()))
    kw = np.array([curve[s] for s in speeds])
    def interp(s):
        if s <= speeds[0]: return kw[0]
        if s >= speeds[-1]: return kw[-1]
        return np.interp(s, speeds, kw)
    return np.vectorize(interp)
pc_dict = parse_power_curve(curve_input)
power_func = interp_powercurve(pc_dict)

energy_source = st.selectbox("Wind speeds for energy estimate:", ["Observed", "ARIMA", "RF"])
if energy_source == "Observed":
    target_ws = series
elif energy_source == "ARIMA":
    target_ws = arima_forecast
else:
    target_ws = rf_forecast

energy_kwh = power_func(target_ws.values)
total_energy = energy_kwh.sum()
capacity_factor = total_energy / (turbine_capacity * len(target_ws)) if len(target_ws) > 0 else 0

st.metric("Total Energy (kWh)", f"{total_energy:.1f}")
st.metric("Hours Used", len(target_ws))
st.metric("Capacity Factor (%)", f"{capacity_factor*100:.2f}")

fig_energy = px.line(x=target_ws.index, y=energy_kwh, title="Estimated Energy per Hour (kWh)")
st.plotly_chart(fig_energy, use_container_width=True)

if st.checkbox("Enable CSV Export", value=True):
    out_df = pd.DataFrame({
        "timestamp": target_ws.index,
        "windspeed_used": target_ws.values,
        "energy_kwh": energy_kwh
    })
    csv_data = out_df.to_csv(index=False).encode()
    st.download_button("Download forecast & energy CSV", data=csv_data, mime="text/csv")


st.markdown("---")
st.caption("For production: integrate forecasting APIs, perform additional data checks, terrain corrections, and add known-site bias adjustment for advanced project assessment.")

# Green skills highlight
st.success("Green skills: project feasibility, sustainable infrastructure planning, and data-driven clean energy decisions are embedded.")
