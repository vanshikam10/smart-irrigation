# dashboard.py
# Run: streamlit run dashboard.py
# polished irrigation dashboard — uses Open-Meteo for historical precipitation (no API key)
#f72abb2b23d5afc8aadcd5b4d2d6b2ca
# dashboard.py
# Run: streamlit run dashboard.py

from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import plotly.express as px
import uuid
import time

# =========================== App config & CSS ===========================
st.set_page_config(page_title="💧 Smart Irrigation — Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .title { font-size:26px; font-weight:700; color:#0b6e4f; }
      .subtitle { color:#556277; margin-bottom:12px; }
      .card { background: linear-gradient(180deg,#ffffff,#f7fbf8); border-radius:12px; padding:14px; box-shadow:0 6px 18px rgba(3,60,43,0.06); }
      .small { font-size:13px; color:#6b7280; }
      .metric { font-weight:700; font-size:20px; color:#0b6e4f; }
      .rec-green { background: linear-gradient(90deg,#e6f8ef,#e6fff3); border-left:6px solid #16a34a; padding:12px; border-radius:10px; }
      .rec-orange { background: linear-gradient(90deg,#fff7e6,#fffdf0); border-left:6px solid #f59e0b; padding:12px; border-radius:10px; }
      .rec-red { background: linear-gradient(90deg,#fff0f0,#fff6f6); border-left:6px solid #ef4444; padding:12px; border-radius:10px; }
      .pill { border: 1px solid #d1fae5; color:#047857; background:#ecfdf5; padding:2px 8px; border-radius:999px; font-size:12px; }
      .muted { color:#6b7280; font-size:12px; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="title">💧 Smart Irrigation Dashboard — Professional</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Place-aware, season-aware, explainable irrigation — Sensors + Forecast + 30-year history</div>', unsafe_allow_html=True)

# =========================== Defaults & Session ===========================
DEFAULT_THRESHOLDS = {
    "Paddy": {"moisture": 50.0, "temp": 28.0},
    "Wheat": {"moisture": 45.0, "temp": 22.0},
    "Groundnut": {"moisture": 40.0, "temp": 30.0},
    "Maize": {"moisture": 50.0, "temp": 25.0},
    "Cotton": {"moisture": 35.0, "temp": 30.0},
    "Sugarcane": {"moisture": 60.0, "temp": 26.0},
    "Mustard": {"moisture": 40.0, "temp": 22.0},
    "Barley": {"moisture": 40.0, "temp": 20.0},
    "Soybean": {"moisture": 45.0, "temp": 27.0},
    "Tomato": {"moisture": 55.0, "temp": 24.0},
    "Onion": {"moisture": 45.0, "temp": 24.0},
    "Potato": {"moisture": 45.0, "temp": 20.0},
    "Chickpea": {"moisture": 40.0, "temp": 20.0},
    "Lentil": {"moisture": 40.0, "temp": 20.0},
    "Pea": {"moisture": 40.0, "temp": 20.0},
    "Cabbage": {"moisture": 50.0, "temp": 22.0},
    "Cauliflower": {"moisture": 50.0, "temp": 22.0},
    "Spinach": {"moisture": 45.0, "temp": 20.0},
    "Green Beans": {"moisture": 45.0, "temp": 24.0},
    "Bell Pepper": {"moisture": 50.0, "temp": 25.0},
}

DEFAULT_CATALOG = pd.DataFrame([
    {"Crop":"Paddy","City":"Dehradun","Season":"Kharif (Monsoon)"},
    {"Crop":"Wheat","City":"Punjab","Season":"Rabi (Winter)"},
    {"Crop":"Groundnut","City":"Gujarat","Season":"Kharif (Monsoon)"},
    {"Crop":"Maize","City":"Karnataka","Season":"Kharif (Monsoon)"},
    {"Crop":"Cotton","City":"Maharashtra","Season":"Kharif (Monsoon)"},
    {"Crop":"Sugarcane","City":"Uttar Pradesh","Season":"Year-round"},
    {"Crop":"Mustard","City":"Rajasthan","Season":"Rabi (Winter)"},
    {"Crop":"Barley","City":"Haryana","Season":"Rabi (Winter)"},
    {"Crop":"Soybean","City":"Madhya Pradesh","Season":"Kharif (Monsoon)"},
    {"Crop":"Tomato","City":"Himachal Pradesh","Season":"Kharif (Monsoon)"},
    {"Crop":"Onion","City":"Nashik","Season":"Kharif & Rabi"},
    {"Crop":"Potato","City":"West Bengal","Season":"Rabi (Winter)"},
    {"Crop":"Chickpea","City":"Madhya Pradesh","Season":"Rabi (Winter)"},
    {"Crop":"Lentil","City":"Bihar","Season":"Rabi (Winter)"},
    {"Crop":"Pea","City":"Uttar Pradesh","Season":"Rabi (Winter)"},
    {"Crop":"Cabbage","City":"Himachal Pradesh","Season":"Kharif & Rabi"},
    {"Crop":"Cauliflower","City":"Punjab","Season":"Kharif & Rabi"},
    {"Crop":"Spinach","City":"Maharashtra","Season":"Kharif & Rabi"},
    {"Crop":"Green Beans","City":"Karnataka","Season":"Kharif (Monsoon)"},
    {"Crop":"Bell Pepper","City":"Andhra Pradesh","Season":"Kharif (Monsoon)"},
])

if "thresholds" not in st.session_state:
    st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()
if "catalog" not in st.session_state:
    st.session_state.catalog = DEFAULT_CATALOG.copy()
if "records" not in st.session_state:
    st.session_state.records = []
if "errors" not in st.session_state:
    st.session_state.errors = []
if "last_forecast" not in st.session_state:
    st.session_state.last_forecast = None
if "last_hist_df" not in st.session_state:
    st.session_state.last_hist_df = None
if "place_bias" not in st.session_state:
    # structure: { city_name: { "Kharif": bias, "Rabi": bias, "Year-round": bias, "Other": bias } }
    st.session_state.place_bias = {}

# =========================== Utilities ===========================
def log_error(msg: str) -> None:
    st.session_state.errors.insert(0, {"ts": datetime.now().isoformat(timespec="seconds"), "msg": msg})
    st.session_state.errors = st.session_state.errors[:150]

def get_season_label(season_text: str) -> str:
    s = season_text.lower()
    if "kharif" in s or "monsoon" in s or "rainy" in s:
        return "Kharif"
    if "rabi" in s or "winter" in s:
        return "Rabi"
    if "year" in s:
        return "Year-round"
    return "Other"

def season_months(season: str) -> List[int]:
    # Simple India-style split
    if season == "Kharif":
        return [6,7,8,9]  # monsoon core
    if season == "Rabi":
        return [12,1,2]   # core winter
    if season == "Year-round":
        return list(range(1,13))
    return list(range(1,13))

def recommend_threshold(sm: float, stemp: float, crop_name: str) -> str:
    """Baseline recommendation from thresholds only."""
    thr_m = st.session_state.thresholds.get(crop_name, {"moisture":40.0,"temp":25.0})["moisture"]
    thr_t = st.session_state.thresholds.get(crop_name, {"moisture":40.0,"temp":25.0})["temp"]
    try:
        if float(sm) < float(thr_m):
            return "Irrigate (low moisture)"
        st_f = float(stemp)
        if st_f > float(thr_t) + 3:
            return "Irrigate (high soil temperature)"
    except Exception:
        return "Data missing"
    return "No Irrigation"

def classify_rainfall_mm_day(mm_day: float) -> str:
    if pd.isna(mm_day):
        return "No Data"
    mm_hr = mm_day / 24.0
    if mm_hr > 50: return "Violent shower"
    if mm_hr > 10: return "Heavy shower"
    if mm_hr > 8:  return "Very heavy rain"
    if mm_hr >= 2: return "Moderate shower"
    if mm_hr > 0:  return "Slight shower"
    return "No rain"

# =========================== Forecast (OpenWeather) ===========================
@st.cache_data(ttl=60*60)
def fetch_openweather_5day(api_key: str, city: Optional[str], lat: Optional[float], lon: Optional[float]) -> Optional[dict]:
    if not api_key:
        return None
    try:
        if city:
            url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        else:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log_error(f"OpenWeather fetch failed: {e}")
        return None

def aggregate_daily_from_forecast(forecast_json: dict, days: int = 5) -> Optional[pd.DataFrame]:
    if not forecast_json or "list" not in forecast_json:
        return None
    daily = {}
    for entry in forecast_json.get("list", []):
        d = entry["dt_txt"].split(" ")[0]
        temp = entry.get("main", {}).get("temp")
        rain = 0.0
        if entry.get("rain"):
            rain = entry["rain"].get("3h", 0.0)
        if d not in daily:
            daily[d] = {"temps": [], "rains": []}
        if temp is not None:
            daily[d]["temps"].append(temp)
        daily[d]["rains"].append(rain)
    rows = []
    for d in sorted(daily.keys())[:days]:
        temps = daily[d]["temps"]
        avg_temp = float(np.mean(temps)) if temps else float("nan")
        total_rain = float(np.sum(daily[d]["rains"]))
        rows.append({"Date": pd.to_datetime(d), "Avg Temp (°C)": round(avg_temp,1), "Total Rain (mm)": round(total_rain,2)})
    return pd.DataFrame(rows)

def forecast_next24_mm(forecast_json: Optional[dict]) -> Optional[float]:
    if not forecast_json:
        return None
    try:
        now_ts = int(datetime.now().timestamp())
        accum = 0.0
        for e in forecast_json.get("list", []):
            if e.get("dt", 0) <= now_ts + 24 * 3600:
                accum += e.get("rain", {}).get("3h", 0.0)
        return round(float(accum), 2)
    except Exception as e:
        log_error(f"forecast next24 compute failed: {e}")
        return None

def coords_from_forecast(forecast_json: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    try:
        c = forecast_json.get("city", {}).get("coord", {})
        return float(c.get("lat")), float(c.get("lon"))
    except Exception:
        return None, None

# =========================== Historical: Open-Meteo ===========================
@st.cache_data(ttl=24*60*60)
def fetch_open_meteo_precip(lat: float, lon: float, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
    base = "https://archive-api.open-meteo.com/v1/archive"
    rows = []
    for y in range(start_year, end_year + 1):
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{y}-01-01", "end_date": f"{y}-12-31",
            "daily": "rain_sum", "timezone": "UTC"
        }
        try:
            r = requests.get(base, params=params, timeout=30)
            r.raise_for_status()
            j = r.json()
            if "daily" in j and "time" in j["daily"] and "rain_sum" in j["daily"]:
                for t, v in zip(j["daily"]["time"], j["daily"]["rain_sum"]):
                    try:
                        dt = pd.to_datetime(t).date()
                        rows.append({"date": pd.to_datetime(dt), "precip_mm": float(v) if v is not None else float("nan")})
                    except Exception:
                        continue
            time.sleep(0.15)  # polite
        except Exception as e:
            log_error(f"Open-Meteo fetch failed for {y}: {e}")
            continue
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def historical_rain_probability_for_date_open_meteo(target_date: datetime, hist_df: Optional[pd.DataFrame],
                                                    years: int, threshold_mm: float) -> Tuple[float, int]:
    if hist_df is None or hist_df.empty:
        return 0.0, 0
    hits = 0
    total = 0
    cur_year = target_date.year
    for y in range(cur_year - 1, cur_year - years - 1, -1):
        try:
            d = date(y, target_date.month, target_date.day)
        except ValueError:
            continue
        mask = (hist_df["date"].dt.date == d)
        if mask.any():
            total += 1
            val = hist_df.loc[mask, "precip_mm"].iloc[0]
            if not pd.isna(val) and val >= threshold_mm:
                hits += 1
    if total == 0:
        return 0.0, 0
    return round((hits / total) * 100.0, 1), total

def seasonal_averages_from_hist(hist_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    seasons = {"Summer":[3,4,5], "Rainy":[6,7,8,9], "Autumn":[10,11], "Winter":[12,1,2]}
    res = {}
    if hist_df is None or hist_df.empty:
        return {k: float("nan") for k in seasons}
    df = hist_df.copy()
    df["month"] = df["date"].dt.month
    for s, months in seasons.items():
        res[s] = float(df[df["month"].isin(months)]["precip_mm"].mean(skipna=True) or 0.0)
    return res

# =========================== Place/Season Bias ("Train") ===========================
def compute_place_bias(hist_df: Optional[pd.DataFrame], threshold_mm: float, season_key: str) -> float:
    """
    Returns bias in [ -0.3 .. +0.3 ]: positive => wetter-than-average for chosen season,
    negative => drier-than-average.
    """
    if hist_df is None or hist_df.empty:
        return 0.0
    df = hist_df.copy()
    df["month"] = df["date"].dt.month
    months = season_months(season_key)
    if not months:
        months = list(range(1,13))
    season_vals = df[df["month"].isin(months)]["precip_mm"]
    overall = df["precip_mm"]
    if season_vals.empty or overall.empty:
        return 0.0
    season_rain_days = (season_vals >= threshold_mm).mean()
    overall_rain_days = (overall >= threshold_mm).mean()
    if overall_rain_days == 0:
        return 0.0
    ratio = season_rain_days / overall_rain_days
    # squash to [-0.3, +0.3]
    bias = max(-0.3, min(0.3, (ratio - 1.0) * 0.5))
    return float(round(bias, 3))

# =========================== Fusion decision (explainable) ===========================
def compute_final_decision(sm: float, stemp: float, crop: str,
                           forecast_next24: Optional[float],
                           hist_prob: float, hist_years_used: int,
                           season_key: str, city: Optional[str],
                           thr_margin: float = 5.0) -> Dict[str, Any]:
    base = recommend_threshold(sm, stemp, crop)
    rationale: List[str] = []
    confidence = 50.0

    thr_sm = st.session_state.thresholds.get(crop, {"moisture":40.0})["moisture"]
    thr_temp = st.session_state.thresholds.get(crop, {"temp":25.0})["temp"]

    # Sensor baseline
    if base.lower().startswith("irrigate"):
        confidence += 22.0
        rationale.append("Sensor threshold indicates irrigation (soil moisture/temperature).")
    else:
        rationale.append("Sensor threshold indicates no immediate irrigation.")

    # Forecast next 24h
    if forecast_next24 is not None:
        if forecast_next24 > 10:
            confidence -= 35.0
            rationale.append(f"Forecast: very likely heavy rain next 24h ({forecast_next24:.2f} mm) — strong delay signal.")
        elif 5 < forecast_next24 <= 10:
            confidence -= 25.0
            rationale.append(f"Forecast: heavy rain ({forecast_next24:.2f} mm) — bias to delay.")
        elif 0.1 <= forecast_next24 <= 5:
            confidence -= 10.0
            rationale.append(f"Forecast: light rain ({forecast_next24:.2f} mm).")
        else:
            confidence += 6.0
            rationale.append("Forecast: no significant rain next 24h.")
    else:
        rationale.append("Forecast: unavailable (no API key or fetch failed).")

    # Historical probability on today's calendar date
    if hist_years_used > 0:
        rationale.append(f"Historical probability ≥1mm on this date: {hist_prob}% (from {hist_years_used} yrs).")
        if hist_prob >= 70: confidence -= 15.0
        elif hist_prob < 20: confidence += 10.0
    else:
        rationale.append("Historical: no usable data; safe fallback (0%).")

    # Season & place bias (trained)
    season_bias = 0.0
    if city:
        season_bias = st.session_state.place_bias.get(city, {}).get(season_key, 0.0)
        if season_bias != 0.0:
            adj = 15.0 * season_bias  # +/- 4.5% at most (since bias in [-0.3,0.3])
            confidence += adj
            if season_bias > 0:
                rationale.append(f"Trained bias: {city} is wetter than average in {season_key} — bias to delay.")
            elif season_bias < 0:
                rationale.append(f"Trained bias: {city} is drier than average in {season_key} — bias to irrigate if dry.")
        else:
            rationale.append(f"No trained bias for {city} in {season_key} yet (press Train).")

    # Very dry override
    try:
        if float(sm) < (thr_sm - thr_margin):
            confidence += 18.0
            rationale.append(f"Soil very dry ({sm:.1f}% vs thr {thr_sm}%) — urgency to irrigate.")
            if forecast_next24 is not None and forecast_next24 > 5:
                confidence -= 8.0
                rationale.append("But rain is forecast — consider staged/low-volume irrigation.")
    except Exception:
        pass

    confidence = max(0.0, min(100.0, confidence))

    # Decision
    final = base
    if forecast_next24 is not None and forecast_next24 > 5:
        final = "Delay Irrigation (forecasted rain)."
        # soften if place is historically dry (negative bias)
        if season_bias < -0.1 and float(sm) < thr_sm:
            final = "Irrigate cautiously (place drier than average; small volume)."

    if final == base:
        if base.lower().startswith("irrigate") and confidence >= 55:
            final = f"Irrigate (Confidence {confidence:.0f}%)"
        elif base.lower().startswith("no") and confidence <= 40:
            final = f"Irrigate (Low confidence in 'no irrigation' — {confidence:.0f}%)"
        else:
            final = f"{base} (Confidence {confidence:.0f}%)"

    return {"final_decision": final, "confidence": round(confidence, 1), "rationale": rationale}

# =========================== Sidebar controls ===========================
menu = st.sidebar.selectbox("Menu", ["Dashboard", "Weather", "Catalog & Train", "Water Tips", "Admin"])
st.sidebar.markdown("---")

st.sidebar.header("Location & APIs")
loc_mode = st.sidebar.selectbox("Location input", ("City Name (recommended)", "Manual lat/lon"))
if loc_mode.startswith("City"):
    default_city = "Dehradun"
    # Prefill with a city from catalog if present
    if not st.session_state.catalog.empty:
        default_city = st.session_state.catalog["City"].iloc[0]
    city = st.sidebar.text_input("City (OpenWeather)", value=default_city)
    lat = lon = None
else:
    lat = st.sidebar.number_input("Latitude", value=30.3165, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=78.0322, format="%.6f")
    city = None

openweather_key = st.sidebar.text_input("OpenWeather API Key (optional)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("Historical (Open-Meteo)")
hist_enabled = st.sidebar.checkbox("Enable historical analysis", value=True)
hist_years = st.sidebar.slider("Historical lookback (years)", 5, 30, 20)
hist_threshold_mm = st.sidebar.number_input("Rain threshold (mm/day)", min_value=0.0, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Logs & Thresholds")
store_logs = st.sidebar.checkbox("Store session logs", value=True)
if st.sidebar.button("Clear session logs"):
    st.session_state.records = []
    st.sidebar.success("Session logs cleared")

if st.sidebar.checkbox("Edit thresholds (advanced)"):
    crop_edit = st.sidebar.selectbox("Crop to edit", sorted(list(st.session_state.thresholds.keys())))
    new_m = st.sidebar.number_input(f"{crop_edit} moisture thresh (%)", 0.0, 100.0, float(st.session_state.thresholds[crop_edit]["moisture"]))
    new_t = st.sidebar.number_input(f"{crop_edit} soil temp thresh (°C)", -10.0, 60.0, float(st.session_state.thresholds[crop_edit]["temp"]))
    if st.sidebar.button("Save threshold"):
        st.session_state.thresholds[crop_edit]["moisture"] = float(new_m)
        st.session_state.thresholds[crop_edit]["temp"] = float(new_t)
        st.sidebar.success("Threshold updated")

# =========================== Catalog & File Import/Export ===========================
def catalog_add_row(crop: str, city_name: str, season_text: str, moisture_thr: Optional[float], temp_thr: Optional[float]):
    # Add/merge into catalog
    new_row = {"Crop": crop.strip(), "City": city_name.strip(), "Season": season_text.strip()}
    st.session_state.catalog = pd.concat([st.session_state.catalog, pd.DataFrame([new_row])], ignore_index=True)
    # Optionally set thresholds for the crop if provided or if new crop
    if crop.strip() not in st.session_state.thresholds:
        st.session_state.thresholds[crop.strip()] = {
            "moisture": float(moisture_thr) if moisture_thr is not None else 45.0,
            "temp": float(temp_thr) if temp_thr is not None else 24.0
        }

def page_catalog_and_train():
    st.header("Catalog — Crops × Places × Seasons")
    st.caption("Maintain your crop/place/season knowledge base. Train local rain bias so recommendations adapt to each place & season.")

    # Show table
    st.dataframe(st.session_state.catalog)

    # Add entry form
    st.markdown("### ➕ Add a crop/place/season")
    c1, c2, c3, c4, c5 = st.columns([1.2,1,1,1,1])
    with c1:
        crop_new = st.text_input("Crop", value="")
    with c2:
        city_new = st.text_input("City/Place", value="")
    with c3:
        season_new = st.selectbox("Season", ["Kharif (Monsoon)", "Rabi (Winter)", "Year-round", "Other"])
    with c4:
        m_thr = st.number_input("Moisture % (opt)", 0.0, 100.0, 45.0, step=1.0)
    with c5:
        t_thr = st.number_input("Soil Temp °C (opt)", -10.0, 60.0, 24.0, step=0.5)
    if st.button("Add to Catalog"):
        if crop_new.strip() and city_new.strip():
            catalog_add_row(crop_new, city_new, season_new, m_thr, t_thr)
            st.success(f"Added: {crop_new} in {city_new} ({season_new})")
        else:
            st.warning("Please provide at least Crop and City/Place.")

    # File import
    st.markdown("### 📥 Import catalog from CSV")
    st.caption("Columns supported: Crop, City, Season, Moisture(optional), Temp(optional)")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            dfu = pd.read_csv(up)
            required_cols = {"Crop", "City", "Season"}
            if not required_cols.issubset(set(dfu.columns)):
                st.error("CSV must include at least: Crop, City, Season")
            else:
                for _, r in dfu.iterrows():
                    catalog_add_row(
                        str(r["Crop"]),
                        str(r["City"]),
                        str(r["Season"]),
                        float(r["Moisture"]) if "Moisture" in dfu.columns and pd.notna(r["Moisture"]) else None,
                        float(r["Temp"]) if "Temp" in dfu.columns and pd.notna(r["Temp"]) else None
                    )
                st.success(f"Imported {len(dfu)} rows into catalog.")
        except Exception as e:
            st.error(f"Import failed: {e}")

    # Export
    st.markdown("### 📤 Export current catalog")
    st.download_button("Download catalog CSV", st.session_state.catalog.to_csv(index=False).encode("utf-8"), file_name="catalog.csv")

    st.markdown("---")
    st.subheader("Train place-season bias (uses Open-Meteo history)")
    st.caption("Select a city/place from the catalog and train. We compute how rainy that place is in the chosen season vs its overall average, and store a bias used in recommendations.")

    # Choose place + season for training
    cities = sorted(st.session_state.catalog["City"].unique().tolist()) if not st.session_state.catalog.empty else []
    tc1, tc2, tc3 = st.columns([1,1,2])
    with tc1:
        train_city = st.selectbox("City/Place", cities)
    with tc2:
        train_season_full = st.selectbox("Season", ["Kharif","Rabi","Year-round","Other"])
    with tc3:
        train_lat = st.number_input("Latitude (if needed)", value=lat if (lat is not None) else 30.3165, format="%.6f")
        train_lon = st.number_input("Longitude (if needed)", value=lon if (lon is not None) else 78.0322, format="%.6f")

    if st.button("Train"):
        # If user has forecast for this city, prefer coordinates from it
        forecast_json = None
        if openweather_key.strip() and train_city:
            forecast_json = fetch_openweather_5day(openweather_key.strip(), train_city, None, None)
        if forecast_json:
            f_lat, f_lon = coords_from_forecast(forecast_json)
        else:
            f_lat, f_lon = train_lat, train_lon

        if not hist_enabled:
            st.warning("Enable historical analysis in sidebar.")
            return

        start_year = datetime.today().year - hist_years
        end_year = datetime.today().year - 1
        hist_df = fetch_open_meteo_precip(f_lat, f_lon, start_year, end_year)
        if hist_df is None:
            st.error("Historical fetch failed. Try manual lat/lon or another city.")
        else:
            bias = compute_place_bias(hist_df, hist_threshold_mm, train_season_full)
            st.session_state.place_bias.setdefault(train_city, {})
            st.session_state.place_bias[train_city][train_season_full] = bias
            st.success(f"Trained bias for {train_city} / {train_season_full}: {bias:+.3f}")
            st.caption("Positive = wetter than average → app will be more conservative (tend to delay). Negative = drier → app irrigates earlier if dry.")

    # Show current trained biases
    if st.session_state.place_bias:
        st.markdown("#### Current trained biases")
        rows = []
        for c, d in st.session_state.place_bias.items():
            for s, b in d.items():
                rows.append({"City": c, "Season": s, "Bias": b})
        st.dataframe(pd.DataFrame(rows))

# =========================== Pages ===========================
def page_dashboard():
    st.header("Dashboard — Live Input & Place-aware Prediction")
    st.caption("Enter sensor readings, choose crop & place. Press Predict to get an explainable recommendation that adapts to city & season.")

    # Select crop + city from catalog
    left, right = st.columns([1.1, 1])
    with left:
        crops = sorted(list(st.session_state.thresholds.keys()))
        crop = st.selectbox("Crop", crops, index=crops.index("Paddy") if "Paddy" in crops else 0)
    with right:
        cat_cities = sorted(st.session_state.catalog["City"].unique().tolist()) if not st.session_state.catalog.empty else []
        city_sel = st.selectbox("City/Place (from catalog)", cat_cities if cat_cities else ["(none)"])

    # Derive season for the selected crop + city (first match)
    season_text = "Year-round"
    if city_sel and not st.session_state.catalog.empty:
        matches = st.session_state.catalog[(st.session_state.catalog["City"] == city_sel) & (st.session_state.catalog["Crop"] == crop)]
        if matches.empty:
            # fallback: any season for that city
            matches = st.session_state.catalog[(st.session_state.catalog["City"] == city_sel)]
        if not matches.empty:
            season_text = str(matches.iloc[0]["Season"])
    season_key = get_season_label(season_text)
    st.markdown(f"Season: <span class='pill'>{season_text}</span>", unsafe_allow_html=True)

    # Sensors
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        soil_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 30.0, step=0.1, format="%.1f")
    with col2:
        soil_temp = st.number_input("Soil Temperature (°C)", -10.0, 60.0, 25.0, step=0.1, format="%.1f")
    with col3:
        air_temp = st.number_input("Air Temperature (°C)", -20.0, 60.0, 28.0, step=0.1, format="%.1f")
    with col4:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, step=0.1, format="%.1f")

    # Persist logs
    if store_logs:
        rec = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "city": city_sel,
            "season": season_key,
            "crop": crop,
            "soil_moisture": round(float(soil_moisture),1),
            "soil_temp": round(float(soil_temp),1),
            "air_temp": round(float(air_temp),1),
            "humidity": round(float(humidity),1)
        }
        st.session_state.records.insert(0, rec)
        st.session_state.records = st.session_state.records[:300]

    # KPIs
    thr = st.session_state.thresholds[crop]["moisture"]
    thr_t = st.session_state.thresholds[crop]["temp"]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="card"><div class="small">Soil Moisture</div><div class="metric">{soil_moisture:.1f}%</div><div class="small">threshold: {thr}%</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div class="small">Soil Temp</div><div class="metric">{soil_temp:.1f}°C</div><div class="small">threshold: {thr_t}°C</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div class="small">Air Temp</div><div class="metric">{air_temp:.1f}°C</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="card"><div class="small">Humidity</div><div class="metric">{humidity:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Forecast & Historical (optional)")
    forecast_json = None
    forecast_df = None
    f_next24 = None
    f_msg = ""

    # Forecast (and auto coords for Open-Meteo)
    f_lat = lat
    f_lon = lon
    if openweather_key.strip():
        forecast_json = fetch_openweather_5day(openweather_key.strip(), city_sel if city_sel and loc_mode.startswith("City") else (city if city else None), lat if lat else None, lon if lon else None)
        if forecast_json:
            st.session_state.last_forecast = forecast_json
            f_df = aggregate_daily_from_forecast(forecast_json, days=5)
            if f_df is not None and not f_df.empty:
                st.dataframe(f_df.assign(Date=f_df["Date"].dt.strftime("%Y-%m-%d")))
            # coords
            c_lat, c_lon = coords_from_forecast(forecast_json)
            if c_lat and c_lon:
                f_lat, f_lon = c_lat, c_lon
            f_next24 = forecast_next24_mm(forecast_json)
        else:
            f_msg = "OpenWeather fetch failed. Using last successful forecast if available."
            if st.session_state.last_forecast:
                forecast_json = st.session_state.last_forecast
                f_df = aggregate_daily_from_forecast(forecast_json, days=5)
                if f_df is not None and not f_df.empty:
                    st.dataframe(f_df.assign(Date=f_df["Date"].dt.strftime("%Y-%m-%d")))
                c_lat, c_lon = coords_from_forecast(forecast_json)
                if c_lat and c_lon:
                    f_lat, f_lon = c_lat, c_lon
                f_next24 = forecast_next24_mm(forecast_json)
    else:
        f_msg = "OpenWeather key not set. Forecast disabled."
    if f_msg:
        st.info(f_msg)

    # Historical (Open-Meteo)
    hist_prob, hist_years_used = 0.0, 0
    hist_df = None
    seasonal_avgs = {}
    if hist_enabled and (f_lat is not None and f_lon is not None):
        start_year = datetime.today().year - hist_years
        end_year = datetime.today().year - 1
        hist_df = fetch_open_meteo_precip(f_lat, f_lon, start_year, end_year)
        if hist_df is not None:
            st.session_state.last_hist_df = hist_df
        else:
            if st.session_state.last_hist_df is not None:
                hist_df = st.session_state.last_hist_df
                st.warning("Historical fetch failed — using last cached historical data.")
            else:
                st.info("Historical data unavailable; using safe fallback (0%).")
                hist_df = None
        hist_prob, hist_years_used = historical_rain_probability_for_date_open_meteo(datetime.today(), hist_df, hist_years, hist_threshold_mm)
        seasonal_avgs = seasonal_averages_from_hist(hist_df) if hist_df is not None else {}

    st.markdown("---")
    st.write("Press **Predict** to combine sensors + forecast + history + trained place bias.")
    if st.button("Predict"):
        fusion = compute_final_decision(
            sm=float(soil_moisture),
            stemp=float(soil_temp),
            crop=crop,
            forecast_next24=f_next24,
            hist_prob=hist_prob,
            hist_years_used=hist_years_used,
            season_key=season_key,
            city=city_sel
        )
        final = fusion["final_decision"]
        conf = fusion["confidence"]
        if "Delay" in final:
            cls = "rec-green"
        elif "Irrigate" in final:
            cls = "rec-red" if conf >= 60 else "rec-orange"
        else:
            cls = "rec-orange"
        st.markdown(f'<div class="{cls}"><h3 style="margin:6px 0 2px 0;">{final}</h3><div class="small">Confidence: {conf}%</div></div>', unsafe_allow_html=True)
        st.progress(int(conf))
        st.subheader("Why this?")
        for r in fusion["rationale"]:
            st.write("-", r)

        # seasonal summary
        if seasonal_avgs:
            st.markdown("**Seasonal averages (30-year):**")
            for s, v in seasonal_avgs.items():
                st.write(f"- {s}: {v:.1f} mm (avg)")

        # demo water-savings if delay
        if f_next24 is not None and "Delay" in final:
            area_ha = st.sidebar.number_input("Demo: farm area (ha)", value=0.5, min_value=0.01, step=0.1)
            liters_saved = int(f_next24 * area_ha * 10000)
            st.success(f"Estimated water saved if you delay and rain occurs: ~{liters_saved:,} L (for {area_ha} ha).")
    else:
        st.info("Enter readings and press Predict.")

    # Logs & trends
    st.markdown("---")
    st.subheader("Recent Sensor Logs & Trend")
    if st.session_state.records:
        df = pd.DataFrame(st.session_state.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df.head(20))
        df["date_only"] = df["timestamp"].dt.date
        daily = df.groupby("date_only").agg({
            "soil_moisture": "mean",
            "soil_temp": "mean",
            "air_temp": "mean",
            "humidity": "mean"
        }).reset_index().rename(columns={"date_only": "Date"})
        daily = daily.sort_values("Date")
        fig = px.line(daily, x="Date", y=["soil_moisture", "soil_temp", "air_temp", "humidity"], markers=True,
                      labels={"value": "Value", "variable": "Parameter"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No session logs yet. Enable 'Store session logs' in sidebar and enter readings to log data.")

def page_weather():
    st.header("Weather — Forecast & Calendar-date History")
    if not openweather_key.strip():
        st.info("Add OpenWeather API key (sidebar) to fetch forecast.")
    else:
        forecast_json = fetch_openweather_5day(openweather_key.strip(), city if city else None, lat if lat else None, lon if lon else None)
        if forecast_json:
            st.session_state.last_forecast = forecast_json
        else:
            if st.session_state.last_forecast:
                st.info("OpenWeather fetch failed; using last successful forecast.")
                forecast_json = st.session_state.last_forecast
            else:
                st.error("OpenWeather fetch failed and no cached forecast available.")
                forecast_json = None
        if forecast_json:
            df = aggregate_daily_from_forecast(forecast_json, days=7)
            if df is not None and not df.empty:
                df["Rain Category"] = df["Total Rain (mm)"].apply(classify_rainfall_mm_day)
                st.dataframe(df.assign(Date=df["Date"].dt.strftime("%Y-%m-%d")))
                col1, col2 = st.columns(2)
                with col1:
                    fig_t = px.line(df, x="Date", y="Avg Temp (°C)", markers=True, title="Avg Temp")
                    st.plotly_chart(fig_t, use_container_width=True)
                with col2:
                    fig_r = px.bar(df, x="Date", y="Total Rain (mm)", title="Total Rain (mm)")
                    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.subheader("Historical (Open-Meteo archive)")
    if not hist_enabled:
        st.info("Enable historical analysis in sidebar to fetch Open-Meteo data.")
        return

    # Determine coords (from forecast if available, else manual)
    f_lat, f_lon = (lat if lat is not None else 30.3165), (lon if lon is not None else 78.0322)
    if st.session_state.last_forecast:
        c_lat, c_lon = coords_from_forecast(st.session_state.last_forecast)
        if c_lat and c_lon:
            f_lat, f_lon = c_lat, c_lon

    start_year = datetime.today().year - hist_years
    end_year = datetime.today().year - 1
    hist_df = fetch_open_meteo_precip(f_lat, f_lon, start_year, end_year)
    if hist_df is not None:
        st.session_state.last_hist_df = hist_df
    else:
        if st.session_state.last_hist_df is not None:
            st.info("Open-Meteo fetch failed; using last cached historical dataset.")
            hist_df = st.session_state.last_hist_df
        else:
            st.error("Historical fetch failed and no cached data available.")
            hist_df = None
    prob, used = historical_rain_probability_for_date_open_meteo(datetime.today(), hist_df, hist_years, hist_threshold_mm)
    st.write(f"Historical probability of ≥ {hist_threshold_mm} mm on this date: **{prob}%** (years used: {used})")

    if hist_df is not None and used > 0:
        vals, yrs = [], []
        for y in range(datetime.today().year - 1, datetime.today().year - hist_years - 1, -1):
            try:
                d = date(y, datetime.today().month, datetime.today().day)
            except ValueError:
                continue
            mask = (hist_df["date"].dt.date == d)
            if mask.any():
                yrs.append(y)
                vals.append(hist_df.loc[mask, "precip_mm"].iloc[0])
        if vals:
            dfp = pd.DataFrame({"Year": yrs[::-1], "Precip_mm": vals[::-1]})
            f = px.bar(dfp, x="Year", y="Precip_mm", title="Precipitation on this calendar date (per year)")
            st.plotly_chart(f, use_container_width=True)
        else:
            st.info("No per-year records for this calendar date in fetched range (e.g., leap-day).")

def page_water_tips():
    st.header("Water Tips — Farmer-Friendly Advice")
    st.markdown("""
      - Prefer early morning / late evening irrigation to reduce evaporation.
      - Use this dashboard's recommendation rather than guesswork.
      - If forecast shows heavy rain (>5 mm/24h) and historical probability is high, delay irrigation.
      - For critically low soil moisture, consider low-volume emergency irrigation even if rain is forecast.
      - Mulch and organic matter increase retention and reduce irrigation frequency.
      - Calibrate with **Catalog & Train** → train your city/season bias for better local accuracy.
    """)
    st.markdown("---")
    st.subheader("Examples")
    st.write("- Paddy: maintain saturated conditions during vegetative phase.")
    st.write("- Maize/Wheat: allow moisture to approach threshold before irrigating to avoid waste.")

def page_admin():
    st.header("Admin / Diagnostics")
    st.subheader("Recent errors (session)")
    if st.session_state.errors:
        st.dataframe(pd.DataFrame(st.session_state.errors))
    else:
        st.info("No errors logged in this session.")

    st.subheader("Session sensor logs (preview & export)")
    if st.session_state.records:
        dfr = pd.DataFrame(st.session_state.records)
        dfr["timestamp"] = pd.to_datetime(dfr["timestamp"])
        st.dataframe(dfr.head(100))
        st.download_button("Download logs CSV", dfr.to_csv(index=False).encode("utf-8"), file_name=f"sensor_logs_{datetime.now().date()}.csv")
    else:
        st.info("No logs yet.")

    st.subheader("Current thresholds")
    st.json(st.session_state.thresholds)

    st.subheader("Catalog snapshot")
    st.dataframe(st.session_state.catalog)

    st.subheader("Trained place biases")
    if st.session_state.place_bias:
        st.json(st.session_state.place_bias)
    else:
        st.info("No biases trained yet.")

# =========================== Router ===========================
if menu == "Dashboard":
    page_dashboard()
elif menu == "Weather":
    page_weather()
elif menu == "Catalog & Train":
    page_catalog_and_train()
elif menu == "Water Tips":
    page_water_tips()
else:
    page_admin()

# =========================== Footer ===========================
st.markdown("---")
st.markdown("""
**Notes:**  
- **Catalog**: comes preloaded with 20 crops, example places, and seasons. Add your own via form or CSV; export anytime.  
- **Trained bias**: uses **Open-Meteo** 5–30 years daily rain to estimate how wet/dry a city is in a season, then adapts recommendations.  
- **Forecast**: uses OpenWeather 3-hour forecast (optional API key). If missing, app still works with sensors + history.  
- **Thresholds**: pre-set per crop but editable in the sidebar. Validate in production with an agronomist.  
- Caching keeps the app responsive even with intermittent network.
""")
