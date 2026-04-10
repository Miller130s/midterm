from importlib.metadata import metadata
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import numpy as np
from scipy import stats
import pydeck as pdk
from PIL import Image
from streamlit_autorefresh import st_autorefresh

# Split and Cross Val
from sklearn.model_selection import train_test_split, GridSearchCV

# Preprocess and Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeCV, LassoCV

# Ensemble Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier

# Metrics
from sklearn.metrics import (mean_absolute_error, 
                             accuracy_score, 
                             root_mean_squared_error, 
                             r2_score, 
                             mean_absolute_error,
                             mean_squared_error, 
                             silhouette_score)

# Environments and serialization
import joblib

# formatting
import datetime

# Datasets
from sklearn.datasets import load_diabetes

st.title("Nicks Midterm")

space_df = pd.read_csv("data/space.csv")
st.dataframe(space_df)
global_image = Image.open("global.png")
usa_image = Image.open("usa.png")
shift_image = Image.open("shift.png")


st.image(global_image)

st.image(usa_image)

st.image(shift_image)

pivot_focus = joblib.load("midterm.joblib")

series_order = [
    "SpaceX",
    "China",
    "Russia",
    "NASA",
    "ULA (United Launch Alliance)"
]

plot_cols = [c for c in series_order if c in pivot_focus.columns]

colors = {
    "China": "#f1c40f",
    "Russia": "#e74c3c",
    "NASA": "#2c3e50",
    "SpaceX": "#3498db",
    "ULA (United Launch Alliance)": "#9b59b6"
}

x_data = [str(x) for x in pivot_focus.index.tolist()]

series = []
for col in plot_cols:
    series.append({
        "name": col,
        "type": "bar",
        "data": pivot_focus[col].tolist(),
        "itemStyle": {"color": colors[col]},
        "markLine": {
            "symbol": ["none", "none"],
            "label": {
                "show": True,
                "formatter": "Commercial Launch Boom Begins",
                "position": "insideEndTop"
            },
            "lineStyle": {
                "color": "black",
                "width": 2,
                "type": "dashed"
            },
            "data": [{"xAxis": "2013"}]
        }
    })

options = {
    "title": {
        "text": "The Shift to Commercial Dominance",
        "left": "center"
    },
    "tooltip": {
        "trigger": "axis",
        "axisPointer": {"type": "shadow"}
    },
    "legend": {
        "top": 50
    },
    "grid": {
        "left": "6%",
        "right": "2%",
        "bottom": "12%",
        "top": "18%",
        "containLabel": True
    },
    "xAxis": {
        "type": "category",
        "name": "Year",
        "data": x_data,
        "axisLabel": {"rotate": 45}
    },
    "yAxis": {
        "type": "value",
        "name": "Launch Count"
    },
    "series": series
}

st_echarts(
    options=options,
    height="600px",
    width="100%",
    key="space_launch_chart"
)

#Map



st.title("Launch Density by Location")

space_df = joblib.load("map_data.joblib")

space_df["lat"] = pd.to_numeric(space_df["lat"], errors="coerce")
space_df["lon"] = pd.to_numeric(space_df["lon"], errors="coerce")
space_df = space_df.dropna(subset=["lat", "lon"]).copy()

space_df["coordinates"] = space_df.apply(lambda row: [row["lon"], row["lat"]], axis=1)


launch_counts = (
    space_df.groupby(["lat", "lon"])
    .agg(
        launch_count=("location", "size"),
        location=("location", "first")
    )
    .reset_index()
)

launch_counts["coordinates"] = launch_counts.apply(
    lambda row: [row["lon"], row["lat"]], axis=1
)

launch_counts["elevation"] = launch_counts["launch_count"] * 10000

max_count = launch_counts["launch_count"].max()

def get_bar_color(count):
    ratio = count / max_count if max_count > 0 else 0
    red = 255
    green = int(220 - 170 * ratio)
    blue = int(120 - 120 * ratio)
    return [red, max(green, 0), max(blue, 0), 180]

launch_counts["color"] = launch_counts["launch_count"].apply(get_bar_color)

layer = pdk.Layer(
    "ColumnLayer",
    data=launch_counts,
    get_position="coordinates",
    get_elevation="elevation",
    get_fill_color="color",
    radius=50000,
    pickable=True,
    extruded=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=20,
    longitude=10,
    zoom=1.1,
    pitch=50,
    bearing=0
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Location:</b> {location}<br/><b>Launch Count:</b> {launch_count}"
    },
    map_style=None
)

st.pydeck_chart(deck, height=700)




# neeeeeeeeeeeeeeeewwww
import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
from streamlit_autorefresh import st_autorefresh

st.title("Launch Density by Location")

# -----------------------------
# Load map data
# -----------------------------
space_df = joblib.load("map_data.joblib")

# -----------------------------
# Clean numeric columns
# -----------------------------
space_df["lat"] = pd.to_numeric(space_df["lat"], errors="coerce")
space_df["lon"] = pd.to_numeric(space_df["lon"], errors="coerce")

# -----------------------------
# Rebuild year safely
# This works even if datum formats vary
# -----------------------------
space_df["datum"] = space_df["datum"].astype(str).str.strip()
space_df["year"] = space_df["datum"].str.extract(r"(\d{4})")
space_df["year"] = pd.to_numeric(space_df["year"], errors="coerce")

# -----------------------------
# Drop bad rows
# -----------------------------
space_df = space_df.dropna(subset=["lat", "lon", "year"]).copy()
space_df["year"] = space_df["year"].astype(int)

# -----------------------------
# Optional debug check
# -----------------------------
st.write("Year range in map data:", int(space_df["year"].min()), "to", int(space_df["year"].max()))

# -----------------------------
# Start autoplay at 1975
# -----------------------------
min_year = 1975
max_year = int(space_df["year"].max())

# If your data starts after 1975, this prevents errors
if min_year < int(space_df["year"].min()):
    min_year = int(space_df["year"].min())

# -----------------------------
# Session state
# -----------------------------
if "selected_year" not in st.session_state:
    st.session_state.selected_year = min_year

if "playing" not in st.session_state:
    st.session_state.playing = False

# Keep year inside valid range
if st.session_state.selected_year < min_year:
    st.session_state.selected_year = min_year
if st.session_state.selected_year > max_year:
    st.session_state.selected_year = max_year

# -----------------------------
# Auto-play
# -----------------------------
if st.session_state.playing:
    st_autorefresh(interval=1200, key="launch_animation")

    if st.session_state.selected_year < max_year:
        st.session_state.selected_year += 1
    else:
        st.session_state.selected_year = min_year

# -----------------------------
# Controls
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("▶ Play"):
        st.session_state.playing = True

with col2:
    selected_year = st.slider(
        "Year",
        min_value=min_year,
        max_value=max_year,
        value=st.session_state.selected_year
    )
    st.session_state.selected_year = selected_year

with col3:
    if st.button("⏸ Pause"):
        st.session_state.playing = False

current_year = st.session_state.selected_year
st.subheader(f"Showing launches through {current_year}")

# -----------------------------
# Cumulative launches through current year
# -----------------------------
df_year = space_df[space_df["year"] <= current_year].copy()

launch_counts = (
    df_year.groupby(["lat", "lon"])
    .agg(
        launch_count=("location", "size"),
        location=("location", "first")
    )
    .reset_index()
)

if launch_counts.empty:
    st.warning(f"No launch data available through {current_year}.")
else:
    launch_counts["coordinates"] = launch_counts.apply(
        lambda row: [row["lon"], row["lat"]], axis=1
    )

    launch_counts["elevation"] = launch_counts["launch_count"] * 10000

    max_count = launch_counts["launch_count"].max()

    def get_bar_color(count):
        ratio = count / max_count if max_count > 0 else 0
        red = 255
        green = int(220 - 170 * ratio)
        blue = int(120 - 120 * ratio)
        return [red, max(green, 0), max(blue, 0), 180]

    launch_counts["color"] = launch_counts["launch_count"].apply(get_bar_color)

    layer = pdk.Layer(
        "ColumnLayer",
        data=launch_counts,
        get_position="coordinates",
        get_elevation="elevation",
        get_fill_color="color",
        radius=50000,
        pickable=True,
        extruded=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=20,
        longitude=10,
        zoom=1.1,
        pitch=50,
        bearing=0
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Location:</b> {location}<br/><b>Launch Count:</b> {launch_count}"
        },
        map_style=None
    )

    st.pydeck_chart(deck, height=700)