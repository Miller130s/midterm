import pandas as pd
import numpy as np
from scipy import stats

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
import pandas as pd
import joblib

space_df = pd.read_csv("data/Master_Space_Data_All.txt")

# -----------------------------
# Clean base data
# -----------------------------
space_df.columns = space_df.columns.str.strip().str.lower()
space_df["datum"] = pd.to_datetime(space_df["datum"], errors="coerce")
space_df = space_df.dropna(subset=["datum"]).copy()
space_df["year"] = space_df["datum"].dt.year.astype(int)

# -----------------------------
# Location coordinates
# -----------------------------
location_coords = {
    "Plesetsk Cosmodrome": [64.69703911171159, 40.23202122101674],
    "Cape Canaveral": [28.496279762592692, -80.57721035730918],
    "Baikonur Cosmodrome": [45.96971306334741, 63.304277886937385],
    "Vandenberg": [34.63707751118516, -120.61462487252487],
    "French Guiana": [5.238019177412824, -52.776268518418654],
    "Kennedy Space Center": [28.574602148274234, -80.65200491882827],
    "Jiuquan": [40.986111395894675, 100.20835167725694],
    "Xichang": [27.900606068582352, 102.24352617642391],
    "Kapustin Yar": [48.64379836121863, 45.7721244816821],
    "Taiyuan": [38.848757118763466, 111.60800445417883],
    "Tanegashima": [30.401876861139385, 130.9774333097495],
    "Sriharikota": [13.740619507371893, 80.23402800803493],
    "Uchinoura": [31.25150534828375, 131.0762021966414],
    "Wallops": [37.9342633502727, -75.47241206565033],
    "Mahia": [-39.238060638304155, 177.8746161716849],
    "Wenchang": [19.57689134663915, 110.74800694686337],
    "Corn Ranch": [31.423072192287673, -104.75717884752879],
    "Vostochny Cosmodrome": [51.849518067287875, 128.35522818077246],
    "Semnan": [35.9543550544185, 53.80750192838674],
    "Yellow Sea": [36.703990039698674, 121.23598239731571],
    "Palmachim": [31.90644163622597, 34.6933143865769],
    "Kwajalein Atoll": [8.983651302616964, 167.57807575820812],
    "Kodiak Launch Complex": [57.43097117418452, -152.35639607941172],
    "Starbase": [25.99235099359263, -97.18482351352796],
    "San Marco Platform": [-2.995544841817323, 40.19486645175857],
    "Woomera": [-31.068851903797793, 136.44265455563396],
    "Point Mugu": [34.087052955939185, -119.06102115015644],
    "Edwards AFB": [34.917527442249266, -117.89127770264322],
    "Naro Space Center": [34.453631884612534, 127.5179146256433],
    "Spaceport America": [32.9903641484431, -106.97509136279191],
    "Hammaguir": [30.86787202447907, -3.0436440278099473],
    "Mojave Air and Space Port": [35.0293525766903, -118.1059669157189],
    "Sohae": [39.66836829084492, 124.70702819417858],
    "Gran Canaria": [27.925204643989716, -15.621479594745013],
    "S. Korea": [36.64325884421309, 127.20682071754388]
}

# -----------------------------
# Clean location names
# -----------------------------
def clean_location(location):
    location = str(location).strip()

    # USA
    if "Vandenberg" in location:
        return "Vandenberg"
    elif "Cape Canaveral" in location or "CCAFS" in location or "Cape Kennedy" in location:
        return "Cape Canaveral"
    elif "Kennedy Space Center" in location or "KSC" in location:
        return "Kennedy Space Center"
    elif "Wallops" in location:
        return "Wallops"
    elif "Kodiak" in location:
        return "Kodiak Launch Complex"
    elif "Starbase" in location:
        return "Starbase"
    elif "Corn Ranch" in location or "Van Horn" in location:
        return "Corn Ranch"
    elif "Point Mugu" in location:
        return "Point Mugu"
    elif "Edwards" in location:
        return "Edwards AFB"
    elif "Spaceport America" in location:
        return "Spaceport America"
    elif "Mojave" in location:
        return "Mojave Air and Space Port"
    elif "Kwajalein" in location:
        return "Kwajalein Atoll"
    elif "San Marco" in location:
        return "San Marco Platform"

    # Russia / USSR / Kazakhstan
    elif "Plesetsk" in location:
        return "Plesetsk Cosmodrome"
    elif "Baikonur" in location or "Tyuratam" in location:
        return "Baikonur Cosmodrome"
    elif "Kapustin Yar" in location:
        return "Kapustin Yar"
    elif "Vostochny" in location:
        return "Vostochny Cosmodrome"

    # China
    elif "Jiuquan" in location:
        return "Jiuquan"
    elif "Xichang" in location:
        return "Xichang"
    elif "Taiyuan" in location:
        return "Taiyuan"
    elif "Wenchang" in location:
        return "Wenchang"
    elif "Yellow Sea" in location:
        return "Yellow Sea"

    # Japan / India / NZ / Korea
    elif "Tanegashima" in location:
        return "Tanegashima"
    elif "Uchinoura" in location:
        return "Uchinoura"
    elif "Sriharikota" in location or "Satish Dhawan" in location:
        return "Sriharikota"
    elif "Mahia" in location:
        return "Mahia"
    elif "Naro" in location or "Goheung" in location:
        return "Naro Space Center"
    elif "S. Korea" in location or "South Korea" in location:
        return "S. Korea"

    # Europe / Middle East / others
    elif "French Guiana" in location or "Kourou" in location or "Guiana Space Centre" in location:
        return "French Guiana"
    elif "Palmachim" in location:
        return "Palmachim"
    elif "Semnan" in location:
        return "Semnan"
    elif "Woomera" in location:
        return "Woomera"
    elif "Hammaguir" in location:
        return "Hammaguir"
    elif "Sohae" in location:
        return "Sohae"
    elif "Gran Canaria" in location:
        return "Gran Canaria"

    return None


space_df = pd.read_csv("data/Master_Space_Data_All.txt")
space_df.columns = space_df.columns.str.strip().str.lower()

space_df["datum"] = space_df["datum"].astype(str).str.strip()
space_df["year"] = space_df["datum"].str.extract(r"(\d{4})")
space_df["year"] = pd.to_numeric(space_df["year"], errors="coerce")

space_df = space_df.dropna(subset=["year"]).copy()
space_df["year"] = space_df["year"].astype(int)

# your clean_location() and location_coords go here
space_df["location_clean"] = space_df["location"].apply(clean_location)
space_df["coords"] = space_df["location_clean"].map(location_coords)

space_df = space_df.dropna(subset=["coords"]).copy()
space_df["lat"] = space_df["coords"].apply(lambda x: x[0])
space_df["lon"] = space_df["coords"].apply(lambda x: x[1])

map_data = space_df[["year", "datum", "location", "location_clean", "lat", "lon"]].copy()

print("map_data year range:", map_data["year"].min(), "to", map_data["year"].max())
print("map_data rows:", len(map_data))

joblib.dump(map_data, "map_data.joblib")
print("saved")

#NNNNNNNEEEEEEEEEEEEEEEEEEEEEWWWWWWWWWWWWWWWWWWWWWW




import streamlit as st
import pydeck as pdk
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("🚀 Global Launch Density Timeline (1957 - 2025)")

# --- 1. Your Provided Coordinate Mapping ---
location_coords = {
    "Plesetsk Cosmodrome": [64.69703911171159, 40.23202122101674],
    "Cape Canaveral": [28.496279762592692, -80.57721035730918],
    "Baikonur Cosmodrome": [45.96971306334741, 63.304277886937385],
    "Vandenberg": [34.63707751118516, -120.61462487252487],
    "French Guiana": [5.238019177412824, -52.776268518418654],
    "Kennedy Space Center": [28.574602148274234, -80.65200491882827],
    "Jiuquan": [40.986111395894675, 100.20835167725694],
    "Xichang": [27.900606068582352, 102.24352617642391],
    "Kapustin Yar": [48.64379836121863, 45.7721244816821],
    "Taiyuan": [38.848757118763466, 111.60800445417883],
    "Tanegashima": [30.401876861139385, 130.9774333097495],
    "Sriharikota": [13.740619507371893, 80.23402800803493],
    "Uchinoura": [31.25150534828375, 131.0762021966414],
    "Wallops": [37.9342633502727, -75.47241206565033],
    "Mahia": [-39.238060638304155, 177.8746161716849],
    "Wenchang": [19.57689134663915, 110.74800694686337],
    "Corn Ranch": [31.423072192287673, -104.75717884752879],
    "Vostochny Cosmodrome": [51.849518067287875, 128.35522818077246],
    "Semnan": [35.9543550544185, 53.80750192838674],
    "Yellow Sea": [36.703990039698674, 121.23598239731571],
    "Palmachim": [31.90644163622597, 34.6933143865769],
    "Kwajalein Atoll": [8.983651302616964, 167.57807575820812],
    "Kodiak Launch Complex": [57.43097117418452, -152.35639607941172],
    "Starbase": [25.99235099359263, -97.18482351352796],
    "San Marco Platform": [-2.995544841817323, 40.19486645175857],
    "Woomera": [-31.068851903797793, 136.44265455563396],
    "Point Mugu": [34.087052955939185, -119.06102115015644],
    "Edwards AFB": [34.917527442249266, -117.89127770264322],
    "Naro Space Center": [34.453631884612534, 127.5179146256433],
    "Spaceport America": [32.9903641484431, -106.97509136279191],
    "Hammaguir": [30.86787202447907, -3.0436440278099473],
    "Mojave Air and Space Port": [35.0293525766903, -118.1059669157189],
    "Sohae": [39.66836829084492, 124.70702819417858],
    "Gran Canaria": [27.925204643989716, -15.621479594745013],
    "S. Korea": [36.64325884421309, 127.20682071754388]
}

# --- 2. Your Provided Cleaning Function ---
def clean_location(location):
    location = str(location).strip()
    if "Vandenberg" in location: return "Vandenberg"
    elif "Cape Canaveral" in location or "CCAFS" in location or "Cape Kennedy" in location: return "Cape Canaveral"
    elif "Kennedy Space Center" in location or "KSC" in location: return "Kennedy Space Center"
    elif "Wallops" in location: return "Wallops"
    elif "Kodiak" in location: return "Kodiak Launch Complex"
    elif "Starbase" in location: return "Starbase"
    elif "Corn Ranch" in location or "Van Horn" in location: return "Corn Ranch"
    elif "Point Mugu" in location: return "Point Mugu"
    elif "Edwards" in location: return "Edwards AFB"
    elif "Spaceport America" in location: return "Spaceport America"
    elif "Mojave" in location: return "Mojave Air and Space Port"
    elif "Kwajalein" in location: return "Kwajalein Atoll"
    elif "San Marco" in location: return "San Marco Platform"
    elif "Plesetsk" in location: return "Plesetsk Cosmodrome"
    elif "Baikonur" in location or "Tyuratam" in location: return "Baikonur Cosmodrome"
    elif "Kapustin Yar" in location: return "Kapustin Yar"
    elif "Vostochny" in location: return "Vostochny Cosmodrome"
    elif "Jiuquan" in location: return "Jiuquan"
    elif "Xichang" in location: return "Xichang"
    elif "Taiyuan" in location: return "Taiyuan"
    elif "Wenchang" in location: return "Wenchang"
    elif "Yellow Sea" in location: return "Yellow Sea"
    elif "Tanegashima" in location: return "Tanegashima"
    elif "Uchinoura" in location: return "Uchinoura"
    elif "Sriharikota" in location or "Satish Dhawan" in location: return "Sriharikota"
    elif "Mahia" in location: return "Mahia"
    elif "Naro" in location or "Goheung" in location: return "Naro Space Center"
    elif "S. Korea" in location or "South Korea" in location: return "S. Korea"
    elif "French Guiana" in location or "Kourou" in location or "Guiana Space Centre" in location: return "French Guiana"
    elif "Palmachim" in location: return "Palmachim"
    elif "Semnan" in location: return "Semnan"
    elif "Woomera" in location: return "Woomera"
    elif "Hammaguir" in location: return "Hammaguir"
    elif "Sohae" in location: return "Sohae"
    elif "Gran Canaria" in location: return "Gran Canaria"
    return None

# --- 3. Loading and Preparing Data ---
@st.cache_data
def load_and_map_data():
    # Load raw CSV
    df = pd.read_csv("Master_Space_Data_All.csv")
    
    # 1. Clean the Location names into keys for our coord dictionary
    df["clean_loc"] = df["Location"].apply(clean_location)
    
    # 2. Map coordinates (Lat/Lon) to the dataframe
    df["lat"] = df["clean_loc"].apply(lambda x: location_coords[x][0] if x in location_coords else None)
    df["lon"] = df["clean_loc"].apply(lambda x: location_coords[x][1] if x in location_coords else None)
    
    # 3. Handle Dates
    df['year'] = pd.to_datetime(df['Datum'], errors='coerce').dt.year
    
    # Drop rows without required info
    return df.dropna(subset=['year', 'lat', 'lon']).copy()

space_df = load_and_map_data()

# --- 4. Animation logic ---
if st.button('▶️ Start Timeline Animation'):
    header_placeholder = st.empty()
    map_placeholder = st.empty()

    for year in range(1957, 2026):
        # Filter for data up to this year
        current_data = space_df[space_df['year'] <= year]
        
        # Aggregate launch counts by coordinate
        launch_counts = (
            current_data.groupby(["lat", "lon", "clean_loc"])
            .size()
            .reset_index(name="launch_count")
        )
        
        launch_counts["coordinates"] = launch_counts.apply(lambda r: [r["lon"], r["lat"]], axis=1)
        launch_counts["elevation"] = launch_counts["launch_count"] * 10000
        
        # Color calculation
        max_val = launch_counts["launch_count"].max()
        def get_color(cnt):
            ratio = cnt / max_val if max_val > 0 else 0
            return [255, int(220 - 170 * ratio), int(120 - 120 * ratio), 180]
        
        launch_counts["color"] = launch_counts["launch_count"].apply(get_color)

        # PyDeck Layer
        layer = pdk.Layer(
            "ColumnLayer",
            data=launch_counts,
            get_position="coordinates",
            get_elevation="elevation",
            get_fill_color="color",
            radius=60000,
            extruded=True,
            pickable=True,
        )

        # Update Display
        header_placeholder.subheader(f"Global Launch Density: {year}")
        map_placeholder.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=1.1, pitch=45),
            tooltip={"html": "<b>Location:</b> {clean_loc}<br/><b>Total Launches:</b> {launch_count}"}
        ))
        
        time.sleep(0.08) # Animation speed
else:
    st.info("Ready to play. Click the button above to start the 1957-2025 timeline.")