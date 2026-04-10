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
    location = str(location)

    if "Vandenberg" in location:
        return "Vandenberg"
    elif "Cape Canaveral" in location:
        return "Cape Canaveral"
    elif "Kennedy Space Center" in location:
        return "Kennedy Space Center"
    elif "Wallops" in location:
        return "Wallops"
    elif "Mahia" in location:
        return "Mahia"
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
    elif "Tanegashima" in location:
        return "Tanegashima"
    elif "Sriharikota" in location or "Satish Dhawan" in location:
        return "Sriharikota"
    elif "Uchinoura" in location:
        return "Uchinoura"
    elif "Plesetsk" in location:
        return "Plesetsk Cosmodrome"
    elif "Baikonur" in location:
        return "Baikonur Cosmodrome"
    elif "Kapustin Yar" in location:
        return "Kapustin Yar"
    elif "Vostochny" in location:
        return "Vostochny Cosmodrome"
    elif "Semnan" in location:
        return "Semnan"
    elif "Palmachim" in location:
        return "Palmachim"
    elif "Kodiak" in location:
        return "Kodiak Launch Complex"
    elif "Starbase" in location:
        return "Starbase"
    elif "Corn Ranch" in location or "Van Horn" in location:
        return "Corn Ranch"
    elif "San Marco" in location:
        return "San Marco Platform"
    elif "Woomera" in location:
        return "Woomera"
    elif "Point Mugu" in location:
        return "Point Mugu"
    elif "Edwards" in location:
        return "Edwards AFB"
    elif "Naro" in location or "Goheung" in location:
        return "Naro Space Center"
    elif "Spaceport America" in location:
        return "Spaceport America"
    elif "Hammaguir" in location:
        return "Hammaguir"
    elif "Mojave" in location:
        return "Mojave Air and Space Port"
    elif "Sohae" in location:
        return "Sohae"
    elif "Gran Canaria" in location:
        return "Gran Canaria"
    elif "French Guiana" in location or "Kourou" in location or "Guiana Space Centre" in location:
        return "French Guiana"
    elif "Kwajalein" in location:
        return "Kwajalein Atoll"
    elif "S. Korea" in location or "South Korea" in location:
        return "S. Korea"
    else:
        return None

space_df["location_clean"] = space_df["location"].apply(clean_location)
space_df["coords"] = space_df["location_clean"].map(location_coords)

space_df = space_df.dropna(subset=["coords"]).copy()

# coords are [lat, lon]
space_df["lat"] = space_df["coords"].apply(lambda x: x[0])
space_df["lon"] = space_df["coords"].apply(lambda x: x[1])

# keep only what the map needs
map_data = space_df[["year", "location", "location_clean", "lat", "lon"]].copy()

print("Year range in map_data:", map_data["year"].min(), "to", map_data["year"].max())
print("Rows in map_data:", len(map_data))

joblib.dump(map_data, "map_data.joblib")
print("map_data.joblib saved successfully")