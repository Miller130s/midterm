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

space_df = pd.read_csv("data/Master_Space_Data_All.txt")

#Data
space_df.columns = space_df.columns.str.strip().str.lower()

space_df["datum"] = pd.to_datetime(space_df["datum"], errors="coerce")

space_df = space_df.dropna(subset=["datum"]).copy()

space_df['countries'] = space_df['location'].str.split(',').str[-1].str.strip()
# print(space_df['countries'])

space_df["year"] = space_df["datum"].dt.year.astype(int)

df_modern = space_df[
    (space_df["year"] >= 1990) & (space_df["year"] <= 2026)
].copy()

# graph
def map_to_focus_groups(row):
    company = str(row["company name"]).strip()
    country = str(row["countries"]).strip()

    if country == "China":
        return "China"

    if country in ["Russia", "USSR", "Russia/USSR"]:
        return "Russia"

    if "SpaceX" in company:
        return "SpaceX"

    if "NASA" in company:
        return "NASA"

    if "ULA" in company or "United Launch Alliance" in company:
        return "ULA (United Launch Alliance)"

    return "Other"

df_modern["Focus_Group"] = df_modern.apply(map_to_focus_groups, axis=1)


df_filtered = df_modern[df_modern["Focus_Group"] != "Other"].copy()


early_years = [1990, 1993, 1994, 1995, 1998, 2001, 2004, 2007, 2013, 2016]
post_2000_years = list(range(2019, 2026))
years_to_keep = early_years + post_2000_years

df_filtered = df_filtered[df_filtered["year"].isin(years_to_keep)]


pivot_focus = (
    df_filtered
    .groupby(["year", "Focus_Group"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

joblib.dump(pivot_focus, "midterm.joblib")

#MAP DATA

space_df.columns = space_df.columns.str.strip().str.lower()

# --- your location dictionary ---
location_coords = {
    "Plesetsk Cosmodrome, Russia": [64.69703911171159, 40.23202122101674],
    "Cape Canaveral, FL, USA": [28.496279762592692, -80.57721035730918],
    "Baikonur Cosmodrome, Kazakhstan": [45.96971306334741, 63.304277886937385],
    "Vandenberg SFB, CA, USA": [34.63707751118516, -120.61462487252487],
    "French Guiana, France": [5.238019177412824, -52.776268518418654],
    "Kennedy Space Center, FL, USA": [28.574602148274234, -80.65200491882827],
    "Jiuquan, China": [40.986111395894675, 100.20835167725694],
    "Xichang, China": [27.900606068582352, 102.24352617642391],
    "Kapustin Yar, Russia": [48.64379836121863, 45.7721244816821],
    "Taiyuan, China": [38.848757118763466, 111.60800445417883],
    "Tanegashima, Japan": [30.401876861139385, 130.9774333097495],
    "Sriharikota, India": [13.740619507371893, 80.23402800803493],
    "Uchinoura, Japan": [31.25150534828375, 131.0762021966414],
    "Wallops Flight Facility, VA, USA": [37.9342633502727, -75.47241206565033],
    "Mahia Peninsula, New Zealand": [-39.238060638304155, 177.8746161716849],
    "Wenchang, China": [19.57689134663915, 110.74800694686337],
    "Corn Ranch, TX, USA": [31.423072192287673, -104.75717884752879],
    "Vostochny Cosmodrome, Russia": [51.849518067287875, 128.35522818077246],
    "Semnan, Iran": [35.9543550544185, 53.80750192838674],
    "Yellow Sea, China": [36.703990039698674, 121.23598239731571],
    "Palmachim, Israel": [31.90644163622597, 34.6933143865769],
    "Kwajalein Atoll, Marshall Islands (US)": [8.983651302616964, 167.57807575820812],
    "Kodiak Launch Complex, AK, USA": [57.43097117418452, -152.35639607941172],
    "Starbase, TX, USA": [25.99235099359263, -97.18482351352796],
    "San Marco Platform, Kenya": [-2.995544841817323, 40.19486645175857],
    "Woomera, Australia": [-31.068851903797793, 136.44265455563396],
    "Point Mugu, CA, USA": [34.087052955939185, -119.06102115015644],
    "Edwards AFB, CA, USA": [34.917527442249266, -117.89127770264322],
    "Naro Space Center, South Korea": [34.453631884612534, 127.5179146256433],
    "Spaceport America, NM, USA": [32.9903641484431, -106.97509136279191],
    "Hammaguir, Algeria": [30.86787202447907, -3.0436440278099473],
    "Mojave Air and Space Port, CA, USA": [35.0293525766903, -118.1059669157189],
    "Sohae, North Korea": [39.66836829084492, 124.70702819417858],
    "Gran Canaria": [27.925204643989716, -15.621479594745013],
    "S. Korea": [36.64325884421309, 127.20682071754388],
    "Vandenberg": [34.63707751118516, -120.61462487252487],
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

# --- match text to coords ---
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

# split lat/lon
space_df = space_df.dropna(subset=["coords"]).copy()
space_df["lat"] = space_df["coords"].apply(lambda x: x[0])
space_df["lon"] = space_df["coords"].apply(lambda x: x[1])

joblib.dump(space_df, "map_data.joblib")

# optional styling
space_df["radius"] = space_df["price"].fillna(10) * 1000

def get_color(company):
    if "SpaceX" in str(company):
        return [52, 152, 219, 180]
    if "CASC" in str(company):
        return [241, 196, 15, 180]
    return [120, 120, 120, 160]

space_df["color"] = space_df["company name"].apply(get_color)

# 🔥 SAVE IT
joblib.dump(space_df, "map_data.joblib")