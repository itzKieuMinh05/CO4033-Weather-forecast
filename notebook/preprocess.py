import pandas as pd
import numpy as np
import glob

files = glob.glob("../data/weather-vn-*.csv")
dfs = []

for f in files:
    print("Loading:", f)
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(df))

# print("Unique cities:", df["city"].unique())

df = df.drop_duplicates()


df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday



df = df.drop(columns=["province"], errors="ignore")
north = [
    'Bac Giang','Bac Kan','Bac Ninh','Tu Son','Cao Bang',
    'Dien Bien Phu','Ha Giang','Ha Noi','Hai Duong','Hai Phong',
    'Hoa Binh','Hung Yen','My Hao','Lai Chau','Lang Son',
    'Lao Cai','Nam Dinh','Ninh Binh','Tam Diep','Viet Tri',
    'Cam Pha','Ha Long','Mong Cai','Uong Bi','Son La',
    'Song Cong','Thai Nguyen','Tuyen Quang','Phuc Yen',
    'Vinh Yen','Yen Bai'
]

central = [
    'Quy Nhon','Phan Thiet','Da Nang','Buon Ma Thuot','Gia Nghia',
    'Pleiku','Ha Tinh','Hong Linh','Cam Ranh','Nha Trang',
    'Kon Tum','Phan Rang - Thap Cham','Tuy Hoa','Dong Hoi',
    'Hoi An','Tam Ky','Quang Ngai','Dong Ha','Thai Hoa',
    'Vinh','Sam Son','Thanh Hoa'
]

south = [
    'Chau Doc','Long Xuyen','Ba Ria','Vung Tau','Bac Lieu',
    'Ben Tre','Di An','Thu Dau Mot','Thuan An','Dong Xoai',
    'Ca Mau','Can Tho','Bien Hoa','Long Khanh','Nga Bay',
    'Vi Thanh','Ha Tien','Rach Gia','Tan An','Soc Trang',
    'Ho Chi Minh','Tay Ninh','My Tho','Tra Vinh','Vinh Long'
]

def map_region(city):
    if city in north:
        return "north"
    elif city in central:
        return "central"
    elif city in south:
        return "south"
    else:
        return "unknown"

df["region"] = df["city"].apply(map_region)

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object","string"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# range nhiệt độ
# if "temp_max" in df.columns and "temp_min" in df.columns:
#     df["temp_range"] = df["temp_max"] - df["temp_min"]
date_key = df["time"].dt.date
df["temp_range"] = (
    df.groupby(["city", date_key])["temp_max"].transform("max")
    - df.groupby(["city", date_key])["temp_min"].transform("min")
)

# wind direction -> vector
if "wind_direction" in df.columns:
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction"])).round(4)
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction"])).round(4)


df = df.drop(columns=["weather_desc"], errors="ignore")
if "rainfall" in df.columns:
    df["rain"] = df["rainfall"].apply(lambda x: 1 if x > 0 else 0)
elif "precipitation" in df.columns:
    df["rain"] = df["precipitation"].apply(lambda x: 1 if x > 0 else 0)

# 2. Create EXTREME WEATHER column
def extreme_weather(row):
    if row.get("temp_max", 0) > 35:
        return "heatwave"
    elif row.get("rainfall", 0) > 50:
        return "heavy_rain"
    elif row.get("wind_speed", 0) > 40:
        return "storm"
    elif row.get("rainfall", 0) > 0:
        return "rain"
    else:
        return "normal"

df["extreme"] = df.apply(extreme_weather, axis=1)

# 3. Discretization (for Apriori)
if "temp_max" in df.columns:
    df["temp_level"] = pd.cut(
        df["temp_max"],
        bins=[-100, 20, 30, 100],
        labels=["temp_low", "temp_medium", "temp_high"]
    )

if "humidity" in df.columns:
    df["humidity_level"] = pd.cut(
        df["humidity"],
        bins=[0, 60, 80, 100],
        labels=["humidity_low", "humidity_medium", "humidity_high"]
    )

if "pressure" in df.columns:
    df["pressure_level"] = pd.cut(
        df["pressure"],
        bins=[0, 1000, 1015, 2000],
        labels=["pressure_low", "pressure_normal", "pressure_high"]
    )

if "wind_speed" in df.columns:
    df["wind_level"] = pd.cut(
        df["wind_speed"],
        bins=[0, 10, 25, 100],
        labels=["wind_low", "wind_medium", "wind_high"]
    )

# 4. Lag features (previous hour)
df = df.sort_values("time")

if "temp_max" in df.columns:
    # df["temp_lag_1"] = df["temp_max"].shift(1)
    df["temp_lag_1"] = df.groupby("city")["temp_max"].shift(1)

if "humidity" in df.columns:
    # df["humidity_lag_1"] = df["humidity"].shift(1)
    df["humidity_lag_1"] = df.groupby("city")["humidity"].shift(1)

if "pressure" in df.columns:
    # df["pressure_lag_1"] = df["pressure"].shift(1)
    df["pressure_lag_1"] = df.groupby("city")["pressure"].shift(1)

# save
df.to_csv("weather_vn_cleaned.csv", index=False)

print("Saved: weather_vn_cleaned.csv")