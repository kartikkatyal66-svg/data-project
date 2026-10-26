import numpy as np
import pandas as pd

# 1. Load data
df = pd.read_csv('data/amazon_delivery.csv')

# 2. Imputations
df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())
df['Weather'] = df['Weather'].fillna('Unknown')

# 3. Clean timestamps and fix midnight bug
df['Order_DateTime'] = pd.to_datetime(
    df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce'
)
df['Pickup_DateTime'] = pd.to_datetime(
    df['Order_Date'] + ' ' + df['Pickup_Time'], errors='coerce'
)
df['Prep_Time_Minutes'] = (
    df['Pickup_DateTime'] - df['Order_DateTime']
).dt.total_seconds() / 60
df.loc[df['Prep_Time_Minutes'] < 0, 'Prep_Time_Minutes'] = (
    df['Prep_Time_Minutes'] + 1440
)
df = df.dropna(subset=['Order_DateTime', 'Pickup_DateTime'])

# 4. FIX GPS COORDINATES (strip negative sign glitches)
df['Store_Latitude'] = df['Store_Latitude'].abs()
df['Store_Longitude'] = df['Store_Longitude'].abs()
df['Drop_Latitude'] = df['Drop_Latitude'].abs()
df['Drop_Longitude'] = df['Drop_Longitude'].abs()


# 5. Calculate Haversine Distance (KM)
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
  R = 6371.0
  lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
  lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

  dlat = lat2_rad - lat1_rad
  dlon = lon2_rad - lon1_rad

  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
  )
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return R * c


df['Distance_KM'] = calculate_haversine_distance(
    df['Store_Latitude'],
    df['Store_Longitude'],
    df['Drop_Latitude'],
    df['Drop_Longitude'],
)

# 6. Filter realistic last-mile delivery distances (between 100m and 30km)
df = df[(df['Distance_KM'] >= 0.1) & (df['Distance_KM'] <= 30)]

# 7. Print the cleaned distance metrics
print("Cleaned Dataset Shape:", df.shape)
print("\nAccurate Delivery Distance Summary (KM):")
print(df['Distance_KM'].describe())
# --- 8. ROOT CAUSE & BOTTLENECK ANALYSIS ---

print("\n--- 1. IMPACT OF TRAFFIC DENSITY ---")
# Compare average and median delivery time by traffic condition
traffic_perf = df.groupby('Traffic')[['Delivery_Time', 'Distance_KM']].agg(
    ['mean', 'count']
)
print(traffic_perf)

print("\n--- 2. IMPACT OF WEATHER ---")
# Compare average delivery time across weather patterns
weather_perf = df.groupby('Weather')['Delivery_Time'].agg(
    ['mean', 'count']
).sort_values(by='mean', ascending=False)
print(weather_perf)

print("\n--- 3. FLEET PERFORMANCE (VEHICLE TYPE) ---")
# Compare vehicle efficiency across delivery times
vehicle_perf = df.groupby('Vehicle')['Delivery_Time'].agg(
    ['mean', 'count']
).sort_values(by='mean', ascending=False)
print(vehicle_perf)

print("\n--- 4. CORRELATION CHECK ---")
# Check how strongly distance and prep time correlate with total delivery time
correlations = df[['Distance_KM', 'Prep_Time_Minutes', 'Delivery_Time']].corr()
print(correlations['Delivery_Time'])
