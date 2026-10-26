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


# 4. Calculate Haversine Distance (KM)
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
  R = 6371.0  # Earth's radius in kilometers
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

# 5. Inspect distance statistics
print("Distance Summary (KM):")
print(df['Distance_KM'].describe())
