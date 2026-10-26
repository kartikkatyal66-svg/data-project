import pandas as pd

# 1. Load the dataset
df = pd.read_csv('data/amazon_delivery.csv')

# 2. Fill missing values
df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())
df['Weather'] = df['Weather'].fillna('Unknown')

# 3. Combine date and time strings
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce')
df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Pickup_Time'], errors='coerce')

# 4. Calculate initial prep time in minutes
df['Prep_Time_Minutes'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60

# --- THE FIX HAPPENS HERE ---
# If Prep_Time_Minutes is negative (crossed midnight), add 1440 minutes (24 hours)
df.loc[df['Prep_Time_Minutes'] < 0, 'Prep_Time_Minutes'] = df['Prep_Time_Minutes'] + 1440

# 5. Clean up any corrupted rows where time parsing failed
df = df.dropna(subset=['Order_DateTime', 'Pickup_DateTime'])

# 6. Verify the fix
print("Corrected Prep Time Summary (in minutes):")
print(df['Prep_Time_Minutes'].describe())