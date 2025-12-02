import pandas as pd

# Load CSV
df = pd.read_csv("social_profiles.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Clean missing values
df['bio'] = df['bio'].fillna('none')
df['followers'] = df['followers'].fillna(0)
df['following'] = df['following'].fillna(0)
df['posts'] = df['posts'].fillna(0)

print("Cleaning Completed Successfully!")
