import pandas as pd

# Load cleaned CSV
df = pd.read_csv("social_profiles.csv")

# Feature Engineering
df['username_length'] = df['username'].apply(len)
df['bio_length'] = df['bio'].apply(len)
df['followers_following_ratio'] = df['followers'] / (df['following'] + 1)  # +1 to avoid divide by zero

# Show first 5 rows
print("Feature Engineering Completed!")
print(df.head())
