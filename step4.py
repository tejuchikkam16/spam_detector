import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("social_profiles.csv")

# Feature Engineering
df['username_length'] = df['username'].apply(len)
df['bio_length'] = df['bio'].apply(len)
df['followers_following_ratio'] = df['followers'] / (df['following'] + 1)

# Features and target
X = df[['username_length','bio_length','followers','following','posts','profile_picture','followers_following_ratio']]
y = df['fake']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "spam_model.pkl")
print("Model saved as spam_model.pkl")
