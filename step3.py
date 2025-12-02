import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset with features
df = pd.read_csv("social_profiles.csv")

# Feature Engineering
df['username_length'] = df['username'].apply(len)
df['bio_length'] = df['bio'].apply(len)
df['followers_following_ratio'] = df['followers'] / (df['following'] + 1)

# Features and target
X = df[['username_length','bio_length','followers','following','posts','profile_picture','followers_following_ratio']]
y = df['fake']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy*100:.2f}%")
