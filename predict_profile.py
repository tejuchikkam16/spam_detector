import pandas as pd
import joblib

# Load saved model
model = joblib.load("spam_model.pkl")

# New profile data
new_profile = {
    'username': 'free_coins2025',
    'followers': 10,
    'following': 500,
    'posts': 1,
    'bio': 'Get free coins now',
    'profile_picture': 0
}

# Feature Engineering
username_length = len(new_profile['username'])
bio_length = len(new_profile['bio'])
followers_following_ratio = new_profile['followers'] / (new_profile['following'] + 1)

# Prepare DataFrame
X_new = pd.DataFrame([[
    username_length,
    bio_length,
    new_profile['followers'],
    new_profile['following'],
    new_profile['posts'],
    new_profile['profile_picture'],
    followers_following_ratio
]], columns=['username_length','bio_length','followers','following','posts','profile_picture','followers_following_ratio'])

# Predict
prediction = model.predict(X_new)[0]

if prediction == 1:
    print("This profile is likely FAKE 🚨")
else:
    print("This profile is REAL ✅")
