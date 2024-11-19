import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create directories if they don't exist
model_dir = Path("src/models/saved")
model_dir.mkdir(parents=True, exist_ok=True)

# Create sample training data
np.random.seed(42)
n_samples = 1000

# Generate features
app_sizes = np.random.lognormal(5, 1, n_samples)  # App sizes in MB
prices = np.random.exponential(2, n_samples)  # Prices in USD
downloads = np.random.lognormal(10, 2, n_samples)  # Number of downloads

# Generate categorical features
app_types = ['communication', 'education', 'entertainment', 'music',
             'productivity', 'social', 'travel', 'video']
stores = ['amazon', 'apple', 'google_play']

app_type_data = np.random.choice(app_types, n_samples)
store_data = np.random.choice(stores, n_samples)

# Create target variable (ratings)
base_ratings = 3.5 + np.random.normal(0, 0.5, n_samples)
# Add some realistic effects
base_ratings += 0.3 * (downloads > np.median(downloads))  # Popular apps tend to have higher ratings
base_ratings += 0.2 * (prices < np.median(prices))  # Free/cheaper apps tend to have higher ratings
base_ratings = np.clip(base_ratings, 1, 5)  # Ensure ratings are between 1 and 5

# Create DataFrame
data = pd.DataFrame({
    'app_size_mb': app_sizes,
    'price_usd': prices,
    'downloads': downloads,
    'app_type': app_type_data,
    'store': store_data,
    'rating': base_ratings
})

# Create dummy variables for categorical features
data_encoded = pd.get_dummies(data, columns=['app_type', 'store'])

# Split features and target
X = data_encoded.drop('rating', axis=1)
y = data_encoded['rating']

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
model_path = model_dir / "random_forest.joblib"
joblib.dump(model, model_path)

print(f"Model trained and saved to {model_path}")

# Save feature names for reference
feature_names = X.columns.tolist()
print("\nFeature names:")
print(feature_names)
