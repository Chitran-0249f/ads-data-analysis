import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Example DataFrame (replace with your actual data)
data = {
    'ad_creative_bodies': [
        'Great deals on electronics',
        'Buy one get one free',
        'Limited time offer',
        'Flash sale',
        'Exclusive discount',
        'Hurry, last chance'
    ],
    'spend': [500, 800, 1000, 1200, 700, 1500],
    'ad_delivery_duration_days': [7, 5, 10, 3, 8, 6],
    'impressions': [1000, 1500, 1200, 2000, 1700, 3000]
}
df = pd.DataFrame(data)

# Step 1: Handle NaN values in 'ad_creative_bodies'
df['ad_creative_bodies'].fillna('', inplace=True)

# Step 2: Define preprocessing pipeline
numeric_features = ['spend', 'ad_delivery_duration_days']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=100, stop_words='english'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('txt', text_transformer, 'ad_creative_bodies')
    ])

# Step 3: Split data into training and testing sets
X = df[['spend', 'ad_delivery_duration_days', 'ad_creative_bodies']]
y = df['impressions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the pipeline with preprocessing and Ridge regression
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# Step 5: Train the model
ridge_pipeline.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = ridge_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Ridge Regression Mean Squared Error: {mse}")
print(f"Actual Impressions: {y_test.values}")
print(f"Predicted Impressions: {y_pred}")

# Cross-Validation
scores = cross_val_score(ridge_pipeline, X, y, cv=3, scoring='neg_mean_squared_error')
mse_scores = -scores
print(f"Ridge Regression Cross-Validation MSE Scores: {mse_scores}")
print(f"Average Cross-Validation MSE: {mse_scores.mean()}")

# Example: Predictions for new data
def predict_impressions(ad_creative_bodies, spend, ad_delivery_duration_days):
    new_data = pd.DataFrame({
        'spend': [spend],
        'ad_delivery_duration_days': [ad_delivery_duration_days],
        'ad_creative_bodies': [ad_creative_bodies]
    })
    predictions = ridge_pipeline.predict(new_data)
    return predictions[0]

# CLI interface for user input
while True:
    print("\nEnter values to predict impressions (type 'exit' to quit):")
    
    # Input ad_creative_bodies
    ad_creative_bodies = input("Ad Creative Bodies: ").strip()
    if ad_creative_bodies.lower() == 'exit':
        break
    
    # Input spend
    while True:
        spend_input = input("Spend ($): ").strip()
        try:
            spend = float(spend_input)
            break
        except ValueError:
            print("Invalid input. Please enter a valid numeric value for Spend.")
    
    # Input ad_delivery_duration_days
    while True:
        ad_delivery_duration_days_input = input("Ad Delivery Duration (days): ").strip()
        try:
            ad_delivery_duration_days = int(ad_delivery_duration_days_input)
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer value for Ad Delivery Duration (days).")
    
    # Predict impressions
    predicted_impressions = predict_impressions(ad_creative_bodies, spend, ad_delivery_duration_days)
    print(f"Predicted Impressions: {predicted_impressions:.2f}")
