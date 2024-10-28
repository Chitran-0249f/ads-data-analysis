import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Example Test values 
data = {
    'ad_creative_bodies': ['Great deals on electronics', 'Buy one get one free', 'Limited time offer', 'Flash sale', 'Exclusive discount', 'Hurry, last chance'],
    'spend': [500, 800, 1000, 1200, 700, 1500],
    'ad_delivery_duration_days': [7, 5, 10, 3, 8, 6],
    'impressions': [1000, 1500, 1200, 2000, 1700, 3000]  # Added more variability
}
df = pd.DataFrame(data)

# Step 1: Handle NaN values in 'ad_creative_bodies'
df['ad_creative_bodies'].fillna('', inplace=True)  # Replace NaN with empty string

# Step 2: Define preprocessing pipeline
numeric_features = ['spend', 'ad_delivery_duration_days']
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('txt', text_transformer, 'ad_creative_bodies')
    ])

# Step 3: Split data into features and labels
X = df[['spend', 'ad_delivery_duration_days', 'ad_creative_bodies']]
y = df['impressions']

# Step 4: Build the pipeline with preprocessing and Random Forest
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_estimators=100, max_depth=3))
])

# Step 5: Apply cross-validation (optional)
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = -1 * np.mean(cv_scores)  # Convert negative MSE to positive

print(f"Mean CV Mean Squared Error: {mean_cv_score}")

# Step 6: Train the model on the full dataset
model_pipeline.fit(X, y)

# Step 7: Prediction function with spend check
def predict_impressions(ad_creative_bodies, spend, ad_delivery_duration_days):
    if spend < 500:
        return "Your $ spend is less than 500"  # Return message if spend is below 500
    
    # If spend is 500 or more, proceed with the prediction
    new_data = pd.DataFrame({
        'spend': [spend],
        'ad_delivery_duration_days': [ad_delivery_duration_days],
        'ad_creative_bodies': [ad_creative_bodies]
    })
    predictions = model_pipeline.predict(new_data)
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
    print(f"Predicted Impressions: {predicted_impressions}")
