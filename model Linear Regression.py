import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Example DataFrame (replace with your actual data)
data = {
    'ad_creative_bodies': ['Great deals on electronics', np.nan, 'Limited time offer'],
    'spend': [500, 800, 1000],
    'ad_delivery_duration_days': [7, 5, 10],
    'impressions': [1000, 1500, 1200]
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

# Step 3: Split data into training and testing sets
X = df[['spend', 'ad_delivery_duration_days', 'ad_creative_bodies']]
y = df['impressions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the pipeline with preprocessing and linear regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 5: Train the model
model_pipeline.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example: Predictions for new data
def predict_impressions(ad_creative_bodies, spend, ad_delivery_duration_days):
    new_data = pd.DataFrame({
        'spend': [spend],
        'ad_delivery_duration_days': [ad_delivery_duration_days],
        'ad_creative_bodies': [ad_creative_bodies]
    })
    predictions = model_pipeline.predict(new_data)
    return abs(predictions[0])

# CLI interface with input validation
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
