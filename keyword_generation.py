import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file
df = pd.read_csv('/home/chetan/ads data analysis/ads analysis/processed.csv')

# Handle missing values, if any
df = df.dropna()

# Extract features and target variable
X = df[['spend', 'top_keywords']]
y = df['impressions']

# One-hot encode 'top_keywords'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['top_keywords'])
    ],
    remainder='passthrough'
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to suggest better keywords
def suggest_keywords(spend, current_keywords, model, preprocessor):
    # Create a dataframe for prediction
    data = pd.DataFrame({'spend': [spend], 'top_keywords': [current_keywords]})
    
    # Predict impressions
    predicted_impressions = model.predict(preprocessor.transform(data))
    
    return predicted_impressions

# Example usage
better_keywords = suggest_keywords(100, 'your_keyword_here', pipeline, preprocessor)
print(f'Suggested Keywords for better impressions: {better_keywords}')