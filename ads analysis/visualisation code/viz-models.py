from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Decision Tree model
tree_model = DecisionTreeRegressor(max_depth=3, random_state=0)
tree_model.fit(X, y)

# Plotting the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=['Ad_Spent', 'Day_Ran', 'Ad_Description_Length'], filled=True, rounded=True)
plt.title('Decision Tree for Predicting Ad Impressions')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
np.random.seed(0)
ad_spent = np.random.rand(100) * 1000  # Random ad spending between 0 and 1000
day_ran = np.random.randint(1, 8, 100)  # Random day of the week (1=Monday, 7=Sunday)
ad_description_length = np.random.randint(20, 150, 100)  # Random length of ad body text

# Simulated impressions with some random noise
impressions = 500 + 0.8 * ad_spent + 50 * day_ran + 2 * ad_description_length + np.random.randn(100) * 100

# Creating a DataFrame
df = pd.DataFrame({
    'Ad_Spent': ad_spent,
    'Day_Ran': day_ran,
    'Ad_Description_Length': ad_description_length,
    'Impressions': impressions
})

# Independent variables
X = df[['Ad_Spent', 'Day_Ran', 'Ad_Description_Length']]
# Dependent variable
y = df['Impressions']

# Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predicting Impressions
df['Predicted_Impressions'] = model.predict(X)

# Plotting the regression line for Ad Spent vs Impressions
plt.figure(figsize=(10, 6))
plt.scatter(df['Ad_Spent'], df['Impressions'], color='blue', label='Actual Impressions')
plt.plot(df['Ad_Spent'], df['Predicted_Impressions'], color='red', label='Regression Line')
plt.xlabel('Ad Spending (in $)')
plt.ylabel('Impressions')
plt.title('Regression Line: Ad Spending vs Impressions')
plt.legend()
plt.grid(True)
plt.show()
