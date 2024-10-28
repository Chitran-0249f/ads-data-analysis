import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame containing all the features
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Ad Features')
plt.show()


# Example with age and ad creation day columns
age_ad_creation = df.groupby(['age', 'ad_creation_day']).size().unstack().fillna(0)
age_ad_creation.plot(kind='barh', stacked=True, figsize=(10, 6))
plt.title('Age Group vs Ad Creation Day')
plt.xlabel('Number of Ads')
plt.ylabel('Age Group')
plt.show()

# Assuming 'year' and 'ad_creation_time' columns in df
plt.figure(figsize=(12, 6))
df['ad_creation_year'] = pd.to_datetime(df['ad_creation_time']).dt.year
df['ad_creation_year'].value_counts().sort_index().plot(kind='bar')
plt.title('Number of Ads Created Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Ads')
plt.show()

# Assuming 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment' columns
df[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']].hist(bins=20, figsize=(12, 6))
plt.suptitle('Distribution of Sentiment Scores')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Example model training and results visualization
X = df[['ad_spend', 'ad_duration', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment']]
y = df['impressions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr)
plt.title('Linear Regression: Actual vs Predicted Impressions')
plt.xlabel('Actual Impressions')
plt.ylabel('Predicted Impressions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Train Decision Tree model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, color='orange')
plt.title('Decision Tree: Actual vs Predicted Impressions')
plt.xlabel('Actual Impressions')
plt.ylabel('Predicted Impressions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()



import matplotlib.pyplot as plt

# Assuming an example where a business optimized ad spend and duration
# before and after using your model

before_optimization = df[df['campaign'] == 'before']
after_optimization = df[df['campaign'] == 'after']

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Before optimization
ax[0].scatter(before_optimization['ad_spend'], before_optimization['impressions'], color='blue')
ax[0].set_title('Before Optimization')
ax[0].set_xlabel('Ad Spend')
ax[0].set_ylabel('Impressions')

# After optimization
ax[1].scatter(after_optimization['ad_spend'], after_optimization['impressions'], color='green')
ax[1].set_title('After Optimization')
ax[1].set_xlabel('Ad Spend')
ax[1].set_ylabel('Impressions')

plt.suptitle('Case Study: Ad Spend vs Impressions Before and After Optimization')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Feature importance visualization
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance in Decision Tree Model")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.show()

