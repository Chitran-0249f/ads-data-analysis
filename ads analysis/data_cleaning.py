import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')

# Assuming df is your DataFrame containing 'ad_creative_bodies' column
# Let's use NLTK's Vader for sentiment analysis
sid = SentimentIntensityAnalyzer()

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# calculate average spend, Extract month and year, column displaying the name of the month, calculate average 'impressions' column, Calculate the difference in days

raw_file = "/home/chetan/ads data analysis/dataset/dataset-40k.csv"
transformed_file = "/home/chetan/ads data analysis/dataset/cleaned_data.csv"


# Define the path to the CSV file
csv_file_path = raw_file

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

num_rows = df.shape[0]
print(f"The number of rows in the DataFrame is: {num_rows}")
#DATA CLEANING

# Function to calculate average spend
def calculate_average_spend(spend_str):
    # Extract the lower and upper bounds using regex
    match = re.search(r'lower_bound: (\d+), upper_bound: (\d+)', spend_str)
    if match:
        lower_bound = int(match.group(1))
        upper_bound = int(match.group(2))
        # Calculate the average
        average_spend = (lower_bound + upper_bound) / 2
        return average_spend
    return None

# Apply the function to the 'spend' column
df['spend'] = df['spend'].apply(calculate_average_spend)

# Convert the ad_creation_time column to datetime format
df['ad_creation_time'] = pd.to_datetime(df['ad_creation_time'])

# Extract month and year from ad_creation_time
df['year_month'] = df['ad_creation_time'].dt.to_period('M')

df['year'] = df['year_month'].dt.year


# Create a new column displaying the name of the month
df['month_name'] = df['year_month'].dt.strftime('%B')

# Count the number of ads created each month
monthly_ads = df['year_month'].value_counts().sort_index()

df.drop(columns=['year_month'], inplace=True)
df.drop(columns=['ad_creative_link_descriptions'], inplace=True)


def calculate_average_audience_size(audience_str):
    if isinstance(audience_str, str):
        # Extract the lower and upper bounds using regex
        match = re.search(r'lower_bound: (\d+), upper_bound: (\d+)', audience_str)
        if match:
            lower_bound = int(match.group(1))
            upper_bound = int(match.group(2))
            # Calculate the average
            average_audience_size = (lower_bound + upper_bound) / 2
            return average_audience_size
    return None

# Apply the function to the 'estimated_audience_size' column
df['estimated_audience_size'] = df['estimated_audience_size'].apply(calculate_average_audience_size)

# Function to calculate average value from lower and upper bounds
def calculate_average_value(value_str):
    if isinstance(value_str, str):
        # Extract the lower and upper bounds using regex
        match = re.search(r'lower_bound: (\d+), upper_bound: (\d+)', value_str)
        if match:
            lower_bound = int(match.group(1))
            upper_bound = int(match.group(2))
            # Calculate the average
            average_value = (lower_bound + upper_bound) / 2
            return average_value
    return None

# Apply the function to the 'impressions' column
df['impressions'] = df['impressions'].apply(calculate_average_value)



# Convert the ad_delivery_start_time and ad_delivery_stop_time columns to datetime format
df['ad_delivery_start_time'] = pd.to_datetime(df['ad_delivery_start_time'])
df['ad_delivery_stop_time'] = pd.to_datetime(df['ad_delivery_stop_time'])

# Calculate the difference in days
df['ad_delivery_duration_days'] = (df['ad_delivery_stop_time'] - df['ad_delivery_start_time']).dt.days

# Drop the original datetime columns
df.drop(columns=['ad_delivery_start_time', 'ad_delivery_stop_time'], inplace=True)


# Function to extract the highest percentage demographic
def extract_highest_percentage(demo_str):
    try:
        # Parse the JSON-like string
        demo_list = json.loads(f'[{demo_str}]')  # Wrap in square brackets to make it a valid JSON array
        # Find the entry with the highest percentage
        highest = max(demo_list, key=lambda x: x.get('percentage', 0))
        return highest.get('age', None), highest.get('gender', None), highest.get('percentage', None)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Handle the case where parsing fails or any other error occurs
        return None, None, None

# Apply the function to the 'demographic_distribution' column
df['age'], df['gender'], df['demographic_percentage'] = zip(*df['demographic_distribution'].apply(extract_highest_percentage))

# Replace "facebook,instagram" with "both" in the publisher_platforms column
df['publisher_platforms'] = df['publisher_platforms'].replace('facebook,instagram', 'both')


# Convert the ad_creation_time column to datetime format
df['ad_creation_time'] = pd.to_datetime(df['ad_creation_time'])

# Create a new column that displays the day of the week
df['ad_creation_day'] = df['ad_creation_time'].dt.day_name()

df['ad_creative_bodies'] = df['ad_creative_bodies'].fillna('')

# Function to categorize sentiment and return binary indicators
def categorize_sentiment(text):
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        return 1  # Positive sentiment
    elif compound_score <= -0.05:
        return 2  # Negative sentiment
    elif compound_score > -0.05 and compound_score < 0.05:
        return 3  # Neutral sentiment
    else:
        return 4  # Mixed sentiment

# Apply the sentiment categorization function
df['sentiment_category'] = df['ad_creative_bodies'].apply(categorize_sentiment)

# Map sentiment categories to binary indicators (1 for the respective sentiment category and 0 otherwise)
df['positive_sentiment'] = df['sentiment_category'].apply(lambda x: 1 if x == 1 else 0)
df['negative_sentiment'] = df['sentiment_category'].apply(lambda x: 1 if x == 2 else 0)
df['neutral_sentiment'] = df['sentiment_category'].apply(lambda x: 1 if x == 3 else 0)


# Step 2: Define a function to categorize spend values based on differences from 500
def categorize_spend_difference(spend):
    difference = abs(spend - 500)
    if difference <= 100:
        return '0-100'
    elif difference <= 200:
        return '101-200'
    elif difference <= 300:
        return '201-300'
    elif difference <= 400:
        return '301-400'
    else:
        return '401+'

# Step 3: Apply the function to create a new column 'spend_difference_group'
df['spend_difference_group'] = df['spend'].apply(categorize_spend_difference)



df.drop(columns=[
    'currency', 
    'languages', 
    'byline', 
    'page_id', 
    'estimated_audience_size', 
    'demographic_distribution', 
    'delivery_by_region', 
    'ad_creative_link_titles', 
    'ad_creative_link_captions',
    'page_name',
    'ad_archive_id',
    'sentiment_category'
], inplace=True)




# Save the transformed DataFrame to a new CSV file
df.to_csv(transformed_file, index=False)
num_rows = df.shape[0]
print(f"The number of rows in the DataFrame is: {num_rows}")
# Display the first few rows of the transformed DataFrame
print(df.head())
