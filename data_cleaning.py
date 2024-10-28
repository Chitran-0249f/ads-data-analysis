import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json


# calculate average spend, Extract month and year, column displaying the name of the month, calculate average 'impressions' column, Calculate the difference in days

raw_file = "/home/chetan/ads data analysis/dataset/dataset-40k.csv"
transformed_file = "/home/chetan/ads data analysis/dataset/transformed.csv"


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
        highest = max(demo_list, key=lambda x: x['percentage'])
        return highest['age'], highest['gender'], highest['percentage']
    except json.JSONDecodeError:
        # Handle the case where parsing fails
        return None, None, None

# Apply the function to the 'demographic_distribution' column
df['age'], df['gender'], df['demographic_percentage'] = zip(*df['demographic_distribution'].apply(extract_highest_percentage))

# Drop the original demographic_distribution column if needed
df.drop(columns=['demographic_distribution'], inplace=True)

# Convert the ad_creation_time column to datetime format
df['ad_creation_time'] = pd.to_datetime(df['ad_creation_time'])

# Create a new column that displays the day of the week
df['ad_creation_day'] = df['ad_creation_time'].dt.day_name()

df.drop(columns=['delivery_by_region'], inplace=True)
df.drop(columns=['currency'], inplace=True)
df.drop(columns=['languages'], inplace=True)
df.drop(columns=['byline'], inplace=True)
df.drop(columns=['page_id'], inplace=True)




#DATA VISUALISATION

# Plot the monthly ad creation counts
plt.figure(figsize=(10, 6))
monthly_ads.plot(kind='bar')
plt.title('Monthly Ad Creation Counts')
plt.xlabel('Month')
plt.ylabel('Number of Ads Created')
plt.xticks(rotation=45)
plt.grid(axis='y')



# Save the plot to a file
plt.tight_layout()
plt.savefig('dataset/meta-ad-library-08_07_2024/meta-ad-library-08/07/plots.png')

# Save the transformed DataFrame to a new CSV file
df.to_csv(transformed_file, index=False)
num_rows = df.shape[0]
print(f"The number of rows in the DataFrame is: {num_rows}")
# Display the first few rows of the transformed DataFrame
print(df.head())
