import pandas as pd
from collections import Counter
import re



raw_file = "/home/chetan/ads data analysis/ads analysis/cleaned_data.csv"
transformed_file = "/home/chetan/ads data analysis/ads analysis/processed.csv"


# Define the path to the CSV file
csv_file_path = raw_file

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)



# Create DataFrame
df = pd.DataFrame(df)

def get_top_keywords(text, n=3):
    if not isinstance(text, str):
        return ''
    
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\[\d+:\d+:[^\]]+\]', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase

    # Tokenize and count word frequencies
    words = text.split()
    word_counts = Counter(words)

    # Get the n most common words
    top_keywords = [word for word, _ in word_counts.most_common(n)]

    return ', '.join(top_keywords)

# Apply the function to the DataFrame
df['top_keywords'] = df['ad_creative_bodies'].apply(get_top_keywords)

# Assuming df is your DataFrame
# Replace NaN values in 'spend' and 'impressions' columns with 0
df['spend'].fillna(0, inplace=True)
df['impressions'].fillna(0, inplace=True)

# Drop rows where 'top_keywords' column has NaN values
df.dropna(subset=['top_keywords'], inplace=True)

# Save the transformed DataFrame to a new CSV file
df.to_csv(transformed_file, index=False)
print(df)
