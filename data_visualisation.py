import pandas as pd
from pandas_profiling import ProfileReport

# Define the path to the CSV file
csv_file_path = '/home/chetan/ads data analysis/dataset/cleaned_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Generate the Pandas Profiling report
profile = ProfileReport(df, title="Ads Data Analysis Report", explorative=True)

# Save the report as an HTML file
profile.to_file("/home/chetan/ads data analysis/dataset/ads_data_analysis_report.html")
