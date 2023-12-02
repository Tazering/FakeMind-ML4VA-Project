import pandas as pd

# Load the weather data
weather_df = pd.read_csv("datasets/weather.csv")

# Convert 'time' to datetime, then to date (string format)
weather_df['time'] = pd.to_datetime(weather_df['time'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')

weather_df = weather_df.drop('wpgt', axis=1)

# Load the drought data
drought_df = pd.read_csv(r"datasets\USA_Drought_Intensity_2000_-_Present.csv")

# Convert 'ddate' to datetime, then to date (string format)
drought_df['ddate'] = pd.to_datetime(drought_df['ddate']).dt.strftime('%Y-%m-%d')

# Sort drought data by 'ddate' and 'state_abbr'
drought_df.sort_values(by=['ddate', 'state_abbr'], inplace=True)
print("\nSorted Drought Data (first 100 rows):")
print(drought_df.head(100))

# Merge the DataFrames on 'state_abbr' and 'time'/'ddate' using a left join
merged_df = pd.merge(drought_df, weather_df, how='left', left_on=['state_abbr', 'ddate'], right_on=['state_abbr', 'time'])

# Drop the 'period' column
merged_df.drop('period', axis=1, inplace=True)

# Sort by 'OBJECTID'
merged_df.sort_values(by='OBJECTID', inplace=True)

# Print the first few rows of the sorted and modified merged DataFrame
print("\nFinal Merged DataFrame (first 5 rows):")
print(merged_df.head())

# Save the merged DataFrame
merged_df.to_csv("datasets/merged_weather_drought.csv", index=False)

print("\nFinal merged data saved to merged_weather_drought.csv")
