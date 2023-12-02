from datetime import datetime, timedelta
import pandas as pd
from meteostat import Point, Daily

# Set time period
start =  datetime(2000, 1, 4)
end = datetime(2023, 10, 30)

# Read the CSV file
df = pd.read_csv("state_data/closest_stations_to_centroids.csv")

# Initialize an empty list to store DataFrames
all_data = []

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    # Create a Point for the station
    location = Point(row['latitude'], row['longitude'], row['elevation'])

    # Fetch daily weather data
    data = Daily(location, start, end)
    data = data.fetch()

    # If the index is not a DatetimeIndex, set it as such
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Resample the data to weekly, aligning on Tuesday
    weekly_data = data.resample('W-TUE').mean()


    # Add the state abbreviation column
    weekly_data['state_abbr'] = row['state']

    # Append the DataFrame to the list
    all_data.append(weekly_data)

# Concatenate all DataFrames into one
combined_data = pd.concat(all_data)

# Save the combined DataFrame to a CSV file
combined_data.to_csv("datasets/weather.csv", index=True)
    
