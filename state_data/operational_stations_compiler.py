# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Stations
import pandas as pd


# List of state codes to iterate over
states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# Initialize an empty DataFrame to store all operational stations
all_operational_stations = pd.DataFrame()

# Iterate over each state to fetch and filter stations
for state in states:
    # Fetch stations in the state
    stations = Stations()
    stations = stations.region('US', state=state)
    stations_df = stations.fetch(stations.count())

    # Convert the 'daily_start' and 'daily_end' columns to datetime
    stations_df['daily_start'] = pd.to_datetime(stations_df['daily_start'])
    stations_df['daily_end'] = pd.to_datetime(stations_df['daily_end'])

    # Filter stations based on the availability of daily data from at least 1999 to 2015
    mask = (stations_df['daily_start'] <= datetime(1999, 1, 1)) & \
           (stations_df['daily_end'] >= datetime(2015, 12, 31))
    filtered_stations = stations_df[mask]

    # Add a 'state' column to the filtered DataFrame
    filtered_stations['state'] = state

    # Append the filtered stations to the all_operational_stations DataFrame
    all_operational_stations = pd.concat([all_operational_stations, filtered_stations], ignore_index=True)

# Export the DataFrame to a CSV file
all_operational_stations.to_csv('operational_stations_1999_2015.csv', index=False)


