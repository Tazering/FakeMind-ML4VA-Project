import pandas as pd
import numpy as np

# Load the datasets
state_centroids = pd.read_csv('datasets/state_centroids.csv')
operational_stations = pd.read_csv('datasets/operational_stations_1999_2015.csv')

# Define the haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Iterate over each state's centroid and find the closest station
closest_stations = pd.DataFrame()

for index, centroid in state_centroids.iterrows():
    state = centroid['state']
    lat1 = centroid['latitude']
    lon1 = centroid['longitude']
    
    # Calculate the distance to all stations
    operational_stations['distance'] = operational_stations.apply(
        lambda row: haversine(lat1, lon1, row['latitude'], row['longitude']), axis=1)
    
    # Find the closest station for the state
    closest_station = operational_stations.loc[operational_stations['distance'].idxmin()]
    
    # Append the closest station to the DataFrame
    closest_station['state'] = state  # Add the state identifier
    closest_stations = closest_stations.append(closest_station, ignore_index=True)

# Export to CSV
closest_stations.to_csv('datasets/closest_stations_to_centroids.csv', index=False)
