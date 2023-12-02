

    # Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Stations
import pandas as pd

# Set time period
start = datetime(2014, 11, 1)
end = datetime(2014, 12, 31)

df = pd.read_csv(r"state_data\\closest_stations_to_centroids.csv")



for index, row in df.iterrows():


    data = Daily(loc=row['wmo'], start=start, end=end)
    data = data.fetch()

    print(data)

    data.to_csv(r"data\\" + row['state'] + ".csv", index=False)
