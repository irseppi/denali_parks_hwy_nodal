import pandas as pd
import numpy as np

nodes = STATIONS_txt = "/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt"
seismo_data = pd.read_csv(nodes, sep="|")

stations = seismo_data['Station'].astype(int).values

station_check = np.arange(1001, 1307)  
missing_stations = [station for station in station_check if station not in stations and station <= 1305]
print("Missing stations between 1001 and 1305:")
print(missing_stations)
#Print number of not missing stations
print("Number of stations between 1001 and 1305:", len(station_check) - len(missing_stations))
station_check = np.arange(1500, 1590)  
missing_stations = [station for station in station_check if station not in stations and station >= 1500]
print("Missing stations between 1500 and 1590:")
print(missing_stations)
#Print number of not missing stations
print("Number of stations between 1500 and 1590:", len(station_check) - len(missing_stations))

#Print any station that are not in those ranges but are in the stations list
other_stations = [station for station in stations if station < 1001 or 
                  (station > 1306 and station < 1500) or station > 1590]

print("Stations that are not between 1001-1305 or 1500-1590:")
print(other_stations)
print("Number of stations that are not between 1001-1305 or 1500-1590:", len(other_stations))
