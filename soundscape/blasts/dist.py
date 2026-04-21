import obspy
import pyproj
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees


utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
for blast_num in [1, 2, 3]:
    if blast_num == 1:
        blast_lon_AEC = -148.66
        blast_lat_AEC = 63.98

        blast_lat = 63.9901
        blast_lon = -148.7392

    elif blast_num == 2:
        blast_lon_AEC = -148.68
        blast_lat_AEC = 63.97

        blast_lat = 63.9901
        blast_lon = -148.7392

    elif blast_num == 3:
        blast_lon_AEC = -148.765
        blast_lat_AEC = 64.010

        blast_lat = 63.9901
        blast_lon = -148.7392


    #convert blast location to UTM
    blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)
    blast_AEC_utm_x, blast_AEC_utm_y = utm_proj(blast_lon_AEC, blast_lat_AEC)
    #find distance between station and blast location
    dist_km = np.sqrt((blast_AEC_utm_x-blast_utm_x)**2 +(blast_AEC_utm_y-blast_utm_y)**2)/1000
    print(dist_km)

    if blast_num == 3:
        dist_hold = np.Infinity
        with open('parks_highway.txt', 'r') as f:
            next(f, None)  # skip header / first line
            for line in f:
                lon, lat = line.strip().split(',')
                lat = float(lat)
                lon = float(lon)
                park_utm_x, park_utm_y = utm_proj(lon, lat)
                dist_km_park = np.sqrt((park_utm_x-blast_utm_x)**2 +(park_utm_y-blast_utm_y)**2)/1000
                if dist_km_park < dist_hold:
                    dist_hold = dist_km_park
        print(dist_hold)


dist_hold = np.Infinity
sta_hold = None
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])

for i, station in enumerate(stations):

        dist_km = np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000
        if dist_km < dist_hold:
            dist_hold = dist_km
            sta_hold = station
print(sta_hold, dist_hold)