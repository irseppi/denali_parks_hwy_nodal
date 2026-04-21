import numpy as np
from matplotlib import pyplot as plt
from soundscape_psd_functions import load_stations_for_distance_calculation
from pyproj import Proj
utm_proj = Proj(proj='utm', zone='6', ellps='WGS84')

seismic_sta = load_stations_for_distance_calculation(fullstations=True)
elevation = seismic_sta['Elevation']
station = seismic_sta['Station']
lat = seismic_sta['Latitude']
lon = seismic_sta['Longitude']

plot_num = 3 #1 is highway array, 2 is fault array, 3 is plot both
if plot_num == 1 or plot_num == 3:
    dist_array = []
    #extract lat and lon of station 1163 and convert to utm
    i = np.where(station == 1163)[0][0]
    lat_1163 = seismic_sta['Latitude'][i]
    lon_1163 = seismic_sta['Longitude'][i]
    center_utm_x, center_utm_y = utm_proj(lon_1163, lat_1163)
    for i in range(len(station)):
        utm_x, utm_y = utm_proj(lon[i], lat[i])
        dist_km = np.sqrt((utm_x - center_utm_x) ** 2 + (utm_y - center_utm_y) ** 2) / 1000
        if station[i] < 1167:
            dist_km = -dist_km
        dist_array.append(dist_km)


    mask = station < 1307
    elevation_1 = elevation[mask]
    station_1 = station[mask]
    dist_1 = np.array(dist_array)[mask]

    plt.figure(figsize=(10, 4))
    for p in station_1:
        if p in (1306, 1001):  # no need to cast to str
            offset = 20
            x = float(dist_1[station_1 == p])
            y = float(elevation_1[station_1 == p]) + offset
            if p == 1306:
                plt.text(x, y, str(p), ha='left')
            elif p == 1001:
                plt.text(x, y, str(p), ha='right')
        if str(p) == "1304":
            offset = 100
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Nenana", ha='center')

        elif str(p) == "1256":
            offset = 100
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "NFFTB", ha='center')

        elif str(p) == "1215":
            offset = 100
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Healy", ha='center')

        elif str(p) == "1199":
            offset = 150
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Denali\nPark Rd", ha='center')

        elif str(p) == "1163":
            offset = 75
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Denali\nFault", ha='center')

        elif str(p) == "1153":
            offset = 110
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Cantwell", ha='center')

        elif str(p) == "1003":
            offset = 100
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Trapper\nCreek", ha='center')

        elif str(p) == "1278":
            offset = 110
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Anderson", ha='center')

        elif str(p) == "1022": 
            offset = 110
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "Susitna\nBasin", ha='center')

        elif str(p) == "1219":
            offset = 100
            plt.text(dist_1[station_1 == p], elevation_1[station_1 == p] + offset, "HCF", ha='left') #Healy Creek fault
        else:
            continue
        x = float(dist_1[station_1 == p])
        y_label = float(elevation_1[station_1 == p]) + offset
        ax = plt.gca()
        bottom = ax.get_ylim()[0]
        offset2 = (y_label - bottom) * 0.05 # small gap below the text
        ax.plot([x, x], [y_label - offset2, bottom], color='k', linestyle='--', linewidth=1, zorder=1000)

    plt.scatter(dist_1, elevation_1, c='k', marker='o')
    plt.ylabel('Elevation, m')
    plt.ylim(50, 800)
    plt.subplots_adjust(top=0.963, bottom=0.117, left=0.066, right=0.985, hspace=0.2, wspace=0.2)
    plt.xlabel('Distance from Denali Fault Crossing, km')
    plt.savefig('elevation_vs_distance_highway.png', dpi=400)

if plot_num == 2 or plot_num == 3:
    #extract lat and lon of station 5575 and convert to utm
    i = np.where(station == 5575)[0][0]
    lat_5575 = seismic_sta['Latitude'][i]
    lon_5575 = seismic_sta['Longitude'][i]
    center_utm_x, center_utm_y = utm_proj(lon_5575, lat_5575)

    dist = []
    for i in range(len(station)):
        utm_x, utm_y = utm_proj(lon[i], lat[i])
        dist_km = np.sqrt((utm_x - center_utm_x) ** 2 + (utm_y - center_utm_y) ** 2) / 1000
        if lat[i] < lat_5575:
            dist_km = -dist_km
        dist.append(dist_km)

    mask = station >= 1500
    elevation_2 = elevation[mask]
    station_2 = station[mask]
    dist_2 = np.array(dist)[mask]

    plt.figure(figsize=(10, 4))
    for p in station_2:
        if str(p) == "5575":
            offset = 20
            plt.text(dist_2[station_2 == p], elevation_2[station_2 == p] + offset, "Denali\nFault", ha='center')
            x = float(dist_2[station_2 == p])
            y_label = float(elevation_2[station_2 == p]) + offset
            ax = plt.gca()
            bottom = ax.get_ylim()[0]
            offset2 = (y_label - bottom) * 0.005 # small gap below the text
            ax.plot([x, x], [y_label - offset2, bottom], color='k', linestyle='--', linewidth=1, zorder=1000)
        if p == 1500 or p == 1589: # label the first and last station in the array
            plt.text(dist_2[station_2 == p], elevation_2[station_2 == p] - 8, str(p), ha='center')
        if p > 1591 and p != 5575:
            plt.text(dist_2[station_2 == p], elevation_2[station_2 == p] - 8, str(p), ha='center')
    plt.scatter(dist_2, elevation_2, c='k', marker='o')
    plt.subplots_adjust(top=0.963, bottom=0.117, left=0.066, right=0.985, hspace=0.2, wspace=0.2)
    plt.ylim(600, 750)
    plt.ylabel('Elevation, m')
    plt.xlabel('Distance from Denali Fault Crossing, km')
    plt.savefig('elevation_vs_distance_fault.png', dpi=400)
