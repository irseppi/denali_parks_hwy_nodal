from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import pyproj

utm_proj = pyproj.Proj(proj='utm', zone=6, ellps='WGS84')
client = Client("http://service.iris.edu") 

times = ["2019-02-16T01:52:21","2019-02-23T02:11:12","2019-03-02T01:42:20"]

blast_coords = [
    (-148.66, 63.98),
    (-148.68, 63.97),
    (-148.76, 64.01),
]

for i, (source_lon, source_lat) in enumerate(blast_coords):

    NETWORK = 'IM'
    STATION = 'I53H?'
    LOCATION = '*'
    CHANNEL = 'BDF'
    starttime = UTCDateTime(times[i])

    endtime = starttime + 250
    inventory = client.get_stations(network=NETWORK, station=STATION, 
                                    starttime=starttime, endtime=endtime,
                                    level="station")
    baz_array = []
    # Extract and print coordinates
    for net in inventory:
        for sta in net:
            station_lat = sta.latitude
            station_lon = sta.longitude

            # Calculate distance (m), azimuth (degrees), and back azimuth (degrees)
            dist, az, baz = gps2dist_azimuth(source_lat, source_lon, 
                                 station_lat, station_lon)
            baz_array.append(baz)

    print(f"Median Back Azimuth: {np.median(baz_array):.2f}°")
