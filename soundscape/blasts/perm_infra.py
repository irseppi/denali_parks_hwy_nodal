from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from scipy.signal import spectrogram
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from obspy.geodetics.base import gps2dist_azimuth
times = [
    "2019-01-08 22:26",
    "2019-01-10 00:44",
    "2019-01-11 00:55",
    "2019-01-19 00:17",
    "2019-02-01 23:15",
    "2019-02-16 01:52",
    "2019-02-23 02:11",
    "2019-03-02 01:42",
    "2019-03-31 23:48",
    "2019-04-13 00:19"
]

blast_coords = [
    (-148.69, 63.92),
    (-148.86, 63.91),
    (-148.92, 63.91),
    (-148.71, 63.99),
    (-148.96, 63.92),
    (-148.66, 63.98),
    (-148.68, 63.97),
    (-148.76, 64.01),
    (-148.67, 63.95),
    (-148.74, 63.99)
]

north_vel = [
    319.37018975355494,
    319.35166364611024,
    315.70844127219544,
    319.6798426719846,
    317.4844939422882,
    323.04169310941063,
    323.04169310941063,
    332.64352290018076,
    336.27782333278856,
    342.70697608369454
]

utm_proj = pyproj.Proj(proj='utm', zone=6, ellps='WGS84')

client = Client("http://service.iris.edu") 
stations = ["NEA2","CUT"]

for i, time in enumerate(times):
    if time != "2019-03-02 01:42":
        continue 
    for station in stations:
        #blast_lon, blast_lat = blast_coords[i]
        blast_lat = 63.9901
        blast_lon = -148.7392
        blast_utm_x_m, blast_utm_y_m = utm_proj(blast_lon, blast_lat)
        blast_utm_x = blast_utm_x_m / 1000.0
        blast_utm_y = blast_utm_y_m / 1000.0

        c = north_vel[i] / 1000
        if i == 5 and station == "NEA2":
            c = 328 / 1000  # Speed of sound in km/s, used for calculating blast arrival times
        elif i == 6 and station == "CUT":
            c = 317 / 1000 
        elif i == 7 and station == "NEA2":
            c =  332 / 1000 
        elif i == 7 and station == "CUT":
            c = 326 / 1000

        starttime = UTCDateTime(time)

        endtime = starttime + 1000
        # Request station-level inventory
        inventory = client.get_stations(network="AK", station=station, 
                                        starttime=starttime, endtime=endtime,
                                        level="station")
        
        # Extract and print coordinates
        for net in inventory:
            for sta in net:
                lat = sta.latitude
                lon = sta.longitude
                utm_x_m, utm_y_m = utm_proj(lon, lat)
                utm_x = utm_x_m / 1000.0
                utm_y = utm_y_m / 1000.0
        dist_km = np.sqrt((blast_utm_x-utm_x)**2 +(blast_utm_y-utm_y)**2)

        arrival = dist_km / c

        true_baz = gps2dist_azimuth(lat, lon, blast_coords[i][1], blast_coords[i][0])[1]
        print(f"True backazimuth Station {station} Blast{i}: {true_baz:.1f} degrees")

        st = client.get_waveforms("AK", station, "*", "BDF", starttime, endtime)
        tr = st[0]
        data = tr.data
        t_wf = tr.times()
        fs = int(tr.stats.sampling_rate)
        print(f"Sampling rate of {station}: {fs} Hz")
        title = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel} − starting {tr.stats["starttime"]}'
        frequencies, times, Sxx = spectrogram(data, fs, scaling='spectrum') 
        spec = 10 * np.log10(Sxx) 

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6)) 
        tr_filt = tr.copy()
        tr_filt.filter("bandpass", freqmin=0.5, freqmax=3, corners=4, zerophase=True)

        tr_filt.detrend(type='constant')
        tr_filt.taper(max_percentage=0.01)
        ax1.plot(tr_filt.times(), tr_filt.data, 'k', linewidth=0.5)
        ax1.set_title(title)
        ax1.set_ylabel('Amplitude, Counts')
        ax1.set_position([0.125, 0.6, 0.775, 0.3]) 
        ax1.axvline(x=arrival, color='r', linestyle='--', label='Expected Blast Arrival')
        # Plot spectrogram
        cax = ax2.pcolormesh(times, frequencies, spec, cmap='plasma_r', vmin = 0, vmax = np.max(spec))

        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('Frequency, Hz')

        ax3 = fig.add_axes([0.9, 0.11, 0.015, 0.376])
        plt.colorbar(mappable=cax, cax=ax3)
        ax3.set_ylabel('Relative Amplitude, dB')
        # scale y-axis to max absolute amplitude inside the plotted x-range
        x0, x1 = arrival - 300, arrival + 300
        t_vec = tr_filt.times()
        mask = (t_vec >= x0) & (t_vec <= x1)
        if np.any(mask):
            y_max = np.nanmax(np.abs(tr_filt.data[mask]))
        else:
            y_max = np.nanmax(np.abs(tr_filt.data))
        if y_max <= 0:
            y_max = 1.0
        ax1.set_ylim(-y_max * 1.1, y_max * 1.1)
        ax2.set_ylim(0,fs/2)
        ax1.set_xlim(arrival - 200, arrival + 300)
        ax2.set_xlim(arrival - 200, arrival + 300)
        #make the space between the two subplots 0
        plt.subplots_adjust(hspace=0.05)

        plt.savefig(f'perm_fig/blast_{station}_{time}.png', dpi=400)
