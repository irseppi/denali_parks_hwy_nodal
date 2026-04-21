import obspy
import os
import pyproj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel, taup_create
from scipy.signal import spectrogram


blast_num = 3


utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
taup_create.build_taup_model("brocher2004.tvel")
model = TauPyModel(model="brocher2004")
folder = '/scratch/naalexeev/NODAL/'

# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
stations = seismo_data['Station']
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])


for i, station in enumerate(stations):

    blast_lat = 63.9901
    blast_lon = -148.7392
    depth = 0
    blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)
    if blast_num == 1:
        c = 328 / 1000
        file = folder + f'2019-02-16T01:00:00.000000Z.2019-02-16T02:00:00.000000Z.{station}.mseed'
    elif blast_num == 2:
        c = 317 / 1000
        file = folder + f'2019-02-23T02:00:00.000000Z.2019-02-23T03:00:00.000000Z.{station}.mseed'
    elif blast_num == 3:
        c =  332 / 1000  
        file = folder + f'2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.{station}.mseed'
    #if file exists
    if not os.path.exists(file):
        continue
    dist_km = np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000
    #if dist_km > 68:
    #    continue
    arrivals_P = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["P"])
    arrivals_S = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["S"])
    try:
        P_a = arrivals_P[0].time
        S_a = arrivals_S[0].time
    except IndexError:
        continue
    print(f'Station: {station}, Distance (km): {dist_km:.2f}, P arrival time (s): {P_a:.2f}, S arrival time (s): {S_a:.2f}')
    tr = obspy.read(file)
    # Compute hour long spectrogram for noise removal
    data = tr[2][:]
    fs = int(tr[2].stats.sampling_rate)
    frequencies_lta, times_lta, Sxx_lta = spectrogram(data, fs)
    window = 800

    if blast_num == 1:
        tr[2].trim(tr[2].stats.starttime + (52 * 60) + 21 ,tr[2].stats.starttime + (52 * 60) + 21 + window)
    elif blast_num == 2:
        tr[2].trim(tr[2].stats.starttime + (11 * 60) + 12 ,tr[2].stats.starttime + (11 * 60) + 12 + window)
    elif blast_num == 3:
        tr[2].trim(tr[2].stats.starttime + (42 * 60) + 19 ,tr[2].stats.starttime + (42 * 60) + 19 + window)

    data = tr[2][:]
    fs = int(tr[2].stats.sampling_rate)
    title = f'{tr[2].stats.network}.{tr[2].stats.station}.{tr[2].stats.location}.{tr[2].stats.channel} − starting {tr[2].stats["starttime"]}'                        
    t_wf = tr[2].times()

    # Compute spectrogram
    frequencies, times, Sxx = spectrogram(data, fs, scaling='spectrum', nperseg=fs, noverlap=fs * .90, detrend = 'constant') 
    spec = 10 * np.log10(Sxx) 


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6)) 

    ax1.plot(t_wf, data, 'k', linewidth=0.5)
    ax1.set_title(title)
    ax1.set_ylabel('Amplitude, Counts')
    ax1.set_position([0.125, 0.6, 0.775, 0.3]) 

    # Plot spectrogram
    cax = ax2.pcolormesh(times, frequencies, spec, cmap='plasma_r', vmin = 0, vmax = 45) 
    ax2.set_xlabel('Time, s')
    ax2.set_ylabel('Frequency, Hz')

    ax3 = fig.add_axes([0.9, 0.11, 0.015, 0.376])
    plt.colorbar(mappable=cax, cax=ax3)
    ax3.set_ylabel('Relative Amplitude, dB')
    ax2.set_ylim(0,250)

    for a in [ax1, ax2]:
        a.margins(x=0)
        a.set_xlim((dist_km/c)-200, (dist_km/c)+300)
        if a == ax2:
            a.axvline(P_a, ymin=.8,ymax=1, color='red', linestyle='-', linewidth=0.7)
            a.axvline(S_a, ymin=.8, ymax=1, color='#984ea3', linestyle='dashdot', linewidth=1)
            a.axvline((dist_km/c), ymin=.8,ymax=1, color='b', linestyle='--', linewidth=0.7)
        else:
            a.axvline(P_a, color='red', linestyle='-', linewidth=0.7)
            a.axvline(S_a, color='#984ea3', linestyle='dashdot', linewidth=1)
            a.axvline((dist_km/c), color='b', linestyle='--', linewidth=0.7)

    plt.subplots_adjust(hspace=0.05)

    plt.show()
    plt.close(fig)
