import obspy
import pyproj
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel
from obspy.geodetics import kilometer2degrees
from obspy.taup import TauPyModel, taup_create
from scipy.signal import spectrogram

utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
taup_create.build_taup_model("brocher2004.tvel")
model = TauPyModel(model="brocher2004")

folder = '/scratch/naalexeev/NODAL/'
# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])

blast_lat = 63.9901
blast_lon = -148.7392
depth = 0

blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)

for blast_num in [1,2,3]:

    if blast_num == 1:
        c = 328 / 1000  # Speed of sound in km/s, used for calculating blast arrival times
        sta = 1257
        file = folder + f'2019-02-16T01:00:00.000000Z.2019-02-16T02:00:00.000000Z.{sta}.mseed'
    elif blast_num == 2:
        c = 317 / 1000 
        sta = 1179 #1192
        file = folder + f'2019-02-23T02:00:00.000000Z.2019-02-23T03:00:00.000000Z.{sta}.mseed'
    elif blast_num == 3:
        c =  332 / 1000  
        sta = 1248
        file = folder + f'2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.{sta}.mseed'
    
    i = stations[stations == sta].index[0]

    dist_km = np.sqrt((np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000)**2 + depth**2)
    arrivals_P = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["P"])
    arrivals_S = model.get_travel_times(source_depth_in_km=depth,distance_in_degree = kilometer2degrees(dist_km),phase_list=["S"])

    P_a = arrivals_P[0].time
    S_a = arrivals_S[0].time

    tr = obspy.read(file)
    chan = 2 # 0 is channel 1, 1 is channel 2, 2 is channel Z
    window = 150
    if blast_num == 1:
        tr[chan].trim(tr[chan].stats.starttime + (52 * 60) + 21, tr[chan].stats.starttime + (52 * 60) + 21 + window)
        start_time_plot = tr[chan].stats.starttime 

    elif blast_num == 2:
        tr[chan].trim(tr[chan].stats.starttime + (11 * 60) + 12 , tr[chan].stats.starttime + (11 * 60) + 12 + window)
        start_time_plot = tr[chan].stats.starttime 

    elif blast_num == 3:
        tr[chan].trim(tr[chan].stats.starttime + (42 * 60) + 20 , tr[chan].stats.starttime + (42 * 60) + 20 + window)
        start_time_plot = tr[chan].stats.starttime 

    data = tr[chan][:]
    fs = int(tr[chan].stats.sampling_rate)
    start_time_str = (start_time_plot).strftime("%Y-%m-%dT%H:%M:%S")

    title = (
        f"{tr[chan].stats.network}.{tr[chan].stats.station}."
        f"{tr[chan].stats.location}.{tr[chan].stats.channel} − starting {start_time_str}"
    )
    t_wf = tr[chan].times()

    # Compute spectrogram
    frequencies, times, Sxx = spectrogram(data, fs, scaling='spectrum', nperseg=fs // 2 , noverlap=(fs // 2) * .90, detrend = 'constant') 
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
        a.set_xlim(0, window)
        if a == ax2:
            a.axvline(P_a, ymin=.8, ymax=1, color='red', linestyle='-', linewidth=0.7)

            a.axvline(S_a, ymin=.8, ymax=1, color='#984ea3', linestyle='dashdot', linewidth=1)

            a.axvline((dist_km/c), ymin=.8, ymax=1, color='b', linestyle='--', linewidth=0.7)

        else:
            a.axvline(P_a, color='red', linestyle='-', linewidth=0.7)
            a.text(P_a - 1, 0.9, 'P', color='k', ha='right', va='bottom', transform=a.get_xaxis_transform(), fontsize=10)

            a.axvline(S_a, color='#984ea3', linestyle='dashdot', linewidth=1)
            a.text(S_a - 1, 0.9, 'S', color='k', ha='right', va='bottom', transform=a.get_xaxis_transform(), fontsize=10)

            a.axvline((dist_km/c), color='b', linestyle='--', linewidth=0.7)
            a.text((dist_km/c) - 1, 0.9, r"$c_{eff}$", color='k', ha='right', va='bottom', transform=a.get_xaxis_transform(), fontsize=10)
    #make the space between the two subplots 0
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(f'blast{blast_num}_spec.png', dpi=400)

    check_mean_spec = False
    if check_mean_spec:
        plt.figure()
        mean_spec = np.mean(Sxx, axis=1)
        plt.plot(frequencies, mean_spec)
        plt.yscale('log')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.show()
