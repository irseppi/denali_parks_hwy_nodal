import obspy
import pyproj
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from obspy.taup import TauPyModel, taup_create
from obspy.geodetics import kilometer2degrees
from matplotlib.ticker import FuncFormatter

folder = '/scratch/naalexeev/NODAL/'
normalize_trace_individually = False
utm_proj = pyproj.Proj(proj='utm', zone='6', ellps='WGS84')
taup_create.build_taup_model("brocher2004.tvel")
model = TauPyModel(model="brocher2004")

# Load the seismometer location data
seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/parkshwy_nodes.txt', sep="|")
seismo_latitudes = seismo_data['Latitude']
seismo_longitudes = seismo_data['Longitude']
stations = seismo_data['Station']
bad_data = [1266,1265,1267,1270,1261,1262,1269,1266,1268,1259,1263,1260,1264,1165,1163,1162,1211,1180,1145,1146,1291,1172,1171,1140,1151,1232,1233,1227,1220,1221,1222,1217,1219,1223,1229,1224,1225,1235,1230,1228,1226,1207,1206,1209,1234,1200,1153,1152,1286,1218,1216]
seismo_utm_x, seismo_utm_y = zip(*[utm_proj(lon, lat) for lat, lon in zip(seismo_latitudes, seismo_longitudes)])

for scale_num, blast_num in enumerate([1,2,3,3]):
    blast_lat = 63.9901
    blast_lon = -148.7392
    depth = 0
    road_lat = 63.9901
    road_lon =  -149.128404464516

    #convert blast location to UTM
    blast_utm_x, blast_utm_y = utm_proj(blast_lon, blast_lat)
    road_utm_x, road_utm_y = utm_proj(road_lon, road_lat)
    dist_2_road =  np.sqrt((blast_utm_x - road_utm_x)**2 + (blast_utm_y - road_utm_y)**2) / 1000  

    S_wave_dict = {}
    P_wave_dict = {}
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[(65-17.8)/(74-17), 1], sharex=True)
    for i, station in enumerate(stations):
        if int(station) in bad_data:
            continue

        #find distance between station and blast location
        dist_km = np.sqrt((seismo_utm_x[i]-blast_utm_x)**2 +(seismo_utm_y[i]-blast_utm_y)**2)/1000

        if blast_num == 1:
            file = folder + f'2019-02-16T01:00:00.000000Z.2019-02-16T02:00:00.000000Z.{station}.mseed'
        elif blast_num == 2:
            file = folder + f'2019-02-23T02:00:00.000000Z.2019-02-23T03:00:00.000000Z.{station}.mseed'
        elif blast_num == 3:
            file = folder + f'2019-03-02T01:00:00.000000Z.2019-03-02T02:00:00.000000Z.{station}.mseed'
        chan = 2 # 0 is channel 1, 1 is channel 2, 2 is channel Z
        if Path(file).exists():
            tr = obspy.read(file)
            if blast_num == 1:
                tr[chan].trim(tr[chan].stats.starttime + (52 * 60) + 21 ,tr[chan].stats.starttime + (52 * 60) + 21 + 300)
            elif blast_num == 2:
                tr[chan].trim(tr[chan].stats.starttime + (11 * 60) + 12 ,tr[chan].stats.starttime + (11 * 60) + 12 + 300)
            elif blast_num == 3:
                tr[chan].trim(tr[chan].stats.starttime + (42 * 60) + 20 ,tr[chan].stats.starttime + (42 * 60) + 20 + 300)

            tr[chan].filter('bandpass', freqmin=0.5, freqmax=3, corners=2, zerophase=True)
            tr[chan].detrend(type='constant')
            tr[chan].taper(max_percentage=0.01)
            start = tr[chan].stats.starttime
            data = tr[chan][:]
            t_wf = tr[chan].times()
        else:
            continue

        if scale_num < 3:
            norm_data = (data / np.max(np.abs(data))) * 1.5
            file_save = f'blast_{blast_num}_record_section.png'
        else:
            norm_data = data / 500
            file_save = f'Blast_2019-03-02T01:42:19.pdf'

        if seismo_latitudes[i] >= blast_lat:
            axs[0].plot(t_wf, norm_data + dist_km, 'k', linewidth=0.5)
            dist_km = dist_km
        elif seismo_latitudes[i] < blast_lat:
            axs[1].plot(t_wf, norm_data - dist_km, 'k', linewidth=0.5)
            dist_km = -dist_km
            if int(station) in [1231]:
                axs[0].plot(t_wf, norm_data + dist_2_road + (dist_km + dist_2_road), 'k', linewidth=0.5)
        
        arrivals_P = model.get_travel_times(source_depth_in_km=0,distance_in_degree = kilometer2degrees(dist_km),phase_list=["P"])
        arrivals_S = model.get_travel_times(source_depth_in_km=0,distance_in_degree = kilometer2degrees(dist_km),phase_list=["S"]) 

        try:
            go = arrivals_P[0].time
            go = arrivals_S[0].time
        except:
            continue

        P_wave_dict[dist_km] = arrivals_P[0].time
        S_wave_dict[dist_km] = arrivals_S[0].time

        if int(station) in [1231]:
            P_wave_dict[-dist_km] = arrivals_P[0].time
            S_wave_dict[-dist_km] = arrivals_S[0].time

    for gg, dir in enumerate(['north', 'south']):
        axs[gg].plot(P_wave_dict.values(), P_wave_dict.keys(), color='red', linestyle='-', linewidth=1, zorder=0)
        axs[gg].plot(S_wave_dict.values(), S_wave_dict.keys(), color='#984ea3', linestyle='dashdot', linewidth=1, zorder=0)
        file_path = f'blast_{blast_num}_picks_{dir}.txt'
        if not Path(file_path).exists():
            continue
        x_top = []
        y_top = []
        with open(file_path, 'r') as f:
            for line in f:
                xb, yb, tb, _ = line.strip().split(',')
                x_top.append(float(xb))
                y_top.append(float(yb))
        x_arr = np.array(x_top)
        y_arr = np.array(y_top)
        m = np.linalg.lstsq(x_arr[:, np.newaxis], y_arr, rcond=None)[0][0]
        if gg == 0:
            slope_top = m
            axs[0].text(215, 62, r"$c_{eff}$" + f" =  {slope_top*1000:.0f} m/s", fontsize=12, color='b',
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=2))
        elif gg == 1:
            slope_bottom = m
            m = -m
            axs[1].text(215, -67, r"$c_{eff}$" + f" =  {abs(slope_bottom)*1000:.0f} m/s", fontsize=12, color='b',
                bbox=dict(facecolor='white', alpha=0.95, edgecolor='none', pad=2))
        # Build a corridor around the blue line: y = m * (x + 1)
        y0, y1 = axs[gg].get_ylim()
        y_grid = np.linspace(min(y0, y1), max(y0, y1), 400)

        if np.isclose(m, 0.0):
            x_center = np.full_like(y_grid, 0)  # fallback for near-zero slope
        else:
            x_center = (y_grid / m)

        cushion = 10 # seconds (x-units) on each side of the blue line
        x_before = x_center
        x_after = x_center
        if scale_num < 3:
            axs[gg].fill_betweenx(
                y_grid,
                x_before - cushion,
                x_after + cushion + 5,
                color='b',
                alpha=0.2,
                edgecolor='none',
                linewidth=0,
                zorder=0
            )
        else:
            axs[gg].axline(xy1=(0, 0), slope=m, color='b', linestyle='--', linewidth=1, alpha=0.8, zorder=0)
                

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, bottom=0.05)

    axs[0].set_ylim(dist_2_road, 68)
    axs[1].set_ylim(-68, -dist_2_road)
    # Format bottom axis tick labels to show positive numbers (absolute values)
    def abs_formatter(val, pos):
        # show integers without decimal when possible
        if abs(val - round(val)) < 1e-6:
            return f"{int(abs(round(val)))}"
        return f"{abs(val):.1f}"

    axs[1].yaxis.set_major_formatter(FuncFormatter(abs_formatter))

    axs[0].spines['bottom'].set_visible(False)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axs[1].spines['top'].set_visible(True)
    axs[1].spines['top'].set_linewidth(0.1)

    if blast_num == 1:
        plt.xlabel('Time after 2019-02-16 01:52:21 UTC, s', fontsize='x-large')
    elif blast_num == 2:
        plt.xlabel('Time after 2019-02-23 02:11:12 UTC, s', fontsize='x-large')
    elif blast_num == 3:
        plt.xlabel('Time after 2019-03-02T01:42:20 UTC, s', fontsize='x-large')

    axs[0].set_ylabel('Northern distance from source, km', fontsize='x-large')
    axs[1].set_ylabel('Southern distance from source, km', fontsize='x-large')
    plt.xlim(0, 250)
    plt.savefig(file_save, dpi = 500)