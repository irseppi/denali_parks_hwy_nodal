import obspy
import numpy as np
import matplotlib.dates as mdates

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from datetime import timezone


folder = '/scratch/irseppi/500sps/2019_02_22/'

stations = np.arange(1123, 1246)
fig, ax = plt.subplots(2, 1, figsize=(12, 8),sharex=True,gridspec_kw={'height_ratios': [1, 5]})
for i,station in enumerate(stations):
    chan = "Z" #actual channel here so component Z is Z
    file = folder + f'ZE_{station}_DP{chan}.msd'
    if Path(file).exists():
        tr = obspy.read(file)
        tr[0].trim(tr[0].stats.starttime + (19 * 60 * 60) +(40 * 60)  ,tr[0].stats.starttime + (19 * 60 * 60) + (140 * 60))
        
        data = tr[0][:]
        t_wf = tr[0].times()
        norm_data = (data / np.max(np.abs(data)))
        if i == 0:
            # convert spectrogram times (seconds since trace start) to timezone-aware datetimes and then to matplotlib date numbers
            start_utc = tr[0].stats.starttime
            times_abs = [(start_utc + t).datetime.replace(tzinfo=timezone.utc) for t in t_wf]
            time_nums = mdates.date2num(times_abs)

        ax[1].plot(time_nums, norm_data + int(station), 'k', linewidth=0.5)

frequencies, times, Sxx = spectrogram(data, 500, scaling='density', nperseg=500, noverlap=int(500 * .6), detrend='constant')

a, b = Sxx.shape
MDF = np.zeros((a,b))
for row in range(len(Sxx)):
    median = np.median(Sxx[row])
    MDF[row, :] = median

# Avoid log10(0) by replacing zeros with a small positive value
Sxx_safe = np.where(Sxx == 0, 1e-10, Sxx)
MDF_safe = np.where(MDF == 0, 1e-10, MDF)

spec = 10 * np.log10(Sxx_safe) - (10 * np.log10(MDF_safe))
max_val = np.max(spec)*0.8
min_val = np.min(spec)*0.1

# convert spectrogram times (seconds since trace start) to timezone-aware datetimes and then to matplotlib date numbers
times_abs = [(start_utc + t).datetime.replace(tzinfo=timezone.utc) for t in times]
time_nums = mdates.date2num(times_abs)

# plot using matplotlib date numbers on the x axis
pcm = ax[0].pcolormesh(time_nums, frequencies, spec, cmap='pink_r', vmin=min_val, vmax=max_val)
start, end = [0,1700,4170], [900,1780,5270]
c_limits = ['r','purple','magenta']
for c_lim in zip(start, end, c_limits):
    s, e, c = c_lim
    s = mdates.date2num((start_utc + s).datetime.replace(tzinfo=timezone.utc))
    e = mdates.date2num((start_utc + e).datetime.replace(tzinfo=timezone.utc))
    ax[0].axvspan(s, e, color=c, alpha=0.2, edgecolor='none', linewidth=0)

files = ['car', 'train', 'air','eq']
plot_individual = False

for file_name in files:

    file_path = f'{file_name}_before.txt'
    station_list = []
    time_before = []
    time_after = []
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            for line in f:
                xb, yb, tb, _ = line.strip().split(',')
                # parse ISO time string tb and add xb seconds, then make timezone-aware
                td = obspy.UTCDateTime(tb) + float(xb)
                times_abs = td.datetime.replace(tzinfo=timezone.utc)
                time_cov = mdates.date2num(times_abs)
                # store station as numeric value so it can be plotted on the y axis
                station_list.append(int(yb))
                time_before.append(time_cov)

    file_path = f'{file_name}_after.txt'
    if file_name != 'eq':
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                for line in f:
                    xa, ya, ta, _ = line.strip().split(',')
                    # parse ISO time string ta and add xa seconds, then make timezone-aware
                    td = obspy.UTCDateTime(ta) + float(xa)
                    times_abs = td.datetime.replace(tzinfo=timezone.utc)
                    time_cov = mdates.date2num(times_abs)
                    time_after.append(time_cov)

    if file_name == 'air':
        c = 'r'
        cushion = 0.0012
    elif file_name == 'eq':
        c = 'blue'
        cushion = 0
    elif file_name == 'train':
        c = 'magenta'
        cushion = 0.002
    elif file_name == 'car':
        c = 'purple'
        cushion = 0.001
    # convert to numpy arrays for consistent plotting; ensure lists are non-empty
    if station_list:
        station_arr = np.array(station_list)
        time_before_arr = np.array(time_before)
        time_after_arr = np.array(time_after)

        y_grid = np.linspace(1122, 1247, 200)
        x_after = None
        pb = np.polyfit(station_arr, time_before_arr, 1)
        x_before = np.polyval(pb, y_grid)
        
        ax[1].plot(time_before_arr, station_arr, color=c, linestyle='--', linewidth=0.7, alpha=0.7)

        if file_name != 'eq' and file_name != 'train':  # only plot after lines for non-earthquake and non-aircraft signals
            pa = np.polyfit(station_arr, time_after_arr,1)
            x_after = np.polyval(pa, y_grid)

            ax[1].plot(time_after_arr, station_arr, color=c, linestyle='--', linewidth=0.7, alpha=0.7)
            ax[1].fill_betweenx(y_grid, x_before - cushion, x_after + cushion,
                                color=c, alpha=0.2, edgecolor='none', linewidth=0)
            
        elif file_name == 'train':  # for train, plot a wider shaded region to indicate uncertainty
            pa = np.polyfit(station_arr, time_after_arr, 1)
            x_after = np.polyval(pa, y_grid)

            #find index of station 1245
            gg = np.where(station_arr == 1245)[0]

            idx = int(gg[0])
            x_dist_A = time_after_arr[idx] - time_before_arr[idx]

            ax[1].plot(time_after_arr, station_arr, color=c, linestyle='--', linewidth=0.7, alpha=0.7)
            ax[1].fill_betweenx(y_grid, x_after - (cushion/3) - x_dist_A, x_after + (cushion),
                                color=c, alpha=0.2, edgecolor='none', linewidth=0)
            
    argmax_station = np.argmax(station_arr)
    ax[0].axvline(time_before_arr[argmax_station], linestyle='--', linewidth=0.7, color=c, alpha=0.7)
    if file_name != 'eq':   
        ax[0].axvline(time_after_arr[argmax_station], linestyle='--', linewidth=0.7, color=c, alpha=0.7) 

ax[1].set_xlim(time_nums.min(), time_nums.max())
# format x axis as HH:MM:SS in UTC
ax[1].xaxis_date()
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
ax[1].set_ylim(1122, 1246)
ax[0].set_ylabel("Frequency, Hz", fontsize='large')
ax[0].tick_params(axis='y', labelsize='large')
ax[1].set_xlabel("Time (UTC) — 2019-02-22", fontsize='x-large')
print(f"Local time: {start_utc.datetime.replace(tzinfo=timezone.utc).astimezone().isoformat()}")
# show every 10th station starting at the first (use actual station names)
tick_positions = np.arange(1122, 1247, 1)
tick_labels = []
first_station = stations[0]
last_station = stations[-1]
for p in tick_positions:
    if p == first_station:
        tick_labels.append(str(first_station))
    elif p == last_station:
        tick_labels.append(str(last_station))
    elif p == 1235:
        tick_labels.append("Anderson")
    elif p == 1215:
        tick_labels.append("Healy")
    elif p == 1196:
        tick_labels.append("Park Road")
    elif p == 1199:
        tick_labels.append("Denali")
    elif p == 1153:
        tick_labels.append("Cantwell")
    else:
        tick_labels.append("")

# Remove tick marks that don't have an associated label (empty string)
filtered = [(p, l) for p, l in zip(tick_positions, tick_labels) if l != ""]
if filtered:
    tick_positions, tick_labels = zip(*filtered)
    tick_positions = np.array(tick_positions)
    tick_labels = list(tick_labels)
else:
    tick_positions = np.array([])
    tick_labels = []

ax[1].set_yticks(tick_positions, tick_labels, fontsize='large')
ax[1].tick_params(axis='x', labelsize='large')

#set the width between plots smaller
plt.subplots_adjust(hspace=0.05)
plt.tight_layout(pad=0, h_pad=0)
plt.savefig('signal_gallery_record_section.png', dpi=400)

if plot_individual:
    start, end = [0,1700,4170], [900,1780,5270]
    labels = ['Aircraft', 'Car', 'Train']
    steps = [200,10,200]
    for s, e, l, step in zip(start, end, labels, steps):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # plot spectrogram for 0-800 s and scale colorbar to max in that range
        mask = (times >= s) & (times <= e)
        vmin_range = np.min(spec[:, mask]) if np.any(mask) else np.min(spec)
        vmax_range = np.max(spec[:, mask]) if np.any(mask) else np.max(spec)
        pcm = ax.pcolormesh(times, frequencies, spec, cmap='pink_r', vmin=min_val, vmax=max_val) #*0.8)
        plt.xlim(s, e)
        ticks = np.arange(s, e + step, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t - s)) for t in ticks])

        ax.tick_params(axis='both', labelsize='large')

        plt.xlabel("Time, s", fontsize='large')
        plt.ylabel("Frequency, Hz", fontsize='large')
        plt.tight_layout()
        plt.savefig(f'spectrogram_{l.lower()}.png', dpi=300)

