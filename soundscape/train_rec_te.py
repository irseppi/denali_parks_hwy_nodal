import os
import obspy
import concurrent.futures

import numpy as np
import matplotlib.dates as mdates

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import timezone
from scipy.signal import hilbert, savgol_filter

num_workers = os.cpu_count()

def process_window(trace, t0_offset_sec, t1_offset_sec):
    # copy and trim using offsets in seconds from original trace start
    if not trace:
        return None
    trc = trace[0].copy()

    start = trc.stats.starttime + t0_offset_sec
    end = trc.stats.starttime + t1_offset_sec

    trc.trim(start, end)
    trc.filter("highpass", freq=20)
    data = trc[:]
    t_wf = trc.times()
    start_time = trc.stats.starttime
    if data.size == 0:
        return None
    
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    win = max(5, (len(envelope) // 200) | 1)  # force odd
    envelope = savgol_filter(envelope, win, 3)
    t_wf = trc.times("matplotlib")
    return t_wf, envelope, start_time #data

def compute_line(station, file_out):
    file = folder + f'ZE_{station}_DPZ.msd'
    print(f"Processing station: {station}")
    if not Path(file).exists():
        return
    st = obspy.read(file)
    if file_out.startswith('train1'):
        # first window: 11:30 -> 14:00  (seconds offset from trace start)
        start = 11 * 3600 + 55 * 60
        end = 14 * 3600 + 15 *60 + 1000
    elif file_out.startswith('train2'):
    # second window: 16:20 -> 19:00
        start = 16 * 3600 + 5 * 60 - 1000
        end = 18 * 3600 + 45 * 60
    res = process_window(st, start, end)
    if res is None:
        return

    tn, env, start_time = res

    normenv = env / np.max(np.abs(env))

    return station, tn, normenv, start_time

folder = '/scratch/irseppi/50sps/2019_02_16/'
stations = np.arange(1119, 1238)
file_names = ['train1_before','train1_after', 'train2_before', 'train2_after']

if __name__ == "__main__":
    for file_name in file_names:
        file_name = file_name + '.txt'
        if Path(file_name).exists():
            print(f"Appending to {file_name}")
            r1 = open(file_name, 'a')
        else:
            r1 = open(file_name, 'w')
        coords = []

        fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharey=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(compute_line, stations, [file_name]*len(stations)))

        for result in results:
            if result is None:
                continue

            station, tn, env, start_time = result

            ax.plot(tn, env + int(station), color='C0', linewidth=0.6)

        if file_name.endswith('after.txt'):
            #read in file name before
            before_file_name = file_name.replace('after.txt', 'before.txt')
            with open(before_file_name, 'r') as f:
                for line in f:
                    x, y, t, _ = line.strip().split(',')
                    ax.scatter(float(x), float(y), color='blue', marker='o')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
        ax.set_xlabel("Time (UTC) — 2019-02-16, HH:MM", fontsize='x-large')

        ax.set_xlim(tn.min(), tn.max())

        ax.set_ylim(stations[0]-1, stations[-1]+1)
        ax.set_ylabel("Amplitude Envelope, counts", fontsize='x-large')
        def onclick(event, coords=coords):
            #global coords
            coords.append((event.xdata, event.ydata))
            plt.scatter(event.xdata, event.ydata, color='red', marker='x')  
            plt.draw() 
            print('Clicked:', event.xdata, event.ydata)  
            r1.write(str(event.xdata) + ',' + str(event.ydata) + ',' + str(start_time) + ',\n')
        plt.gcf().canvas.mpl_connect('button_press_event', onclick)

        plt.grid(False)
        plt.show()
    