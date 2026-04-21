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
    if not trace:
        return None

    trc = trace[0].copy()

    start = trc.stats.starttime + t0_offset_sec
    end = trc.stats.starttime + t1_offset_sec

    trc.trim(start, end)

    data = trc[:]
    if data.size == 0:
        return None

    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)

    win = max(5, (len(envelope) // 200) | 1)  # force odd
    envelope = savgol_filter(envelope, win, 3)

    t_wf = trc.times("matplotlib")
    trace_start = trc.stats.starttime

    return t_wf, envelope, trace_start


def compute_line(station):
    file = folder + f'ZE_{station}_DPZ.msd'
    print(f"Processing station: {station}")

    if not Path(file).exists():
        return None

    st = obspy.read(file)

    # window 1: 11:30 → 15:00
    t0_off_1 = 11 * 3600 + 30 * 60
    t1_off_2 = 19 * 3600

    res = process_window(st, t0_off_1, t1_off_2)
    if res is None:
        return None

    tn, env, trace_start = res

    # normalize (fix: both should use same reference)
    cutoff_num = mdates.date2num((trace_start + 3 * 3600).datetime)
    first_3h_mask = tn <= cutoff_num

    if np.any(first_3h_mask):
        norm = np.max(np.abs(env[first_3h_mask]))
    else:
        norm = np.max(np.abs(env))
    if norm == 0:
        return None

    env = (env / norm)

    return station, tn, env

# -------------------------
# CONFIG
# -------------------------
folder = '/scratch/irseppi/50sps/2019_02_16/'
stations = np.arange(1119, 1238)
file_names = ['train1', 'train2']

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(compute_line, stations))

    # -------------------------
    # plot waveforms
    # -------------------------
    for result in results:
        if result is None:
            continue

        station, tn, env = result
        offset = int(station)

        # window 1
        ax.plot(tn, env + offset, color='k', linewidth= 0.6, alpha=0.7)
        if station == 1119:
            ax.plot(tn, env + offset, color='k', linewidth=1)
    # -------------------------
    # train annotations
    # -------------------------
    for file_name in file_names:

        station_list = []
        time_before = []
        time_after = []
        time_before_ex = []
        time_after_ex = []

        file_path = f'{file_name}_before.txt'
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                for line in f:
                    xb, yb, tb, _ = line.strip().split(',')
                    station_list.append(int(yb))
                    time_before.append(float(xb) - 0.001)
                    time_before_ex.append(float(xb) - 0.005)

        file_path = f'{file_name}_after.txt'
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                for line in f:
                    xa, ya, ta, _ = line.strip().split(',')
                    time_after.append(float(xa) + 0.001)
                    time_after_ex.append(float(xa) + 0.005)

        if file_name == 'train1':
            ax.plot(time_before, station_list, color='C0', linewidth=0.4)
            ax.plot(time_after, station_list, color='C0', linewidth=0.4)
            ax.fill_betweenx(station_list, time_before_ex, time_after_ex,
                             color='C0', alpha=0.3)

        elif file_name == 'train2':
            ax.plot(time_before, station_list, color='C1', linewidth=0.4)
            ax.plot(time_after, station_list, color='C1', linewidth=0.4)
            ax.fill_betweenx(station_list, time_before_ex, time_after_ex,
                             color='C1', alpha=0.3)

    # -------------------------
    # time formatting
    # -------------------------
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))

    fig.text(0.5, 0.04,
             "Time (UTC) — 2019-02-16",
             ha='center', va='center', fontsize='x-large')

    print(f"Total stations plotted: {sum(result is not None for result in results)}")
    # -------------------------
    # station labels
    # -------------------------
    tick_positions = np.arange(1119, 1238)

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

    filtered = [(p, l) for p, l in zip(tick_positions, tick_labels) if l != ""]

    if filtered:
        tick_positions, tick_labels = zip(*filtered)
        tick_positions = np.array(tick_positions)
        tick_labels = list(tick_labels)

    ax.set_yticks(tick_positions, tick_labels, fontsize='large')
    ax.tick_params(axis='x', labelsize='large')

    # -------------------------
    # limits
    # -------------------------
    ax.set_ylim(stations[0]-1, stations[-1]+1)
    ax.set_xlim(mdates.date2num(obspy.UTCDateTime("2019-02-16T11:30:00").datetime),
                mdates.date2num(obspy.UTCDateTime("2019-02-16T19:00:00").datetime))
    plt.subplots_adjust(top=0.98,
                        bottom=0.08,
                        left=0.1,
                        right=0.975)

    plt.grid(False)
    plt.show()