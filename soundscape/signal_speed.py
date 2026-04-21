import numpy as np

from pathlib import Path

files = ['car', 'train', 'air', 'eq']

for file_name in files:

    file_path = f'{file_name}_before.txt'
    station_list = []
    time_before = []

    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            for line in f:
                xb, yb, tb, _ = line.strip().split(',')
                td = float(xb)
                station_list.append(int(yb))
                time_before.append(td)


    station_arr = np.array(station_list)
    time_before_arr = np.array(time_before)

    pb = np.polyfit(station_arr, time_before_arr, 1)
    slope = np.polyfit(time_before_arr, station_arr, 1)[0] * 1000
    print("Slope " + file_name + ": " + str(abs(np.round(slope, 2))) + " m/sec")
