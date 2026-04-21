import obspy
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from datetime import timezone

folder = '/scratch/irseppi/500sps/2019_02_22/'

files = ['car_before', 'car_after', 'train_before', 'train_after', 'air_before','air_after', 'eq_before', 'eq_after']

air_dic = {"hour": [19,20], "min": [48,20]}
eq_dic = {"hour": [20,20], "min": [36,39]}
train_dic = {"hour": [20,21], "min": [49,20]}
car_dic = {"hour": [20,21], "min": [8,26]}


for file_name in files:

    file_name = file_name + '.txt'
    if Path(file_name).exists():
        continue

    print(f"Plotting {file_name}...")
    # format x axis a
    r1 = open(file_name, 'w')
    coords = []
    stations = np.arange(1123, 1246)
    if file_name.startswith('air'):
        time_dic = air_dic
    elif file_name.startswith('eq'):
        time_dic = eq_dic
    elif file_name.startswith('train'):
        time_dic = train_dic
    elif file_name.startswith('car'):
        time_dic = car_dic

    hour_start = time_dic["hour"][0]
    hour_end = time_dic["hour"][1]
    min_start = time_dic["min"][0]
    min_end = time_dic["min"][1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for i,station in enumerate(stations):
        file = folder + f'ZE_{station}_DPZ.msd'
        if Path(file).exists():
            tr = obspy.read(file)
            tr[0].trim(tr[0].stats.starttime + (hour_start * 60 * 60) + (min_start * 60), tr[0].stats.starttime + (hour_end * 60 * 60) + (min_end * 60))

            data = tr[0][:]
            t_wf = tr[0].times()
            norm_data = (data / np.max(np.abs(data)))
            start_time = tr[0].stats.starttime
            ax.plot(t_wf, norm_data + int(station), 'k', linewidth=0.5)
    if file_name.endswith('after.txt'):
        #read in file name before
        before_file_name = file_name.replace('after.txt', 'before.txt')
        with open(before_file_name, 'r') as f:
            for line in f:
                x, y, t, _ = line.strip().split(',')
                ax.scatter(float(x), float(y), color='blue', marker='o')
    if file_name.startswith('train'):
        print("Setting y limits for train")
        ax.set_ylim(1212, 1247)
    else:
        ax.set_ylim(1122, 1247)
    ax.set_xlim(0, max(t_wf))
    def onclick(event, coords=coords):
        #global coords
        coords.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, color='red', marker='x')  
        plt.draw() 
        print('Clicked:', event.xdata, event.ydata)  
        r1.write(str(event.xdata) + ',' + str(event.ydata) + ',' + str(start_time) + ',\n')
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.show(block=True)
    plt.close(fig)
    r1.close()

