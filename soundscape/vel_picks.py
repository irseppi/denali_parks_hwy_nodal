from pathlib import Path

files = ['car', 'train', 'air','eq']

for file_name in files:

    file_path = f'{file_name}_before.txt'
    station_list = []
    time_before = []
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            for line in f:
                xb, yb, tb, _ = line.strip().split(',')
                station_list.append(int(yb))
                time_before.append(float(xb))
    slope = (station_list[-1] - station_list[0]) / (time_before[-1] - time_before[0])
    print(f"{file_name} velocity before: {slope*1000:.3f} m/s")
    mph = 2.23694 * slope*1000
    print(f"{file_name} velocity before: {mph:.1f} mph")
    
    if file_name != 'eq':

        file_path = f'{file_name}_after.txt'
        station_list = []
        time_before = []
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                for line in f:
                    xb, yb, tb, _ = line.strip().split(',')
                    station_list.append(int(yb))
                    time_before.append(float(xb))
        slope = (station_list[-1] - station_list[0]) / (time_before[-1] - time_before[0])
        print(f"{file_name} velocity after: {slope*1000:.3f} m/s")
        mph = 2.23694 * slope*1000
        print(f"{file_name} velocity after: {mph:.1f} mph")
