"""This code fetches PSD data for the nodal 
stations form MUSTANG.
"""

import pandas as pd
from urllib.request import urlopen

input_dir = './denali_parks_hwy_nodal_supp/EXAMPLES/soundscape/input/'

seismo_data = pd.read_csv(f'{input_dir}nodes_stations.txt',
                           sep="|")
stations = seismo_data['Station']
channels = ['1', '2', 'Z']

for channel in channels:
    for station in stations:
        station = str(station)
        xml_file = f"psd_stations/psd_{station}_DP{channel}.xml"
        xml = open(xml_file, "w")
        try:
            xml_url = f'https://service.iris.edu/mustang/noise-psd/1/query?net='
            xml_url += f'ZE&sta={station}&loc=--&cha=DP{channel}&quality='
            xml_url += f'D&starttime=2019-02-11T00:00:00&endtime=2019-03-26&'
            xml_url += f'T00:00:00correct=true&format=xml&nodata=404'

            xml.write(urlopen(xml_url).read().decode('utf-8'))

        except Exception as e:
            print(f"Error fetching {station}: {e}")
        finally:
            xml.close()
