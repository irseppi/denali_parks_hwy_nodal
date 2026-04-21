import pytz
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime

day_data = {}
night_data = {}
day_count = 0
night_count = 0
local_tz = pytz.timezone('US/Alaska')

with open("nodal_ZE.arrival", "r") as text:
    for line in text:
        val = line.split()
        timestamp = UTCDateTime(float(val[1]))

        # Convert Obspy UTCDateTime to a timezone-aware UTC datetime
        utc_dt = timestamp.datetime.replace(tzinfo=pytz.utc)
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        hour = local_dt.hour

        # Define time windows
        # Night: 19:00–7:00 (overnight)
        # Day: 7:00–19:00 
        local_day = local_dt.day
        local_month = local_dt.month

        #combine month and day into date to save but just save month and day not year
        local_date = f"{local_month:02d}-{local_day:02d}"

        if 7 <= hour < 19:
            if local_date not in day_data:
                day_data[local_date] = []
            day_data[local_date].extend([local_dt])
        else:
            if local_date not in night_data:
                night_data[local_date] = []
            night_data[local_date].extend([local_dt])

print("Total # of picks:", sum(len(v) for v in day_data.values()) + sum(len(v) for v in night_data.values()))
#Print out how many times more picks per hour in the daytime than during the nighttime
total_day_hours = len(day_data) * 12
total_night_hours = len(night_data) * 12
day_picks_per_hour = sum(len(v) for v in day_data.values()) / total_day_hours
night_picks_per_hour = sum(len(v) for v in night_data.values()) / total_night_hours


print(f"Day picks per hour is {day_picks_per_hour / night_picks_per_hour:.2f} times higher than night picks per hour")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].bar(day_data.keys(),
           height=[len(v) for v in day_data.values()],
           color='lightgray',
           edgecolor='black',
           linewidth=0.8,
           width=0.85)

ax[0].set_title(f'Day Time [{sum(len(v) for v in day_data.values())} total]')
ax[0].set_ylabel('Count')

ax[1].bar(night_data.keys(),
           height=[len(v) for v in night_data.values()],
           color='lightgray',
           edgecolor='black',
           linewidth=0.8,
           width=0.85)

# only put a label on every fourth tick mark but always show the last one
labels = ax[1].xaxis.get_majorticklabels()

for i, label in enumerate(labels):
    if i % 4 != 0:
        label.set_visible(False)

labels[-1].set_visible(True)
ax[1].set_title(f'Night Time [{sum(len(v) for v in night_data.values())} total]')
ax[1].set_xlabel('Date, 2019')
ax[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('day_vs_night_count.pdf')
plt.show()