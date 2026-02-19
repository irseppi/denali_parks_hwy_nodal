import pytz
import matplotlib.pyplot as plt
from datetime import datetime
from obspy.core import UTCDateTime

day_data = []
night_data = []
num = 0

local_tz = pytz.timezone('US/Alaska')

with open("nodal_ZE.arrival", "r") as text:
    for line in text:
        val = line.split()
        timestamp = UTCDateTime(float(val[1]))

        # Convert to local time
        utc_dt = datetime.utcfromtimestamp(timestamp.timestamp)
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)

        hour = local_dt.hour

        # Define time windows
        # Night: 19:00–7:00 (overnight)
        # Day: 7:00–19:00 

        if 7 <= hour < 19:
            day_data.append(local_dt)
        else:
            night_data.append(local_dt)

        num += 1

print("# of Day arrivals:", len(day_data))
print("# of Night arrivals:", len(night_data))
print("Total # of picks:", num, len(day_data) + len(night_data))    

# Plot
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax[0].hist(day_data,
           bins=43,
           color='lightgray',
           edgecolor='black',
           linewidth=0.8,
           rwidth=0.85)

ax[0].set_title(f'Day Time [{len(day_data)} total]')
ax[0].set_ylabel('Count')

ax[1].hist(night_data,
           bins=43,
           color='lightgray',
           edgecolor='black',
           linewidth=0.8,
           rwidth=0.85)

ax[1].set_title(f'Night Time [{len(night_data)} total]')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('day_vs_night_count.pdf')
plt.show()