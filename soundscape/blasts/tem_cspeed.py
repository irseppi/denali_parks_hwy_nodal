# Comparison stations
targets = ["PAIN", "PANN"]

temp = [-2,-2]
wind_speed = [19, 9]
wind_dir = ['N', 'WNW']

for i, Tc in enumerate(temp):
    c = 331.3+0.6*Tc
    if targets[i] == 'PAIN': 
        ceff = c +  (wind_speed[i]*0.277778)
        print(f"Wind speed PAIN {wind_speed[i]*0.277778} m/s")
    elif targets[i] == 'PANN': 
        ceff = c -  (wind_speed[i]*0.277778)
        print(f"Wind speed PANN {wind_speed[i]*0.277778} m/s")
    print(f"Speed of sound at {Tc}°C (Station {targets[i]}): {c} m/s")
    print(f"Effective speed of sound at {Tc}°C (Station {targets[i]}): {ceff} m/s")
