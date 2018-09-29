import matplotlib.pyplot as plt
import numpy as np

radial_velocity = np.array(range(1001))
frequency = np.array([150e6, 450e6, 3e9, 10e9, 35e9])
wavelength = 3e8 / frequency
for i in range(len(wavelength)):
    doppler = (2 * radial_velocity) / wavelength[i]
    plt.loglog(radial_velocity, doppler)

plt.grid('on')
plt.xlabel('Radial Velocity')
plt.ylabel('Doppler Frequency')
plt.show()