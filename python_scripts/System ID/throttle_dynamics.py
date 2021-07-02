import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('..')

from Common.util import *
from Common.custom_dataclass import *

filename = "states_throttle_0.100000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_1 = np.array([d.time for d in data])
vx_1 = np.array([d.v_lon for d in data])
vy_1 = np.array([d.v_lat for d in data])
dt1 = np.average(np.diff(t_1))

filename = "states_throttle_0.200000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_2 = np.array([d.time for d in data])
vx_2 = np.array([d.v_lon for d in data])
vy_2 = np.array([d.v_lat for d in data])
dt2 = np.average(np.diff(t_2))

filename = "states_throttle_0.300000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_3 = np.array([d.time for d in data])
vx_3 = np.array([d.v_lon for d in data])
vy_3 = np.array([d.v_lat for d in data])
dt3 = np.average(np.diff(t_3))

filename = "states_throttle_0.400000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_4 = np.array([d.time for d in data])
vx_4 = np.array([d.v_lon for d in data])
vy_4 = np.array([d.v_lat for d in data])
dt4 = np.average(np.diff(t_4))

filename = "states_throttle_0.500000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_5 = np.array([d.time for d in data])
vx_5 = np.array([d.v_lon for d in data])
vy_5 = np.array([d.v_lat for d in data])
dt5 = np.average(np.diff(t_5))

filename = "states_throttle_0.600000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_6 = np.array([d.time for d in data])
vx_6 = np.array([d.v_lon for d in data])
vy_6 = np.array([d.v_lat for d in data])
dt6 = np.average(np.diff(t_6))

filename = "states_throttle_0.700000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_7 = np.array([d.time for d in data])
vx_7 = np.array([d.v_lon for d in data])
vy_7 = np.array([d.v_lat for d in data])
dt7 = np.average(np.diff(t_7))

filename = "states_throttle_0.800000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_8 = np.array([d.time for d in data])
vx_8 = np.array([d.v_lon for d in data])
vy_8 = np.array([d.v_lat for d in data])
dt8 = np.average(np.diff(t_8))

filename = "states_throttle_0.900000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_9 = np.array([d.time for d in data])
vx_9 = np.array([d.v_lon for d in data])
vy_9 = np.array([d.v_lat for d in data])
dt9 = np.average(np.diff(t_9))

filename = "states_throttle_1.000000.pickle"
with open(filename, 'rb') as f:
    data = pickle.load(f)
t_10 = np.array([d.time for d in data])
vx_10 = np.array([d.v_lon for d in data])
vy_10 = np.array([d.v_lat for d in data])
dt10 = np.average(np.diff(t_10))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

ax1.plot(t_1, vx_1)
ax1.set_xlabel('Time')
ax1.set_ylabel('Velocity')
ax1.set_title('Throttle = 0.1')

ax2.plot(t_2, vx_2)
ax2.set_xlabel('Time')
ax2.set_ylabel('Velocity')
ax2.set_title('Throttle = 0.2')

ax3.plot(t_3, vx_3)
ax3.set_xlabel('Time')
ax3.set_ylabel('Velocity')
ax3.set_title('Throttle = 0.3')

ax4.plot(t_4, vx_4)
ax4.set_xlabel('Time')
ax4.set_ylabel('Velocity')
ax4.set_title('Throttle = 0.4')

ax5.plot(t_5, vx_5)
ax5.set_xlabel('Time')
ax5.set_ylabel('Velocity')
ax5.set_title('Throttle = 0.5')

ax6.plot(t_6, vx_6)
ax6.set_xlabel('Time')
ax6.set_ylabel('Velocity')
ax6.set_title('Throttle = 0.6')

ax7.plot(t_7, vx_7)
ax7.set_xlabel('Time')
ax7.set_ylabel('Velocity')
ax7.set_title('Throttle = 0.7')

ax8.plot(t_8, vx_8)
ax8.set_xlabel('Time')
ax8.set_ylabel('Velocity')
ax8.set_title('Throttle = 0.8')

ax9.plot(t_9, vx_9)
ax9.set_xlabel('Time')
ax9.set_ylabel('Velocity')
ax9.set_title('Throttle = 0.9')

plt.show()

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

ax1.plot(vx_1[1:], np.diff(vx_1) / dt1)
ax1.set_xlabel('Velocity')
ax1.set_ylabel('Acceleration')
ax1.set_title('Throttle = 0.1')

ax2.plot(vx_2[1:], np.diff(vx_2) / dt2)
ax2.set_xlabel('Velocity')
ax2.set_ylabel('Acceleration')
ax2.set_title('Throttle = 0.2')

ax3.plot(vx_3[1:], np.diff(vx_3) / dt3)
ax3.set_xlabel('Velocity')
ax3.set_ylabel('Acceleration')
ax3.set_title('Throttle = 0.3')

ax4.plot(vx_4[1:], np.diff(vx_4) / dt4)
ax4.set_xlabel('Velocity')
ax4.set_ylabel('Acceleration')
ax4.set_title('Throttle = 0.4')

ax5.plot(vx_5[1:], np.diff(vx_5) / dt5)
ax5.set_xlabel('Velocity')
ax5.set_ylabel('Acceleration')
ax5.set_title('Throttle = 0.5')

ax6.plot(vx_6[1:], np.diff(vx_6) / dt6)
ax6.set_xlabel('Velocity')
ax6.set_ylabel('Acceleration')
ax6.set_title('Throttle = 0.6')

ax7.plot(vx_7[1:], np.diff(vx_7) / dt7)
ax7.set_xlabel('Velocity')
ax7.set_ylabel('Acceleration')
ax7.set_title('Throttle = 0.7')

ax8.plot(vx_8[1:], np.diff(vx_8) / dt8)
ax8.set_xlabel('Velocity')
ax8.set_ylabel('Acceleration')
ax8.set_title('Throttle = 0.8')

ax9.plot(vx_9[1:], np.diff(vx_9) / dt9)
ax9.set_xlabel('Velocity')
ax9.set_ylabel('Acceleration')
ax9.set_title('Throttle = 0.9')

plt.show()