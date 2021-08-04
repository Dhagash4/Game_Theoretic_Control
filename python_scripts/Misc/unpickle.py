import numpy as np
import _pickle as cpickle
import pandas as pd

filename = '../../Data/MPC/mpc_1_car_states.pickle'
data = []

with open(filename, 'rb') as f:
    while True:
        try:
            data.append(cpickle.load(f))
        except EOFError:
            break

print(np.array(data))
