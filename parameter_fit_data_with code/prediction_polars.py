import keras.models
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import tensorflow
import numpy as np
from math import e, pi


raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/damped/damped_results_all_dir.csv", sep=',')
load_model = keras.models.load_model('damped_spring_all_dir1')

baseline_dimensions = np.asarray(raw_data.sample(n=1))[0][0:3]

directions = [0, -45, -90, -135, -180]
r_axis = np.linspace(0.1, 2.5, 60)
radii = []
dirs = []
pred_x = []
pred_y = []
pred_z = []
pred_rx = []
pred_ry = []
pred_rz = []


def func(x, a, b, c):
    y = c * e**-(a*x) + b*x*e**-(a*x)
    return y


order = 3
print(baseline_dimensions)
print('\n\n')
for theta in directions:
    radii.extend(r_axis)
    baseline_input = [np.append(baseline_dimensions, theta).tolist()]
    print(baseline_input[0])
    new_pred = load_model.predict(baseline_input)[0]
    for i in r_axis:
        dirs.append(abs(theta))
        pred_x.append(func(i, *new_pred[0*order:0*order+order]))
        pred_y.append(func(i, *new_pred[1*order:1*order+order]))
        pred_z.append(func(i, *new_pred[2*order:2*order+order]))
        pred_rx.append(func(i, *new_pred[3*order:3*order+order]))
        pred_ry.append(func(i, *new_pred[4*order:4*order+order]))
        pred_rz.append(func(i, *new_pred[5*order:5*order+order]))


# grid_r, grid_theta = np.meshgrid(radii, dirs)
# data = griddata(np.asarray(radii).reshape(300,1), np.asarray(pred_x), (grid_r, grid_theta), method='cubic', fill_value=0)
ax = plt.subplot(projection='polar')
ax.pcolormesh(dirs, radii, pred_x[:-1, :-1], cmap='coolwarm')
#
print(dirs)
# ax.set_theta_zero_location('N')
# ax.set_thetamin(0)
# ax.set_thetamax(180)
plt.show()