import pandas as pd
import tensorflow as tf
import numpy as np
from math import e
from statistics import mean
import matplotlib.pyplot as plt

waterplanes = []
rpd_errs = []
raw_errs = []
colors = []

deg0_rpd = []
deg45_rpd = []
deg90_rpd = []
deg135_rpd = []
deg180_rpd = []
deg0_raw = []
deg45_raw = []
deg90_raw = []
deg135_raw = []
deg180_raw = []

avg_x_err_rpd = []
avg_y_err_rpd = []
avg_z_err_rpd = []
avg_rx_err_rpd = []
avg_ry_err_rpd = []
avg_rz_err_rpd = []
avg_x_err_raw = []
avg_y_err_raw = []
avg_z_err_raw = []
avg_rx_err_raw = []
avg_ry_err_raw = []
avg_rz_err_raw = []


load_model = tf.keras.models.load_model("D:\IdeaProjects\PyCharm\TAMU_Work\OCEN 485/damped_spring_all_dir3")

print(load_model.summary())
raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/damped/damped_results_all_dir.csv", sep=',')
raw_data.dropna(axis=0, inplace=True)

raw_data.pop('r_squared_surge')
raw_data.pop('r_squared_sway')
raw_data.pop('r_squared_heave')
raw_data.pop('r_squared_roll')
raw_data.pop('r_squared_pitch')
raw_data.pop('r_squared_yaw')
count = 0

while count < 120:
    baseline = np.asarray(raw_data.sample(n=1))[0]
    baseline_input = baseline[0:4]
    baseline_prediction = baseline[4:]

    new_input = [baseline_input.tolist()]
    wp_area = baseline_input[0] * baseline_input[1]
    new_pred = load_model.predict(new_input)[0]
    orig_x = []
    orig_y = []
    orig_z = []
    orig_rx = []
    orig_ry = []
    orig_rz = []
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
    # print(baseline_input)
    # print('\n\n')

    x_axis = np.linspace(0.1, 2.5, 60)
    for i in x_axis:
        orig_x.append(func(i, *baseline_prediction[0*order:0*order+order]))
        orig_y.append(func(i, *baseline_prediction[1*order:1*order+order]))
        orig_z.append(func(i, *baseline_prediction[2*order:2*order+order]))
        orig_rx.append(func(i, *baseline_prediction[3*order:3*order+order]))
        orig_ry.append(func(i, *baseline_prediction[4*order:4*order+order]))
        orig_rz.append(func(i, *baseline_prediction[5*order:5*order+order]))
        pred_x.append(func(i, *new_pred[0*order:0*order+order]))
        pred_y.append(func(i, *new_pred[1*order:1*order+order]))
        pred_z.append(func(i, *new_pred[2*order:2*order+order]))
        pred_rx.append(func(i, *new_pred[3*order:3*order+order]))
        pred_ry.append(func(i, *new_pred[4*order:4*order+order]))
        pred_rz.append(func(i, *new_pred[5*order:5*order+order]))


    x_err_rpd = abs(round(mean(200*np.subtract(orig_x, pred_x)/(np.add(np.absolute(orig_x), np.absolute(pred_x)))), 3))
    y_err_rpd = abs(round(mean(200*np.subtract(orig_y, pred_y)/(np.add(np.absolute(orig_y), np.absolute(pred_y)))), 3))
    z_err_rpd = abs(round(mean(200*np.subtract(orig_z, pred_z)/(np.add(np.absolute(orig_z), np.absolute(pred_z)))), 3))
    rx_err_rpd = abs(round(mean(200*np.subtract(orig_rx, pred_rx)/(np.add(np.absolute(orig_rx), np.absolute(pred_rx)))), 3))
    ry_err_rpd = abs(round(mean(200*np.subtract(orig_ry, pred_ry)/(np.add(np.absolute(orig_ry), np.absolute(pred_ry)))), 3))
    rz_err_rpd = abs(round(mean(200*np.subtract(orig_rz, pred_rz)/(np.add(np.absolute(orig_rz), np.absolute(pred_rz)))), 3))

    avg_x_err_rpd.append(x_err_rpd)
    avg_y_err_rpd.append(y_err_rpd)
    avg_z_err_rpd.append(z_err_rpd)
    avg_rx_err_rpd.append(rx_err_rpd)
    avg_ry_err_rpd.append(ry_err_rpd)
    avg_rz_err_rpd.append(rz_err_rpd)
    avg_err_rpd = mean([x_err_rpd, y_err_rpd, z_err_rpd, rx_err_rpd, ry_err_rpd, rz_err_rpd])

    x_err_raw = round(abs(mean(np.subtract(orig_x, pred_x))), 3)
    y_err_raw = round(abs(mean(np.subtract(orig_y, pred_y))), 3)
    z_err_raw = round(abs(mean(np.subtract(orig_z, pred_z))), 3)
    rx_err_raw = round(abs(mean(np.subtract(orig_rx, pred_rx))), 3)
    ry_err_raw = round(abs(mean(np.subtract(orig_ry, pred_ry))), 3)
    rz_err_raw = round(abs(mean(np.subtract(orig_rz, pred_rz))), 3)
    avg_x_err_raw.append(x_err_raw)
    avg_y_err_raw.append(y_err_raw)
    avg_z_err_raw.append(z_err_raw)
    avg_rx_err_raw.append(rx_err_raw)
    avg_ry_err_raw.append(ry_err_raw)
    avg_rz_err_raw.append(rz_err_raw)
    avg_err_raw = mean([x_err_raw, y_err_raw, z_err_raw, rx_err_raw, ry_err_raw, rz_err_raw])

    waterplanes.append(wp_area)
    colors.append(abs(baseline_input[3]))
    rpd_errs.append(avg_err_rpd)
    raw_errs.append(avg_err_raw)
    if abs(baseline_input[3]) == 0.0:
        deg0_rpd.append((avg_err_rpd))
        deg0_raw.append((avg_err_raw))
    if abs(baseline_input[3]) == 45.0:
        deg45_rpd.append((avg_err_rpd))
        deg45_raw.append((avg_err_raw))
    if abs(baseline_input[3]) == 90.0:
        deg90_rpd.append((avg_err_rpd))
        deg90_raw.append((avg_err_raw))
    if abs(baseline_input[3]) == 135.0:
        deg135_rpd.append((avg_err_rpd))
        deg135_raw.append((avg_err_raw))
    if abs(baseline_input[3]) == 180.0:
        deg180_rpd.append((avg_err_rpd))
        deg180_raw.append((avg_err_raw))
    count += 1

# TODO: plot waterplane error variation

plt.scatter(waterplanes, rpd_errs, c=colors, cmap='rainbow')
plt.title('RPD Error Variation with Waterplane Area')
plt.xlabel('Waterplane Area ($m^2$)')
plt.ylabel('Average RPD Error')
plt.colorbar().ax.set_ylabel('Wave Heading, degrees')
# plt.show()
plt.scatter(waterplanes, raw_errs, c=colors, cmap='rainbow')
plt.title('Raw Error Variation with Waterplane Area')
plt.xlabel('Waterplane Area ($m^2$)')
plt.ylabel('Average Raw Error')
plt.colorbar().ax.set_ylabel('Wave Heading, degrees')
# plt.show()
plt.clf()

# TODO: plot wave heading error variation
rpd = [deg0_rpd, deg45_rpd, deg90_rpd, deg135_rpd, deg180_rpd]
raw = [deg0_raw, deg45_raw, deg90_raw, deg135_raw, deg180_raw]
plt.boxplot(rpd, positions=[0, 45, 90, 135, 180], widths=(30, 30, 30, 30, 30), showfliers=False)
plt.xlim(-20, 200)
plt.xlabel('Wave Heading Angle, degrees')
plt.ylabel('RPD Error')
plt.title('RPD Error Variation with Wave Heading')
# plt.show()
plt.boxplot(raw, positions=[0, 45, 90, 135, 180], widths=(30, 30, 30, 30, 30), showfliers=False)
plt.xlim(-20, 200)
plt.xlabel('Wave Heading Angle, degrees')
plt.ylabel('Raw Error')
plt.title('Raw Error Variation with Wave Heading')
# plt.show()
plt.clf()

# TODO: plot degree of freedom error variation
rpd = [avg_x_err_rpd, avg_y_err_rpd, avg_z_err_rpd, avg_rx_err_rpd, avg_ry_err_rpd, avg_rz_err_rpd]
raw = [avg_x_err_raw, avg_y_err_raw, avg_z_err_raw, avg_rx_err_raw, avg_ry_err_raw, avg_rz_err_raw]
plt.boxplot(rpd, labels=('x', 'y', 'z', 'rx', 'ry', 'rz'), showfliers=False)
plt.xlabel('Degree of Freedom')
plt.ylabel('RPD Error')
plt.title('RPD Error Variation with Degree of Freedom')
plt.show()
plt.boxplot(raw, labels=('x', 'y', 'z', 'rx', 'ry', 'rz'), showfliers=False)
plt.xlabel('Degree of Freedom')
plt.ylabel('Raw Error, m/m or deg/m')
plt.title('Raw Error Variation with Degree of Freedom')
plt.show()

#

# plt.subplot(2, 3, 1)
# title = 'Barge Dimensions ' + str(baseline_input[0]) + ' m Length, ' + str(baseline_input[1]) + ' m Beam, ' + \
#     str(abs(baseline_input[2])) + ' m Draft  -  Waves Heading of: ' + str(baseline_input[3])
# plt.suptitle(title)
# plt.plot(x_axis, pred_x, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_x, color='blue', linestyle='-.', label='True RAO')
# plt.title('Surge RAO')
# plt.text(mean(x_axis), (mean(orig_x)+mean(pred_x))/2, "Relative % diff: "+ str(x_err_rpd)+'\n'+"Average raw diff: " + str(x_err_raw))
# plt.grid()
# # plt.ylim([-0.5, 1.5])
# plt.ylabel('Response (m/m)')
#
# plt.subplot(2, 3, 2)
# plt.plot(x_axis, pred_y, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_y, color='blue', linestyle='-.', label='True RAO')
# plt.title('Sway RAO')
# plt.text(mean(x_axis), (mean(orig_y)+mean(pred_y))/2, "Relative % diff: "+ str(y_err_rpd)+'\n'+"Average raw diff: " + str(y_err_raw))
# plt.grid()
# # plt.ylim([-0.5, 1.5])
#
# plt.subplot(2, 3, 3)
# plt.plot(x_axis, pred_z, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_z, color='blue', linestyle='-.', label='True RAO')
# plt.title('Heave RAO')
# plt.text(mean(x_axis), (mean(orig_z)+mean(pred_z))/2, "Relative % diff: "+ str(z_err_rpd)+'\n'+"Average raw diff: " + str(z_err_raw))
# plt.grid()
# # plt.ylim([-0.5, 1.5])
#
# plt.subplot(2, 3, 4)
# plt.plot(x_axis, pred_rx, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_rx, color='blue', linestyle='-.', label='True RAO')
# plt.title('Roll RAO')
# plt.text(mean(x_axis), (mean(orig_rx)+mean(pred_rx))/2, "Relative % diff: "+ str(rx_err_rpd)+'\n'+"Average raw diff: " + str(rx_err_raw))
# # plt.ylim([-0.5, 1.5])
# plt.ylabel('Response (Deg/m)')
# plt.xlabel('Wave Frequency (rad/s)')
# plt.grid()
#
# plt.subplot(2, 3, 5)
# plt.plot(x_axis, pred_ry, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_ry, color='blue', linestyle='-.', label='True RAO')
# plt.title('Pitch RAO')
# plt.text(mean(x_axis), (mean(orig_ry)+mean(pred_ry))/2, "Relative % diff: "+ str(ry_err_rpd)+'\n'+"Average raw diff: " + str(ry_err_raw))
# # plt.ylim([-0.5, 50])
# plt.grid()
# plt.xlabel('Wave Frequency (rad/s)')
#
# plt.subplot(2, 3, 6)
# plt.plot(x_axis, pred_rz, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_rz, color='blue', linestyle='-.', label='True RAO')
# plt.title('Yaw RAO')
# plt.text(mean(x_axis), (mean(orig_rz)+mean(pred_rz))/2, "Relative % diff: "+ str(rz_err_rpd)+'\n'+"Average raw diff: " + str(rz_err_raw))
# # plt.ylim([-0.5, 1.5])
# plt.grid()
# plt.xlabel('Wave Frequency (rad/s)')
#
# plt.legend()
# # plt.get_current_fig_manager().full_screen_toggle()
# plt.show()