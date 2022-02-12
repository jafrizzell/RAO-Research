import pandas as pd
import tensorflow as tf
import numpy as np
from math import e
from statistics import mean
import matplotlib.pyplot as plt

load_model = tf.keras.models.load_model("C:/Users/jafri/Documents/GitHub/RAO-Research/parameter_fit_data_with code/damped_spring_1dof/")

print(load_model.summary())
raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/damped/damped_results_all_dir.csv", sep=',')
#print(raw_data.isna().sum())
raw_data.dropna(axis=0, inplace=True)
column1 = raw_data['Length (m)']
column2 = raw_data['Beam (m)']
column3 = raw_data['Draft (m)']
column4 = raw_data['Heading']

raw_data.pop('r_squared_surge')
raw_data.pop('r_squared_sway')
raw_data.pop('r_squared_heave')
raw_data.pop('r_squared_roll')
raw_data.pop('r_squared_pitch')
raw_data.pop('r_squared_yaw')


train_dataset = raw_data.sample(frac=0.8, random_state=0)
test_dataset = raw_data.drop(train_dataset.index)


train_features = train_dataset.copy()
test_features = test_dataset.copy()

order = 2
train_labels = np.asarray(train_features.drop(train_features.columns[list(range(4,6*(order+1)+4))], axis=1, inplace=False))
test_labels = np.asarray(test_features.drop(test_features.columns[list(range(4,6*(order+1)+4))], axis=1, inplace=False))

train_features = train_features.drop(train_features.columns[list(range(0,4))], axis=1, inplace=False)
test_features = test_features.drop(test_features.columns[list(range(0,4))], axis=1, inplace=False)

# baseline = np.asarray(raw_data.sample(n=1))[0]
baseline = np.asarray(raw_data.loc[751])
baseline_input = baseline[0:4]
baseline_prediction = baseline[4:]

new_input = [baseline_input.tolist()]
# new_input = [[35, 18, 16, 180]]
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


order = order+1
print(baseline_input)
print('\n\n')

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

x_err_raw = round(abs(mean(np.subtract(orig_x, pred_x))), 3)
y_err_raw = round(abs(mean(np.subtract(orig_y, pred_y))), 3)
z_err_raw = round(abs(mean(np.subtract(orig_z, pred_z))), 3)
rx_err_raw = round(abs(mean(np.subtract(orig_rx, pred_rx))), 3)
ry_err_raw = round(abs(mean(np.subtract(orig_ry, pred_ry))), 3)
rz_err_raw = round(abs(mean(np.subtract(orig_rz, pred_rz))), 3)

plt.subplot(2, 3, 1)
plt.rc('axes', titlesize=25)
plt.rc('legend',fontsize=25)
# title = 'Barge Dimensions ' + str(baseline_input[0]) + ' m Length, ' + str(baseline_input[1]) + ' m Beam, ' + \
   # str(abs(baseline_input[2])) + ' m Draft  -  Waves Heading of: ' + str(baseline_input[3])
# plt.suptitle(title)
plt.plot(x_axis, pred_x, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_x, color='blue', linestyle='-.', label='True RAO')
plt.title('Surge RAO')
plt.ylabel('Response (m/m)')
plt.text(mean(x_axis), (mean(orig_x)+mean(pred_x))/2, "Relative % diff: "+ str(x_err_rpd)+'\n'+"Average raw diff: " + str(x_err_raw))
plt.grid()
# plt.ylim([-0.5, 1.5])
# plt.legend()

# plt.show()
plt.subplot(2, 3, 2)
# plt.rc('font', size=25)
plt.plot(x_axis, pred_y, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_y, color='blue', linestyle='-.', label='True RAO')
plt.title('Sway RAO')

plt.text(mean(x_axis), (mean(orig_y)+mean(pred_y))/2, "Relative % diff: "+ str(y_err_rpd)+'\n'+"Average raw diff: " + str(y_err_raw))

plt.grid()
# plt.legend()
# plt.rc('font', size=10)
# plt.ylim([-0.5, 1.5])
# plt.show()
plt.subplot(2, 3, 3)
plt.plot(x_axis, pred_z, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_z, color='blue', linestyle='-.', label='True RAO')
plt.title('Heave RAO')
plt.text(mean(x_axis), (mean(orig_z)+mean(pred_z))/2, "Relative % diff: "+ str(z_err_rpd)+'\n'+"Average raw diff: " + str(z_err_raw))
plt.grid()
# plt.ylim([-0.5, 1.5])

plt.subplot(2, 3, 4)
plt.plot(x_axis, pred_rx, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rx, color='blue', linestyle='-.', label='True RAO')
plt.title('Roll RAO')
plt.text(mean(x_axis), (mean(orig_rx)+mean(pred_rx))/2, "Relative % diff: "+ str(rx_err_rpd)+'\n'+"Average raw diff: " + str(rx_err_raw))
# plt.ylim([-0.5, 1.5])
plt.ylabel('Response (Deg/m)')
plt.xlabel('Wave Frequency (rad/s)')
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(x_axis, pred_ry, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_ry, color='blue', linestyle='-.', label='True RAO')
plt.title('Pitch RAO')
plt.text(mean(x_axis), (mean(orig_ry)+mean(pred_ry))/2, "Relative % diff: "+ str(ry_err_rpd)+'\n'+"Average raw diff: " + str(ry_err_raw))
# plt.ylim([-0.5, 50])
plt.grid()
plt.xlabel('Wave Frequency (rad/s)')

plt.subplot(2, 3, 6)
plt.plot(x_axis, pred_rz, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rz, color='blue', linestyle='-.', label='True RAO')
plt.title('Yaw RAO')
plt.text(mean(x_axis), (mean(orig_rz)+mean(pred_rz))/2, "Relative % diff: "+ str(rz_err_rpd)+'\n'+"Average raw diff: " + str(rz_err_raw))
# plt.ylim([-0.5, 1.5])
plt.grid()
plt.xlabel('Wave Frequency (rad/s)')
                                                                
plt.legend()
# plt.get_current_fig_manager().full_screen_toggle()
plt.show()