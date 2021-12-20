import pandas as pd
from sklearn.metrics import r2_score
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

load_model = tf.keras.models.load_model("D:\IdeaProjects\PyCharm\TAMU_Work\OCEN 485/2nd_order1")


raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/2nd_order_results.csv", sep=',')
raw_data.dropna(axis=0, inplace=True)
column1 = raw_data['Length (m)']
column2 = raw_data['Beam (m)']
column3 = raw_data['Draft (m)']

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
train_labels = np.asarray(train_features.drop(train_features.columns[list(range(3,6*(order+1)+3))], axis=1, inplace=False))
test_labels = np.asarray(test_features.drop(test_features.columns[list(range(3,6*(order+1)+3))], axis=1, inplace=False))

train_features = train_features.drop(train_features.columns[list(range(0,3))], axis=1, inplace=False)
test_features = test_features.drop(test_features.columns[list(range(0,3))], axis=1, inplace=False)
baseline = np.asarray(test_dataset.sample(n=1))[0]
baseline_input = baseline[0:3]
baseline_prediction = baseline[3:]

new_input = [baseline_input.tolist()]
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



order = order+1
print(baseline_input)
print('\n\n')

x_axis = np.linspace(0.1, 2.5, 60)
for i in x_axis:
    orig_x.append(np.polyval(baseline_prediction[0*order:0*order+order], i))
    orig_y.append(np.polyval(baseline_prediction[1*order:1*order+order], i))
    orig_z.append(np.polyval(baseline_prediction[2*order:2*order+order], i))
    orig_rx.append(np.polyval(baseline_prediction[3*order:3*order+order], i))
    orig_ry.append(np.polyval(baseline_prediction[4*order:4*order+order], i))
    orig_rz.append(np.polyval(baseline_prediction[5*order:5*order+order], i))
    pred_x.append(np.polyval(new_pred[0*order:0*order+order], i))
    pred_y.append(np.polyval(new_pred[1*order:1*order+order], i))
    pred_z.append(np.polyval(new_pred[2*order:2*order+order], i))
    pred_rx.append(np.polyval(new_pred[3*order:3*order+order], i))
    pred_ry.append(np.polyval(new_pred[4*order:4*order+order], i))
    pred_rz.append(np.polyval(new_pred[5*order:5*order+order], i))

plt.subplot(2, 3, 1)
plt.plot(x_axis, pred_x, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_x, color='blue', label='True RAO')
plt.title('Surge RAO')
plt.ylim([-0.5, 1.5])
plt.ylabel('Response (m/m)')

plt.subplot(2, 3, 2)
plt.plot(x_axis, pred_y, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_y, color='blue', label='True RAO')
plt.title('Sway RAO')
plt.ylim([-0.5, 1.5])

plt.subplot(2, 3, 3)
plt.plot(x_axis, pred_z, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_z, color='blue', label='True RAO')
plt.title('Heave RAO')
plt.ylim([-0.5, 1.5])

plt.subplot(2, 3, 4)
plt.plot(x_axis, pred_rx, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rx, color='blue', label='True RAO')
plt.title('Roll RAO')
plt.ylim([-0.5, 1.5])
plt.ylabel('Response (Deg/m)')
plt.xlabel('Wave Frequency (Hz)')

plt.subplot(2, 3, 5)
plt.plot(x_axis, pred_ry, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_ry, color='blue', label='True RAO')
plt.title('Pitch RAO')
#plt.ylim([-0.5, 50])
plt.xlabel('Wave Frequency (Hz)')

plt.subplot(2, 3, 6)
plt.plot(x_axis, pred_rz, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rz, color='blue', label='True RAO')
plt.title('Yaw RAO')
plt.ylim([-0.5, 1.5])
plt.xlabel('Wave Frequency (Hz)')

plt.legend()
plt.get_current_fig_manager().full_screen_toggle()
plt.show()