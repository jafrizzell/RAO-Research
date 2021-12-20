import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

# Import data
raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/damped/damped_results_all_dir.csv", sep=',')

# Pre-process data, split into train and test datasets
raw_data.dropna(axis=0, inplace=True)

raw_data = raw_data.apply(pd.to_numeric)
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

order = 3  # Numbert of coefficients in the curve fit data
train_labels = np.asarray(train_features.drop(train_features.columns[list(range(4,6*(order)+4))], axis=1, inplace=False))
test_labels = np.asarray(test_features.drop(test_features.columns[list(range(4,6*(order)+4))], axis=1, inplace=False))

train_features = train_features.drop(train_features.columns[list(range(0,4))], axis=1, inplace=False)
test_features = test_features.drop(test_features.columns[list(range(0,4))], axis=1, inplace=False)

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def build_and_compile_model(norm):
    # Adjust the number of hidden layers and neurons per layer that results in best fit NN
    inputs = keras.Input(shape=(4,))
    dense1 = layers.Dense(256, activation='elu')(inputs)
    dense2 = layers.Dense(256, activation='elu')(dense1)
    # dense3 = layers.Dense(256, activation='relu')(dense2)
    # dense4 = layers.Dense(64, activation='relu')(dense3)
    # dense5 = layers.Dense(64, activation='relu')(dense4)
    # dense6 = layers.Dense(64, activation='relu')(dense5)
    outputs = layers.Dense(6*(order+1))(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(train_labels, train_features, validation_split=0.2, verbose=0, epochs=1000)
plot_loss(history)

test_results = dnn_model.evaluate(test_labels, test_features, verbose=0)

test_predictions = dnn_model.predict(test_labels).flatten()

print("R Squared results = ", r2_score(np.asarray(test_features).flatten(), test_predictions))

a = plt.axes(aspect='equal')
plt.scatter(test_features, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 35]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

dnn_model.save('damped_spring_all_dir3')

baseline = np.asarray(test_dataset.sample(n=1))[0]
baseline_input = baseline[0:4]
baseline_prediction = baseline[4:]

new_input = [baseline_input.tolist()]
new_pred = dnn_model.predict(new_input)[0]
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
plt.suptitle(baseline_input)
plt.plot(x_axis, pred_x, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_x, color='blue', label='True RAO')
plt.title('Surge RAO')
# plt.ylim([-0.5, 1.5])
plt.ylabel('Response (m/m)')

plt.subplot(2, 3, 2)
plt.plot(x_axis, pred_y, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_y, color='blue', label='True RAO')
plt.title('Sway RAO')
# plt.ylim([-0.5, 1.5])

plt.subplot(2, 3, 3)
plt.plot(x_axis, pred_z, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_z, color='blue', label='True RAO')
plt.title('Heave RAO')
# plt.ylim([-0.5, 1.5])

plt.subplot(2, 3, 4)
plt.plot(x_axis, pred_rx, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rx, color='blue', label='True RAO')
plt.title('Roll RAO')
# plt.ylim([-0.5, 1.5])
plt.ylabel('Response (Deg/m)')
plt.xlabel('Wave Frequency (rad/s)')

plt.subplot(2, 3, 5)
plt.plot(x_axis, pred_ry, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_ry, color='blue', label='True RAO')
plt.title('Pitch RAO')
# plt.ylim([-0.5, 50])
plt.xlabel('Wave Frequency (rad/s)')

plt.subplot(2, 3, 6)
plt.plot(x_axis, pred_rz, color='red', label='Predicted RAO')
plt.plot(x_axis, orig_rz, color='blue', label='True RAO')
plt.title('Yaw RAO')
# plt.ylim([-0.5, 1.5])
plt.xlabel('Wave Frequency (rad/s)')

plt.legend()
plt.get_current_fig_manager().full_screen_toggle()
plt.show()