import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import RegscorePy
from itertools import product

# Import data
raw_data = pd.read_csv("C:/Users/jafri/Documents/GitHub/RAO-Research/new_fit/damped/damped_results_all_dir.csv", sep=',')

# TODO: Correlation between independent variables

# Pre-process data, split into train and test datasets
print(raw_data.isna().sum())
raw_data = raw_data.apply(pd.to_numeric)
# raw_data = raw_data[(raw_data['Length (m)'] >= 2) | (raw_data['Heading'] != -90)]
raw_data = raw_data[(raw_data['Length (m)'] >= 2)]
raw_data.dropna(axis=0, inplace=True)
print(raw_data.shape)
column1 = raw_data['Length (m)']
column2 = raw_data['Beam (m)']
column3 = raw_data['Draft (m)']
column4 = raw_data['Heading']

raw_data.pop('R2surge')
raw_data.pop('R2sway')
raw_data.pop('R2heave')
raw_data.pop('R2roll')
raw_data.pop('R2pitch')
raw_data.pop('R2yaw')
raw_data.pop('MAEsurge')
raw_data.pop('MAEsway')
raw_data.pop('MAEheave')
raw_data.pop('MAEroll')
raw_data.pop('MAEpitch')
raw_data.pop('MAEyaw')

# raw_data = raw_data[raw_data['Heading'] == (-180.0 or 0)]
# TODO: K-Fold Validation instead of train/test
train_dataset = raw_data.sample(frac=0.8, random_state=0)

test_dataset = raw_data.drop(train_dataset.index)


train_features = train_dataset.copy()
test_features = test_dataset.copy()
order = 3  # Number of coefficients in the curve fit data
num_dof = 6

train_labels = np.asarray(train_features.drop(train_features.columns[list(range(4,6*(order)+4))], axis=1, inplace=False))
test_labels = np.asarray(test_features.drop(test_features.columns[list(range(4,6*(order)+4))], axis=1, inplace=False))

train_features = train_features.drop(train_features.columns[list(range(0,4))], axis=1, inplace=False)
test_features = test_features.drop(test_features.columns[list(range(0,4))], axis=1, inplace=False)

# print(train_labels)

# train_features = train_features[['Aheave', 'Bheave', 'Cheave', 'Apitch', 'Bpitch', 'Cpitch']]
# test_features = test_features[['Aheave', 'Bheave', 'Cheave', 'Apitch', 'Bpitch', 'Cpitch']]

# train_features = train_features[['Apitch', 'Bpitch', 'Cpitch']]
# test_features = test_features[['Apitch', 'Bpitch', 'Cpitch']]

print(train_features.shape)
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


def fit_and_evaluate(norm, architecture, dof):
    dnn_model = build_and_compile_model(norm, architecture, dof)
    # dnn_model.summary()

    history = dnn_model.fit(train_labels, train_features, validation_split=0.2, verbose=0, epochs=1000)
    plot_loss(history)

    test_results = dnn_model.evaluate(test_labels, test_features, verbose=0)
    # print(test_results)
    test_predictions = dnn_model.predict(test_labels).flatten()

    r2 = r2_score(np.asarray(test_features).flatten(), test_predictions)
    mae = mean_absolute_error(np.asarray(test_features).flatten(), test_predictions)
    print(mae)
    aic = RegscorePy.aic.aic(np.asarray(test_features, dtype=float).flatten(), np.asarray(test_predictions).astype(float), 4+2)

    return dnn_model, aic, r2, test_predictions


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 30])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    # pass


def build_and_compile_model(norm, arch, dof):
    # Adjust the number of hidden layers and neurons per layer that results in best fit NN
    inputs = keras.Input(shape=(4,))
    if arch[1] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        # dense2 = layers.Dense(arch[1], activation='elu')(inputs)
        dense3 = layers.Dense(arch[2], activation='relu')(dense1)
        outputs = layers.Dense(dof*(order))(dense3)

    elif arch[2] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense2 = layers.Dense(arch[1], activation='elu')(dense1)
        # dense3 = layers.Dense(arch[2], activation='relu')(dense1)
        outputs = layers.Dense(dof*(order))(dense2)

    elif arch[1] == 0.0 and arch[2] == 0.0:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        # dense2 = layers.Dense(arch[1], activation='elu')(inputs)
        # dense3 = layers.Dense(arch[2], activation='relu')(dense1)
        outputs = layers.Dense(dof*(order))(dense1)

    else:
        norm_layer = layers.BatchNormalization()(inputs)
        dense1 = layers.Dense(arch[0], activation='relu')(norm_layer)
        dense2 = layers.Dense(arch[1], activation='relu')(dense1)
        # norm_layer2 = layers.BatchNormalization()(dense2)
        dense3 = layers.Dense(arch[2], activation='relu')(dense2)
        dense4 = layers.Dense(arch[3], activation='relu')(dense3)
        dense5 = layers.Dense(arch[4], activation='relu')(dense4)
        outputs = layers.Dense(dof*(order))(dense5)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# tuner = kt.Hyperband(build_and_compile_model,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='intro_to_kt')
#
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# tuner.search(train_features, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


models = []
aic_scores = []
r2_scores = []
l1 = np.linspace(32, 256, 8)
l2 = np.linspace(0, 256, 9)
l3 = np.linspace(0, 256, 9)
# parametric_space = list(product(*[l1, l2, l3]))
parametric_space = [[512, 512, 512, 512, 512]]
print(parametric_space)
start_t = time.time()
c = 1

for arch in parametric_space:
    print('Progress: ' + str(c) + '/' + str(len(parametric_space)))
    dnn_model, aic, r2, test_predictions = fit_and_evaluate(normalizer, arch, num_dof)
    models.append(dnn_model)
    aic_scores.append(aic)
    r2_scores.append(r2)

    curr_time = time.time()
    diff_t = curr_time - start_t
    t_per_model = diff_t / c
    num_mods_rem = len(parametric_space) - c
    t_rem = t_per_model * num_mods_rem
    print("Estimated Time Remaining: " + time.strftime('%H:%M:%S', time.gmtime(t_rem)) + ' seconds')
    c += 1

parametric_space_t = np.asarray(parametric_space).transpose().tolist()
output_data = [parametric_space_t[0], parametric_space_t[1], parametric_space_t[1], aic_scores, r2_scores]
output_data = np.asarray(output_data).transpose().tolist()
print(output_data)
oput = pd.DataFrame(output_data, columns=['L1', 'L2', 'L3', 'AIC', 'R2'])
print(oput)
# oput.to_csv('Parametric_space_study.csv', index=False)


a = plt.axes(aspect='equal')
plt.scatter(test_features, test_predictions, s=0.8)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 35]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

dnn_model.save('multi_eq_0.7')


# baseline = np.asarray(test_dataset.sample(n=1))[0]
# baseline_input = baseline[0:4]
# baseline_prediction = baseline[4:]
#
# new_input = [baseline_input.tolist()]
# new_pred = dnn_model.predict(new_input)[0]
# orig_x = []
# orig_y = []
# orig_z = []
# orig_rx = []
# orig_ry = []
# orig_rz = []
# pred_x = []
# pred_y = []
# pred_z = []
# pred_rx = []
# pred_ry = []
# pred_rz = []
#
#
# print(baseline_input)
# print('\n\n')
#
# x_axis = np.linspace(0.1, 2.5, 60)
# for i in x_axis:
#     orig_x.append(np.polyval(baseline_prediction[0*order:0*order+order], i))
#     orig_y.append(np.polyval(baseline_prediction[1*order:1*order+order], i))
#     orig_z.append(np.polyval(baseline_prediction[2*order:2*order+order], i))
#     orig_rx.append(np.polyval(baseline_prediction[3*order:3*order+order], i))
#     orig_ry.append(np.polyval(baseline_prediction[4*order:4*order+order], i))
    # orig_rz.append(np.polyval(baseline_prediction[5*order:5*order+order], i))
#     pred_x.append(np.polyval(new_pred[0*order:0*order+order], i))
#     pred_y.append(np.polyval(new_pred[1*order:1*order+order], i))
#     pred_z.append(np.polyval(new_pred[2*order:2*order+order], i))
#     pred_rx.append(np.polyval(new_pred[3*order:3*order+order], i))
#     pred_ry.append(np.polyval(new_pred[4*order:4*order+order], i))
#     pred_rz.append(np.polyval(new_pred[5*order:5*order+order], i))
#
# plt.subplot(2, 3, 1)
# plt.suptitle(baseline_input)
# plt.plot(x_axis, pred_x, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_x, color='blue', label='True RAO')
# plt.title('Surge RAO')
# # plt.ylim([-0.5, 1.5])
# plt.ylabel('Response (m/m)')
#
# plt.subplot(2, 3, 2)
# plt.plot(x_axis, pred_y, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_y, color='blue', label='True RAO')
# plt.title('Sway RAO')
# # plt.ylim([-0.5, 1.5])
#
# plt.subplot(2, 3, 3)
# plt.plot(x_axis, pred_z, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_z, color='blue', label='True RAO')
# plt.title('Heave RAO')
# # plt.ylim([-0.5, 1.5])
#
# plt.subplot(2, 3, 4)
# plt.plot(x_axis, pred_rx, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_rx, color='blue', label='True RAO')
# plt.title('Roll RAO')
# # plt.ylim([-0.5, 1.5])
# plt.ylabel('Response (Deg/m)')
# plt.xlabel('Wave Frequency (rad/s)')
#
# plt.subplot(2, 3, 5)
# plt.plot(x_axis, pred_ry, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_ry, color='blue', label='True RAO')
# plt.title('Pitch RAO')
# # plt.ylim([-0.5, 50])
# plt.xlabel('Wave Frequency (rad/s)')
#
# plt.subplot(2, 3, 6)
# plt.plot(x_axis, pred_rz, color='red', label='Predicted RAO')
# plt.plot(x_axis, orig_rz, color='blue', label='True RAO')
# plt.title('Yaw RAO')
# # plt.ylim([-0.5, 1.5])
# plt.xlabel('Wave Frequency (rad/s)')
#
# plt.legend()
# plt.get_current_fig_manager().full_screen_toggle()
#plt.show()