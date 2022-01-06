import pandas as pd
import numpy as np
import os
from math import e
from scipy.optimize import curve_fit
import shutil
import matplotlib.pyplot as plt
import sklearn as sklearn

from sklearn.metrics import r2_score

base_path = 'D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/small_barge/'

output_file = pd.read_csv("D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/small_barge_data.csv", sep=',')
writer = []
writer_data = pd.DataFrame()


def func(x, a, b, c):
    # Motion of a critically-damped harmonic motion system
    # Change this function to change the shape of the initial data, to better fit it.
    y = c * e**-(a*x) + b*x*e**-(a*x)
    return y


#  COLLECT RAO DATA FROM EACH FILE IN THE DIRECTORY  #
for subdirs, dirs, files in os.walk(base_path):
    count = 0
    for file in files:
        print("Fitting curve from file ", count)
        data_holder = []
        data_holder2 = []
        data_holder3 = []
        filedata = open(base_path+file).readlines()
        for i in range(len(filedata)):
            if "R.A.O.S-VARIATION WITH WAVE PERIOD/FREQUENCY" in filedata[i] and "VEL" not in filedata[i] and "ACC" not in filedata[i]:
                data_holder.append(np.array(filedata[i+7:i+27]))
        try:
            data_holder = data_holder[2]

            for j in data_holder:
                data_holder2.append(j.tolist().split('\n')[0])
            for k in data_holder2:
                new_split = k.split(' ')
                for item in new_split:
                    try:
                        item = float(item)
                        data_holder3.append(item)
                    except ValueError:
                        pass
            heading = data_holder3[2]
            data_holder3.pop(2)
            data_holder3 = np.asarray(data_holder3).reshape(20,14)
            data = pd.DataFrame(data_holder3)

            data.drop([3, 5, 7, 9, 11, 13], axis=1)
            line = []
            l = output_file['Length (m)'][count]
            b = output_file['Beam (m)'][count]
            try:
                d = float(output_file['Draft (m)'][count])
                line.extend([l, b, d, heading])
                for i in range(6):
                    print('Fitting DoF ', i+1)
                    try:
                        coeffs, cov = curve_fit(func, data[1], data[2+2*i])
                        predicted = func(data[1], *coeffs)
                        r_squared = [float(r2_score(data[2+2*i], predicted))]
                        coeffs = list(coeffs)
                        coeffs.extend(r_squared)
                        line.extend(coeffs)
                    except RuntimeError:
                        line.extend(['na', 'na', 'na', 'na'])
            except ValueError:
                pass
            writer.append(line)
        except IndexError:
            pass

        count += 1
    writer_data = pd.DataFrame(writer)

writer_data.to_csv(base_path+'small_data6.csv')
