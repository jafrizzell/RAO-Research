import pandas as pd
import numpy as np
import os
from math import e, sqrt
from scipy.optimize import curve_fit
from statistics import mean

from sklearn.metrics import mean_squared_error

base_path = 'D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/med_barge/'

output_file = pd.read_csv("D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/mid_size_barges.csv", sep=',')
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

        filedata = open(base_path+file).readlines()
        for i in range(len(filedata)):
            if "R.A.O.S-VARIATION WITH WAVE PERIOD/FREQUENCY" in filedata[i] and "VEL" not in filedata[i] and "ACC" not in filedata[i]:
                data_holder.append(np.array(filedata[i+7:i+27]))
                data_holder.append(np.array(filedata[i+28:i+48]))
        for a in range(5):
            data_holder2 = []
            data_holder3 = list()
            try:
                data_holder_t = data_holder[a]
                for j in data_holder_t:
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
                            rmse = [float(sqrt(mean_squared_error(data[2+2*i], predicted)))]
                            m = [float(mean(predicted))]
                            coeffs = list(coeffs)
                            coeffs.extend(rmse)
                            coeffs.extend(m)
                            line.extend(coeffs)
                        except RuntimeError:
                            line.extend(['na', 'na', 'na', 'na', 'na'])
                except ValueError:
                    pass
                writer.append(line)
            except IndexError:
                pass

        count += 1
    writer_data = pd.DataFrame(writer)

col = ['Length', 'Beam', 'Draft', 'Heading',
           'Asurge', 'Bsurge', 'Csurge', 'RMSEsurge', 'Msurge',
           'Asway', 'Bsway', 'Csway', 'RMSEsway', 'Msway',
           'Aheave', 'Bheave', 'Cheave', 'RMSEheave', 'Mheave',
           'Aroll', 'Broll', 'Croll', 'RMSEroll', 'Mroll',
           'Apitch', 'Bpitch', 'Cpitch', 'RMSEpitch', 'Mpitch',
           'Ayaw', 'Byaw', 'Cyaw', 'RMSEyaw', 'Myaw']

writer_data.columns = col
writer_data.to_csv(base_path+'mid_data8.csv', index=False)
