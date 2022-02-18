import pandas as pd
import numpy as np
import os
from math import e
from scipy.optimize import curve_fit
from statistics import mean

from sklearn.metrics import r2_score, mean_absolute_error

base_path = 'D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/large_barge/'

output_file = pd.read_csv("D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/large_size_barges.csv", sep=',')
writer = []
writer_data = pd.DataFrame()


def damped_func(x, a, b, c):
    # Motion of a critically-damped harmonic motion system
    # Change this function to change the shape of the initial data, to better fit it.
    y = c * e**-(a*x) + b*x*e**-(a*x)
    return y


def gauss_func(x, a, b, c):
    # Motion of a critically-damped harmonic motion system
    # Change this function to change the shape of the initial data, to better fit it.
    y = a * e**-((x-b)**2/c)
    return y


def arctan_func(x, a, b, c):
    # Motion of a critically-damped harmonic motion system
    # Change this function to change the shape of the initial data, to better fit it.
    y = a * np.arctan((x * b + c)) + 0.5
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
        for a in range(8):
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
                            if i  in [0, 1]:
                                coeffs, cov = curve_fit(damped_func, data[1], data[2+2*i], bounds=((0, -0.5, 0), (1, 0.5, 1.5)))
                                predicted = damped_func(data[1], *coeffs)
                                r2 = [float(r2_score(data[2+2*i], predicted))]
                                mae = [float(mean_absolute_error(data[2+2*i], predicted))]
                                coeffs = list(coeffs)
                                coeffs.extend(r2)
                                coeffs.extend(mae)
                                line.extend(coeffs)
                            elif i == 2:
                                coeffs, cov = curve_fit(arctan_func, data[1], data[2+2*i], p0=[-0.5, 0.5, -5],
                                                        bounds=((-2, 0, -10), (0, 2, 0)))
                                predicted = arctan_func(data[1], *coeffs)
                                r2 = [float(r2_score(data[2+2*i], predicted))]
                                mae = [float(mean_absolute_error(data[2+2*i], predicted))]
                                coeffs = list(coeffs)
                                coeffs.extend(r2)
                                coeffs.extend(mae)
                                line.extend(coeffs)
                            elif i in [3, 4, 5]:
                                coeffs, cov = curve_fit(gauss_func, data[1], data[2+2*i])
                                predicted = gauss_func(data[1], *coeffs)
                                r2 = [float(r2_score(data[2+2*i], predicted))]
                                mae = [float(mean_absolute_error(data[2+2*i], predicted))]
                                coeffs = list(coeffs)
                                coeffs.extend(r2)
                                coeffs.extend(mae)
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
           'Asurge', 'Bsurge', 'Csurge', 'R2surge', 'MAEsurge',
           'Asway', 'Bsway', 'Csway', 'R2sway', 'MAEsway',
           'Aheave', 'Bheave', 'Cheave', 'R2heave', 'MAEheave',
           'Aroll', 'Broll', 'Croll', 'R2roll', 'MAEroll',
           'Apitch', 'Bpitch', 'Cpitch', 'R2pitch', 'MAEpitch',
           'Ayaw', 'Byaw', 'Cyaw', 'R2yaw', 'MAEyaw']


writer_data.columns = col
writer_data.to_csv(base_path+'large_data10.csv', index=False)
