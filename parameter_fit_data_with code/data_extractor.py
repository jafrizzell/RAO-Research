import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import sklearn as sklearn

from sklearn.metrics import r2_score

base_path = 'D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/small_barge/'
# raw_data = open('D:/IdeaProjects/PyCharm/TAMU_Work/OCEN 485/ANALYSIS.txt').readlines()
# columns = ['Length (m)', 'Beam (m)', 'Draft (m)', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'r_squared']
output_file = pd.read_csv("D:/TAMU Work/TAMU 2021 FALL/OCEN 485/Analysis Data/small_barge_data.csv", sep=',')
writer = []
writer_data = pd.DataFrame()

#  COLLECT RAO DATA FROM EACH FILE IN THE DIRECTORY  #
for subdirs, dirs, files in os.walk(base_path):
    count = 0
    for file in files:
        data_holder = []
        data_holder2 = []
        data_holder3 = []
        filedata = open(base_path+file).readlines()
        for i in range(len(filedata)):
            if "R.A.O.S-VARIATION WITH WAVE PERIOD/FREQUENCY" in filedata[i] and "VEL" not in filedata[i] and "ACC" not in filedata[i]:
                #print(np.array(filedata[i+7:i+27]))
                data_holder.append(np.array(filedata[i+7:i+27]))
        try:
            data_holder = data_holder[0]
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

            data_holder3.pop(2)
            data_holder3 = np.asarray(data_holder3).reshape(20,14)
            data = pd.DataFrame(data_holder3)

            data.drop([3, 5, 7, 9, 11, 13], axis=1)
            deg = 2
            line = []
            l = output_file['Length (m)'][count]
            b = output_file['Beam (m)'][count]
            try:
                d = float(output_file['Draft (m)'][count])
                line.extend([l, b, d])
                for i in range(6):
                    coeffs = list(np.polyfit(data[1], data[2+2*i], deg))
                    predicted = np.polynomial.polynomial.polyval(data[1], coeffs[::-1])
                    r_squared = [float(r2_score(data[2+2*i], predicted))]
                    coeffs = coeffs
                    coeffs.extend(r_squared)
                    line.extend(coeffs)
            except ValueError:
                pass
            writer.append(line)
        except IndexError:
            pass

        count += 1
    writer_data = pd.DataFrame(writer)

writer_data.to_csv(base_path+'small_data5.csv')
