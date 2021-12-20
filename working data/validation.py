import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

path = "D:\TAMU Work\TAMU 2021 FALL\OCEN 485/validation/ANALYSIS.LIS"
new_path = "D:\TAMU Work\TAMU 2021 FALL\OCEN 485/validation/validation_fit.csv"
output_file = pd.read_csv( "D:\TAMU Work\TAMU 2021 FALL\OCEN 485/validation/validation.csv")
data_holder = []
data_holder2 = []
data_holder3 = []

writer = []
writer_data = pd.DataFrame()

filedata = open(path, 'r').readlines()
count = 0
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

    data = data.drop([3, 5, 7, 9, 11, 13], axis=1)
    print(data)
    deg = 3
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
            plt.plot(data[1], data[2+2*i])
            plt.plot(data[1], predicted)
            plt.show()
    except ValueError:
        pass
    writer.append(line)
except IndexError:
    pass
writer_data = pd.DataFrame(writer)

writer_data.to_csv(new_path)
