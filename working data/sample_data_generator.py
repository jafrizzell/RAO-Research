import csv
import numpy as np
import itertools

l = np.linspace(0.3, 0.7, 9)
b = np.linspace(0.15, 0.3, 4)
load = np.linspace(10, 25, 16)

dataset = itertools.permutations(l, len(b))

print(dataset)
