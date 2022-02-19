from RaoPrediction_class import RaoPrediction
import cProfile
import timeit
import os

# Start inputs

length = 23.8  # Barge length
beam = 12.5  # Barge beam
draft = 0.78  # Barge draft
heading = 160  # Wave heading, 0 degrees is head seas, 90 is beam

model = '/multi_eq_1.0.h5'
base = os.getcwd()
model_path = base+model  # path the the trained NN model directory  # path the the trained NN model directory

low_freq = 0.1  # lowest wave frequency to predict the RAOs at
high_freq = 2.5  # highest wave frequency to predict the RAOs at
n_points = 25  # Number of points to predict the RAOs at, higher = more resolution in plot

# End Inputs

# Initialize rao class structure and predict RAOs with given inputs


def main():
    rao = RaoPrediction()

    rao.dnn(model_path)
    rao.predict(length, beam, draft, heading)
    rao.visualize(low_freq, high_freq, n_points)


if __name__ == "__main__":
    cProfile.run('main()')
    # n = 100
    # print(timeit.timeit(stmt=main, number=n)/n)
