from RaoPrediction_class import RaoPrediction
import cProfile
import timeit

# Start inputs

length = .7  # Barge length
beam = .3  # Barge beam
draft = .21  # Barge draft
heading = 180  # Wave heading, 0 degrees is head seas, 90 is beam

model_path = "C:/Users/jafri/Documents/GitHub/RAO-Research/parameter_fit_data_with code/damped_spring_1dof/"  # path the the trained NN model directory

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
    n = 100
    print(timeit.timeit(stmt=main, number=n)/n)
