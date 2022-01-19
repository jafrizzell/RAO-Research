from RaoPrediction_class import RaoPrediction

# Start inputs

length = 40  # Barge length
beam = 22  # Barge beam
draft = 2.5  # Barge draft
heading = 180  # Wave heading, 0 degrees is head seas, 90 is beam

model_path = "D:/IdeaProjects/PyCharm/TAMU_Work/OCEN 485/damped_spring_all_dir3"  # path the the trained NN model directory

low_freq = 0.1  # lowest wave frequency to predict the RAOs at
high_freq = 2.5  # highest wave frequency to predict the RAOs at
n_points = 25  # Number of points to predict the RAOs at, higher = more resolution in plot

# End Inputs

# Initialize rao class structure and predict RAOs with given inputs

rao = RaoPrediction()

rao.dnn(model_path)
rao.predict(length, beam, draft, heading)
rao.visualize(low_freq, high_freq, n_points)
