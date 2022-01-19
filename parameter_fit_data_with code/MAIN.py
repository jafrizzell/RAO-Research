from RaoPrediction_class import RaoPrediction

length = 40
beam = 22
draft = 2.5
heading = 180

model_path = "D:\IdeaProjects\PyCharm\TAMU_Work\OCEN 485/damped_spring_all_dir3"

rao = RaoPrediction()

rao.dnn(model_path)
rao.predict(length, beam, draft, heading)
rao.visualize(0.1, 2.5, 25)
