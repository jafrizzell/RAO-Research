# **Rapid RAO Prediction**

A basic package to predict the Response Amplitude Operators of a box barge vessel using deep learning techniques. Read the conference paper (here).

# **Current Capabilities:**

 - Predict and visualize the RAOs of box barges
 - 6 degrees of freedom (surge, sway, heave, roll, pitch, yaw) supported
 - Assign wave direction from any angle (-180 to 180 degrees)

# **Technology:**

This package uses a Feed-Forward Neural Network (FFNN) made in Tensorflow and Keras to predict the parameters of equations that have been fit to RAO curves.

That is to say - this model does not directly predict the RAO, it predicts a curve that should represent the RAO.

Equations that the model fits to:

| Equation Type | Equation | Degrees of Freedom |
|---------------|----------|--------------------|
|       Critically damped spring-mass-damper        |   ![img.png](Docs/springeq.png)       |    surge, sway                |
|        arctangent       |  ![img.png](Docs/arctaneq.png)        |                 heave   |
|           Gaussian distribution    |      ![img.png](Docs/gausseq.png)    |          roll, pitch, yaw          |


Using these equations, the model predicts three coefficient parameters based on the input parameters to generate an RAO spectrum for the vessel.

The conference paper goes into great detail about the data collection process, model creation, and error analysis.

# **Dependencies**

- Python 3.+
- Tensorflow 2.7.+
- Numpy
- matplotlib

# **What's New? - v0.1.0**

- Initial release