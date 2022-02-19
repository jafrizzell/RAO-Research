import tensorflow as tf
import numpy as np
from math import e
from matplotlib import pyplot as plt


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


class RaoPredictor:
    def __init__(self):
        self.model = None
        self.params = [0, 0, 0]
        self.f = []
        self.surge = []
        self.sway = []
        self.heave = []
        self.roll = []
        self.pitch = []
        self.yaw = []

    def dnn(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, length, beam, draft, heading):
        if heading > 0:
            heading = -1 * heading
        self.model.summary()
        self.params = self.model.predict([[length, beam, draft, heading]])[0]

    def visualize(self, low, high, n):
        self.f = np.linspace(low, high, n)
        order = 3
        for i in self.f:
            self.surge.append(damped_func(i, *self.params[0*order:0*order+order]))
            self.sway.append(damped_func(i, *self.params[1*order:1*order+order]))
            self.heave.append(arctan_func(i, *self.params[2*order:2*order+order]))
            self.roll.append(gauss_func(i, *self.params[3*order:3*order+order]))
            self.pitch.append(gauss_func(i, *self.params[4*order:4*order+order]))
            self.yaw.append(gauss_func(i, *self.params[5*order:5*order+order]))
        plt.subplot(2, 3, 1)
        plt.rc('axes', titlesize=25)
        plt.rc('legend', fontsize=25)
        # title = 'Barge Dimensions ' + str(baseline_input[0]) + ' m Length, ' + str(baseline_input[1]) + ' m Beam, ' + \
        # str(abs(baseline_input[2])) + ' m Draft  -  Waves Heading of: ' + str(baseline_input[3])
        # plt.suptitle(title)
        plt.plot(self.f, self.surge, color='blue')
        plt.title('Surge')
        plt.ylabel('Response (m/m)')
        plt.grid()
        # plt.ylim([-0.5, 1.5])
        # plt.legend()

        #plt.show()
        plt.subplot(2, 3, 2)
        # plt.rc('font', size=25)
        plt.plot(self.f, self.sway, color='blue')
        plt.title('Sway')


        plt.grid()
        # plt.legend()
        # plt.rc('font', size=10)
        # plt.ylim([-0.5, 1.5])
        #plt.show()
        plt.subplot(2, 3, 3)
        plt.plot(self.f, self.heave, color='blue')
        plt.title('Heave')
        plt.grid()
        # plt.ylim([-0.5, 1.5])

        plt.subplot(2, 3, 4)
        plt.plot(self.f, self.roll, color='blue')
        plt.title('Roll')
        # plt.ylim([-0.5, 1.5])
        plt.ylabel('Response (Deg/m)')
        plt.xlabel('Wave Frequency (rad/s)')
        plt.grid()

        plt.subplot(2, 3, 5)
        plt.plot(self.f, self.pitch, color='blue')
        plt.title('Pitch')
        # plt.ylim([-0.5, 50])
        plt.grid()
        plt.xlabel('Wave Frequency (rad/s)')

        plt.subplot(2, 3, 6)
        plt.plot(self.f, self.yaw, color='blue')
        plt.title('Yaw')
        # plt.ylim([-0.5, 1.5])
        plt.grid()
        plt.xlabel('Wave Frequency (rad/s)')

        # plt.get_current_fig_manager().full_screen_toggle()
        plt.show()
