#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from std_msgs.msg import Float64
import matplotlib.pyplot as plt

class GaussianProcessNode:
    def __init__(self):
        rospy.init_node('gaussian_process_node')

        # Initialize Gaussian Process Regressor
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

        # ROS Subscribers and Publishers
        self.input_sub = rospy.Subscriber('/vrpn_client_node/Arm/pose', Float64, self.input_callback)
        self.output_pub = rospy.Publisher('/predicted_arm_tf', Float64, queue_size=10)

        # Data storage
        self.X_train = np.array([])
        self.y_train = np.array([])

        # Visualization
        self.time_history = []
        self.actual_output_history = []
        self.predicted_output_history = []

        # Set up plotting
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Output')
        self.ax.set_title('Gaussian Process Prediction')
        self.line_actual, = self.ax.plot([], [], label='Actual Output', marker='o')
        self.line_predicted, = self.ax.plot([], [], label='Predicted Output', linestyle='--', marker='x')
        self.ax.legend()
        self.fig.canvas.draw()

    def input_callback(self, data):
        # Callback for processing incoming data
        x_new = np.array([[rospy.get_time()]])  # Assuming time as input, modify as needed
        y_new = np.array([[data.data]])

        # Update training data
        self.X_train = np.append(self.X_train, x_new)
        self.y_train = np.append(self.y_train, y_new)

        # Train the Gaussian Process
        self.gp.fit(self.X_train, self.y_train)

        # Predict using the trained GP
        x_pred = np.array([[rospy.get_time()]])  # Modify as needed
        y_pred_mean, y_pred_std = self.gp.predict(x_pred, return_std=True)

        # Publish the predicted output
        predicted_output = Float64()
        predicted_output.data = y_pred_mean[0]
        self.output_pub.publish(predicted_output)

        # Update visualization history
        self.time_history.append(x_new[0, 0])
        self.actual_output_history.append(y_new[0, 0])
        self.predicted_output_history.append(y_pred_mean[0])

        # Update the plot
        self.line_actual.set_xdata(self.time_history)
        self.line_actual.set_ydata(self.actual_output_history)
        self.line_predicted.set_xdata(self.time_history)
        self.line_predicted.set_ydata(self.predicted_output_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == '__main__':
    try:
        gp_node = GaussianProcessNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
