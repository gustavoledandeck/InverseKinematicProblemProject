import os
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# This is the sketch of tensorflowmodel implementation version one

class TensorFlowModel:
    """
    NNA model for IK with tensorflow
    """

    def __init__(self):
        """
        Initialize the tensorflow model.

        Args:  (this will be the key args)
                x
                y
                z
                a
                b
        """

    def _build_model(self):
        """
        A kind of Builder, or something else you wanna call it, for the
        tensorflow NNA model.

        :return:
            Should return:
                tf.keras.Model ----> Compiled TensorFlow model
        """

    def train(self):
        """
        Train the model on the provided dataset.

        Args:
            not yet known, but ASAP will.
        :return:
        """

    def predict(self):
        """
        Predict joint angles for given end-effector positions.

        Args: TO-DO
        :return:
        """


    def evaluate(self):
        """
        ARGS: TO-DO
        :return:
        """

    def save(self):
        """
        Save the model to disk.

        Args: TO-DO
        :return:
        """

    def load(self):
        """
        Load the model from disk.

        TO-DO
        :return:
        """


"""
class PyTorchModel:
    AN IDEA FOR THE FUTURE, TEST PYTORCH FOR THIS PROJECT
"""
