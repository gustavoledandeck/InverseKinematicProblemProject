import os

import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf




# This is the sketch of tensorflowmodel implementation version one

class TensorFlowModel:
    """
    NNA model for IK with tensorflow
    """

    def __init__(self, input_dimension, output_dimension, hidden_layers=[128, 256, 128], activation='relu'):
        """
            This is the constructor of TF model.
            Initialize the tensorflow model.

            Args:
                input_dimension (int): Input dimension (end-effector position)
                output_dimension (int): Output dimension (joint angles)
                hidden_layers (list) : List of neurons in each hidden layer
                activation (str) : Activation function to use
        """

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        self.activation = activation

        # Input and output scalers:
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        # Build the model
        self.model = self._build_model()

    def _build_model(self):
        """
            A kind of Builder, or something else you wanna call it, for the
            tensorflow NNA model.

        :return:
            tf.keras.Model ----> Compiled TensorFlow model
        """

        model = keras.Sequential()

        #input layer
        model.add(tf.keras.layers.InputSpec(shape=(self.input_dimension,)))

        #Hidden layers
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=self.activation))
            model.add(tf.keras.layers.BatchNormalization())

        #output layer
        model.add(tf.keras.layers.Dense(self.output_dimension))

        #compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']

        )

        return model
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the model on the provided dataset.

        Args:
            X (numpy.ndarray) : Input data (End-Effector positions)
            y (numpy.ndarray) : Target data (joint angles)
            epochs (int) : Number of training epochs
            batch_size (int) : Batch size for training
            validation_split (float) : Fraction of data to use for validation
            Verbose (int) : Verbosity level

        :return:
            Dict: Training history
        """

        #Fit the scalers
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)

        #Train the model
        start_time = time.time()
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        training_time = time.time() - start_time

        print(f"TF model training completed in {training_time:.2f} seconds")

        return history.history

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


class PyTorchModel:
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
class ScikitLearnModel:
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

if __name__ == "__main__":
