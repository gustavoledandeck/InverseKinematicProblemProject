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

        model.add(keras.layers.Input(shape=(self.input_dimension,)))

        #Hidden layers
        for units in self.hidden_layers:

            model.add(keras.layers.Dense(units, activation=self.activation))
            model.add(keras.layers.BatchNormalization())


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

    def predict(self, X):
        """
        Predict joint angles for given end-effector positions.

        Args:
            X (numpy.ndarray) : Input data (End-effector positions)
        :return:
            numpy.ndarray : Predicted joint angles
        """
        #Scale the input
        X_scaled = self.input_scaler.transform(X)


        #Make predictions
        start_time = time.time()
        y_scaled_pred = self.model.predict(X_scaled)
        inference_time = time.time() - start_time

        #Inverse transform to get original scale
        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)

        print(f"TF model inference time: {inference_time*1000:.2f} ms")

        return y_pred
    def evaluate(self, X, y_true):
        """
            Evaluate the model performance.

            Args:
                   X (numpy.ndarray) : Input data (end-effector positions)
                   y_true (numpy.ndarray) : True joint angles

        :return:
            dict : Evaluation metrics
        """
        # Predict joint angles
        y_pred = self.predict(X)

        # Calculate mean absolute error for each joint
        mae_per_joint = np.mean(np.abs(y_true - y_pred), axis=0)

        # Calculate overall mean absolute error
        mae_overall = np.mean(mae_per_joint)

        # Calculate Euclidean distance error in joint space
        euclidean_error = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        mean_euclidean_error = np.mean(euclidean_error)

        return {
            'mae_per_joint': mae_per_joint,
            'mae_overall': mae_overall,
            'mean_euclidean_error': mean_euclidean_error,
            'max_euclidean_error': np.max(euclidean_error)
        }

    def save(self, model_path):
        """
            Save the model to disk.

        Args: model_path (str) : Path to save the model


        """

        #Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        #Save the model
        self.model.save(model_path)

        #Save scalers
        np.save(os.path.join(os.path.dirname(model_path), 'input_scaler.npy'),
                [self.input_scaler.data_min_, self.input_scaler.data_max_])
        np.save(os.path.join(os.path.dirname(model_path), 'output_scaler.npy'),
                [self.output_scaler.data_min_], self.output_scaler.data_max_)


    def load(self, model_path):
        """
            Load the model from disk.

            Args:
                model_path (str) : Path to load the model from

        """

        #Load the model
        self.model = keras.models.load_model(model_path)

        #Load scalers
        input_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'input_scaler.npy'))
        output_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'output_scaler.npy'))



        self.input_scaler.data_min_ = input_scaler_data[0]
        self.input_scaler.data_max_ = input_scaler_data[1]
        self.input_scaler.scale_ = 1.0 / (self.input_scaler.data_max_ - self.input_scaler.data_min_)

        self.output_scaler.data_min_ = output_scaler_data[0]
        self.output_scaler.data_max_ = output_scaler_data[1]
        self.output_scaler.scale_ = 1.0 / (self.output_scaler.data_max_ - self.output_scaler.data_min_)

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
    #Generate some random data to use in the models

    np.random.seed(42)
    X = np.random.rand(1000, 3) # END-EFFECTOR POSITIONS (X, Y, Z)
    y = np.random.rand(1000, 4) # JOINT ANGLES (Theta 0, Theta 1, Theta 02,Theta 3)

    #Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Create and train TF model
    tf_model = TensorFlowModel(input_dimension=3, output_dimension=4)
    tf_history = tf_model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)





    #Evaluate models
    tf_metrics = tf_model.evaluate(X_test, y_test)
    #print(tf_metrics['mae_per_joint'])

    print("\nTF Model Metrics: ")
    print("MAE per Joint: ")
    print(tf_metrics['mae_per_joint'])
    print("\n")
    print("Mean Euclidean Error: ")
    print(tf_metrics['mean_euclidean_error'])


