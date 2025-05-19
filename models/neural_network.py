import os

import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPRegressor
import joblib

# This is the sketch of tensorflowmodel implementation version one

class TensorFlowModel:
    """
    NNA model for IK with tensorflow
    """

    def __init__(self, input_dimension, output_dimension, hidden_layers=[32, 64, 32], activation='relu'):
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

            #Add dropout
            model.add(keras.layers.Dropout(0.3))


        #output layer
        model.add(keras.layers.Dense(self.output_dimension))

        #compile
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=10000,
                    decay_rate=0.9
                )
            ),
                loss='mse',
                metrics=['mae']

        )

        return model
    def train(self, X, y, epochs=1000, batch_size=64, validation_split=0.2, verbose=1, callbacks=None):
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

        # Handle callbacks
        default_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            )
        ]

        if callbacks is None:
            callbacks = default_callbacks
        elif isinstance(callbacks, list):
            callbacks = default_callbacks + callbacks

        #Train the model
        start_time = time.time()
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
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
        NNA model for IK with PyTorch.
    """

    def __init__(self, input_dimension, output_dimension, hidden_layers=[32, 64, 32], activation='relu'):
        """
            Initialize the PyTorch model.

        Args:
            input_dim (int): Input dimension (end-effector position)
            output_dim (int): Output dimension (joint angles)
            hidden_layers (list): List of neurons in each hidden layer
            activation (str): Activation function to use
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers

        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()


        # Create input and output scalers
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        # Build the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _build_model(self):
        """
        Build the PyTorch neural network model

        :return:
            torch.nn.Module: PyTorch model
        """

        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dimension, self.hidden_layers[0]))
        layers.append(self.activation)
        layers.append(nn.BatchNorm1d(self.hidden_layers[0]))


        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation)
            layers.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))

        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_dimension))

        return nn.Sequential(*layers)

    def train(self, X, y, epochs=1000, batch_size=64, validation_split=0.2, verbose=1):
        """
            Train the model on the provided dataset.

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y (numpy.ndarray): Target data (joint angles)
                epochs (int): Number of training epochs
                batch_size (int): Batch size for training
                validation_split (float): Fraction of data to use for validation
                verbose (int): Verbosity level
        :return:
            dict: Training history
        """
        #Fit the scalers
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)

        #Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )

        #Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        #Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        #Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': []
        }

        #Early stopping parameters
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        best_model_state = None

        #Train the model
        start_time = time.time()
        for epoch in range(epochs):
            #Training
            self.model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                #Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                #Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            #Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_mae = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()

            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            #Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

            #Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        #Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        print(f"PyTorch model training completed in {training_time:.2f} seconds")

        return history

    def predict(self, X):
        """
        Predict joint angles for given end-effector positions.

        Args:
            X (numpy.ndarray): Input data (end-effector positions)
        :return:
            numpy.ndarray: Predicted joint angles
        """
        #Scale the input
        X_scaled = self.input_scaler.transform(X)

        #Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        #Make predictions
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            y_scaled_pred = self.model(X_tensor).cpu().numpy()
            inference_time = time.time() - start_time

        #Inverse transform to get original scale
        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)

        print(f"PyTorch model inference time: {inference_time * 1000:.2f} ms")

        return y_pred

    def evaluate(self, X, y_true):
        """
            Evaluate the model performance.

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y_true (numpy.ndarray): True joint angles
        :return:
            dict: Evaluation metrics
        """
        #Predict joint angles
        y_pred = self.predict(X)

        #Calculate mean absolute error for each joint
        mae_per_joint = np.mean(np.abs(y_true - y_pred), axis=0)

        #alculate overall mean absolute error
        mae_overall = np.mean(mae_per_joint)

        #Calculate Euclidean distance error in joint space
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

            Args:
                model_path (str): Path to save the model

        """
        #Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        #Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dimension,
            'output_dim': self.output_dimension,
            'hidden_layers': self.hidden_layers
        }, model_path)

        #Save scalers
        np.save(os.path.join(os.path.dirname(model_path), 'pt_input_scaler.npy'),
                [self.input_scaler.data_min_, self.input_scaler.data_max_])
        np.save(os.path.join(os.path.dirname(model_path), 'pt_output_scaler.npy'),
                [self.output_scaler.data_min_, self.output_scaler.data_max_])

    def load(self, model_path):
        """
            Load the model from disk.

            Args:
                model_path (str): Path to load the model from

        """
        #Load the model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_dimension = checkpoint['input_dimension']
        self.output_dimension = checkpoint['output_dimension']
        self.hidden_layers = checkpoint['hidden_layers']

        #Rebuild the model
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        #Load scalers
        input_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'pt_input_scaler.npy'))
        output_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'pt_output_scaler.npy'))

        self.input_scaler.data_min_ = input_scaler_data[0]
        self.input_scaler.data_max_ = input_scaler_data[1]
        self.input_scaler.scale_ = 1.0 / (self.input_scaler.data_max_ - self.input_scaler.data_min_)

        self.output_scaler.data_min_ = output_scaler_data[0]
        self.output_scaler.data_max_ = output_scaler_data[1]
        self.output_scaler.scale_ = 1.0 / (self.output_scaler.data_max_ - self.output_scaler.data_min_)


class ScikitLearnModel:
    """
        Neural network model for IK using scikit-learn.
    """

    def __init__(self, input_dimension, output_dimension, hidden_layers=[32, 64, 32], activation='relu'):
        """
        Initialize the tensorflow model.

        Args:
            input_dimension (int): Input dimension (end-effector position)
            output_dimension (int): Output dimension (joint angles)
            hidden_layers (list): List of neurons in each hidden layer
            activation (str): Activation function to use
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        self.activation = activation

        #Create input and output scalers
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        #Build the model
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            alpha=0.1,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=100,
            shuffle=True,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            verbose=False
        )

    def train(self, X, y, epochs=None, batch_size=None, validation_split=None, verbose=1):
        """
            Train the model on the provided dataset.

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y (numpy.ndarray): Target data (joint angles)
                epochs (int): Not used in scikit-learn (controlled by max_iter)
                batch_size (int): Not used in scikit-learn (controlled by batch_size)
                validation_split (float): Not used in scikit-learn (controlled by validation_fraction)
                verbose (int): Verbosity level
        :return:
            dict: Training history (not available in scikit-learn)
        """
        #Fit the scalers
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)

        #Train the model
        start_time = time.time()
        self.model.fit(X_scaled, y_scaled)
        training_time = time.time() - start_time

        print(f"scikit-learn model training completed in {training_time:.2f} seconds")

        #scikit-learn doesn't provide training history
        return {'loss': [self.model.loss_], 'iterations': self.model.n_iter_}

    def predict(self, X):
        """
            Predict joint angles for given end-effector positions.

            Args:
            X (numpy.ndarray): Input data (end-effector positions)
        :return:
            numpy.ndarray: Predicted joint angles
        """
        #Scale the input
        X_scaled = self.input_scaler.transform(X)

        #Make predictions
        start_time = time.time()
        y_scaled_pred = self.model.predict(X_scaled)
        inference_time = time.time() - start_time

        #Inverse transform to get original scale
        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)

        print(f"scikit-learn model inference time: {inference_time * 1000:.2f} ms")

        return y_pred

    def evaluate(self, X, y_true):
        """
            Evaluate the model performance.

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y_true (numpy.ndarray): True joint angles
        :return:
            dict: Evaluation metrics
        """
        #Predict joint angles
        y_pred = self.predict(X)

        #Calculate mean absolute error for each joint
        mae_per_joint = np.mean(np.abs(y_true - y_pred), axis=0)

        #Calculate overall mean absolute error
        mae_overall = np.mean(mae_per_joint)

        #Calculate Euclidean distance error in joint space
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

            Args:
                model_path (str): Path to save the model
        :return:
        """
        #Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        #Save the model
        joblib.dump(self.model, model_path)

        #Save scalers
        np.save(os.path.join(os.path.dirname(model_path), 'sklearn_input_scaler.npy'),
                [self.input_scaler.data_min_, self.input_scaler.data_max_])
        np.save(os.path.join(os.path.dirname(model_path), 'sklearn_output_scaler.npy'),
                [self.output_scaler.data_min_, self.output_scaler.data_max_])

    def load(self, model_path):
        """
            Load the model from disk.

            Args:
                model_path (str): Path to load the model from

        """
        #Load the model
        self.model = joblib.load(model_path)

        #Load scalers
        input_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'sklearn_input_scaler.npy'))
        output_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'sklearn_output_scaler.npy'))

        self.input_scaler.data_min_ = input_scaler_data[0]
        self.input_scaler.data_max_ = input_scaler_data[1]
        self.input_scaler.scale_ = 1.0 / (self.input_scaler.data_max_ - self.input_scaler.data_min_)

        self.output_scaler.data_min_ = output_scaler_data[0]
        self.output_scaler.data_max_ = output_scaler_data[1]
        self.output_scaler.scale_ = 1.0 / (self.output_scaler.data_max_ - self.output_scaler.data_min_)


if __name__ == "__main__":
    # Generate some random data to use in the models

    np.random.seed(42)

    X = np.random.rand(1000, 3)  # END-EFFECTOR POSITIONS (X, Y, Z)

    y = np.random.rand(1000, 4)  # JOINT ANGLES (Theta 0, Theta 1, Theta 02,Theta 3)

    # Split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train TF model

    tf_model = TensorFlowModel(input_dimension=3, output_dimension=4)

    tf_history = tf_model.train(X_train, y_train, epochs=1000, batch_size=64, verbose=2)

    # Evaluate models

    tf_metrics = tf_model.evaluate(X_test, y_test)




    print("\nTF Model Metrics: ")
    #print("MAE per Joint: ")
    #print(tf_metrics['mae_per_joint'])
    #print("\n")
    #print("Mean Euclidean Error: ")
    #print(tf_metrics['mean_euclidean_error'])
    print(f"MAE per joint: {tf_metrics['mae_per_joint']}")
    print(f"Overall MAE: {tf_metrics['mae_overall']:.4f}")
    print(f"Mean Euclidean Error: {tf_metrics['mean_euclidean_error']:.4f}")
    """
    print("\nPyTorch Model Metrics:")
    print(f"MAE per joint: {pt_metrics['mae_per_joint']}")
    print(f"Overall MAE: {pt_metrics['mae_overall']:.4f}")
    print(f"Mean Euclidean Error: {pt_metrics['mean_euclidean_error']:.4f}")

    print("\nscikit-learn Model Metrics:")
    print(f"MAE per joint: {sklearn_metrics['mae_per_joint']}")
    print(f"Overall MAE: {sklearn_metrics['mae_overall']:.4f}")
    print(f"Mean Euclidean Error: {sklearn_metrics['mean_euclidean_error']:.4f}")
    """