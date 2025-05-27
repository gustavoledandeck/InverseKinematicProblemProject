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
from scipy.optimize import lbfgsb
# This is the sketch of tensorflowmodel implementation version one

class TensorFlowModel:
    """
    NNA model for IK with tensorflow
    """

    def __init__(self, input_dimension, output_dimension, hidden_layers=(128, 64, 32), activation='relu'):
        """
            This is the constructor of TF model.
            Initialize the tensorflow model.

            Args:
                input_dimension (int): Input dimension (end-effector position)
                output_dimension (int): Output dimension (joint angles)
                hidden_layers (tuple) : List of neurons in each hidden layer
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
        #initializer = tf.keras.initializers.RandomUniform()
        #initializer = keras.initializers.RandomNormal()
        model = keras.Sequential()


        #input layer

        model.add(keras.layers.Input(shape=(self.input_dimension,)))


        #Hidden layers
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.L1L2(l1=0.00001, l2=0.0001),
                bias_regularizer=keras.regularizers.L2(0.0001),
                activity_regularizer=keras.regularizers.L2(0.00001)
                #,
                #kernel_initializer=initializer,
                #bias_initializer=initializer
            ))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.3))

            # Output layer

        model.add(keras.layers.Dense(self.output_dimension, activation=self.activation))



        #compile
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.0001,
                    decay_steps=10000,
                    decay_rate=0.9
                ),
                epsilon=10**(-12)

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
            callbacks_to_use = default_callbacks
        elif isinstance(callbacks, list):
            callbacks_to_use = default_callbacks + callbacks
        else:
            callbacks_to_use = callbacks

        #Train the model
        start_time = time.time()
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks_to_use
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

        print(f"TensorFlow model inference for {X.shape[0]} samples: {inference_time * 1000:.2f} ms")
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
        euclidean_error_joint_space = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        mean_euclidean_error_joint_space = np.mean(euclidean_error_joint_space)
        max_euclidean_error_joint_space = np.max(euclidean_error_joint_space)

        return {
            'mae_per_joint': mae_per_joint,
            'mae_overall': mae_overall,
            'mean_euclidean_error_joint_space': mean_euclidean_error_joint_space,
            'max_euclidean_error_joint_space': max_euclidean_error_joint_space
        }

    def save(self, model_path):
        """
            Save the model to disk.

        Args: model_path (str) : Path to save the model


        """

        #Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

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



        #self.input_scaler.data_min_ = input_scaler_data[0]
        #self.input_scaler.data_max_ = input_scaler_data[1]
        #self.input_scaler.scale_ = 1.0 / (self.input_scaler.data_max_ - self.input_scaler.data_min_)

        #self.output_scaler.data_min_ = output_scaler_data[0]
        #self.output_scaler.data_max_ = output_scaler_data[1]
        #self.output_scaler.scale_ = 1.0 / (self.output_scaler.data_max_ - self.output_scaler.data_min_)

class PyTorchModel:

    """
        NNA model for IK with PyTorch.
    """

    def __init__(self, input_dimension, output_dimension,
                 hidden_layers=(128, 64, 32),
                 activation='relu',
                 learning_rate=0.0001,
                 weight_decay=1e-4):
        """
            Initialize the PyTorch model.

        Args:
            input_dim (int): Input dimension (end-effector position)
            output_dim (int): Output dimension (joint angles)
            hidden_layers (tuple): Tuple of neurons in each hidden layer
            activation (str): Activation function to use
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2 Regularization

        # Set activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        else:
            self.activation_fn = nn.Sigmoid()


        # Create input and output scalers
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        # Build the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)





        self.criterion = nn.MSELoss()


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=10**(-12)
        )

    def _build_model(self):
        """
        Build the PyTorch neural network model

        :return:
            torch.nn.Module: PyTorch model
        """

        layers = []
        current_dim = self.input_dimension
        for units in self.hidden_layers:
            layers.append(nn.Linear(current_dim, units))
            layers.append(self.activation_fn)
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(0.3))
            current_dim = units
        """
        # Input layer
        layers.append(nn.Linear(self.input_dimension, self.hidden_layers[0]))
        layers.append(self.activation_fn)
        layers.append(nn.BatchNorm1d(self.hidden_layers[0]))


        # Hidden layers
        for i in range(len(self.hidden_layers) - 1):
            layers.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            layers.append(self.activation)
            layers.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))

        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_dimension))

        return nn.Sequential(*layers)
        """
        # Output layer - Linear activation for regression
        layers.append(nn.Linear(current_dim, self.output_dimension))

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
        start_time = time.time()
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

        for epoch in range(epochs):
            #Training
            self.model.train()
            train_loss = 0.0
            """
            for inputs, targets in train_loader:
                #Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                #Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            """


            for inputs, targets in train_loader:
                self.optimizer.zero_grad()  # Zero gradients

                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Compute loss
                loss.backward() # Backward pass

                self.optimizer.step()  # Update weights

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
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
        start_time = time.time()
        with torch.no_grad():
            #y_scaled_pred = self.model(X_tensor).cpu().numpy()
            y_scaled_pred_tensor = self.model(X_tensor)
        inference_time = time.time() - start_time

        #Inverse transform to get original scale
        #y_pred = self.output_scaler.inverse_transform(y_scaled_pred)
        y_scaled_pred = y_scaled_pred_tensor.cpu().numpy()
        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)
        print(f"PyTorch model inference for {X.shape[0]} samples: {inference_time * 1000:.2f} ms")

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
        euclidean_error_joint_space = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        mean_euclidean_error_joint_space = np.mean(euclidean_error_joint_space)
        max_euclidean_error_joint_space = np.max(euclidean_error_joint_space)

        return {
            'mae_per_joint': mae_per_joint,
            'mae_overall': mae_overall,
            'mean_euclidean_error_joint_space': mean_euclidean_error_joint_space,
            'max_euclidean_error_joint_space': max_euclidean_error_joint_space
        }

    def save(self, model_path):
        """
             Save the model to disk.

            Args:
                model_path (str): Path to save the model

        """
        #Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        #Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dimension': self.input_dimension,
            'output_dimension': self.output_dimension,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation_fn.__class__.__name__,  # Store activation name
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }, model_path)
        print(f"PyTorch model saved to {model_path}")

        joblib.dump(self.input_scaler, os.path.join(model_dir, 'pt_input_scaler.joblib'))
        joblib.dump(self.output_scaler, os.path.join(model_dir, 'pt_output_scaler.joblib'))
        print(f"PyTorch scalers saved in {model_dir}")
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
        activation_name = checkpoint.get('activation', 'ReLU').lower()
        self.learning_rate = checkpoint.get('learning_rate', 0.001)
        self.weight_decay = checkpoint.get('weight_decay', 1e-4)

        if activation_name == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation_name == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation_name == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = nn.ReLU()

        #Rebuild the model
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"PyTorch model loaded from {model_path}")


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=10**(-12)
        )
        self.criterion = nn.MSELoss()
        #self.criterion = nn.BCELoss()
        """
        #Load scalers
        input_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'pt_input_scaler.npy'))
        output_scaler_data = np.load(os.path.join(os.path.dirname(model_path), 'pt_output_scaler.npy'))

        self.input_scaler.data_min_ = input_scaler_data[0]
        self.input_scaler.data_max_ = input_scaler_data[1]
        self.input_scaler.scale_ = 1.0 / (self.input_scaler.data_max_ - self.input_scaler.data_min_)

        self.output_scaler.data_min_ = output_scaler_data[0]
        self.output_scaler.data_max_ = output_scaler_data[1]
        self.output_scaler.scale_ = 1.0 / (self.output_scaler.data_max_ - self.output_scaler.data_min_)
        """
        model_dir = os.path.dirname(model_path)
        self.input_scaler = joblib.load(os.path.join(model_dir, 'pt_input_scaler.joblib'))
        self.output_scaler = joblib.load(os.path.join(model_dir, 'pt_output_scaler.joblib'))
        print(f"PyTorch scalers loaded from {model_dir}")

class ScikitLearnModel:
    """
        Neural network model for IK using scikit-learn.
    """

    def __init__(self, input_dimension, output_dimension,
                 hidden_layer_sizes=(128, 64),
                 activation='relu',
                 alpha=0.0001,
                 solver='lbfgs',
                 learning_rate_init=0.0001,
                 max_iter=1000, # Max iterations for solver
                 n_iter_no_change=30):
        """
        Initialize the scikit-learn model.

        Args:
            input_dimension (int): Input dimension (end-effector position)
            output_dimension (int): Output dimension (joint angles)
            hidden_layers (list): List of neurons in each hidden layer
            activation (str): Activation function to use
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension


        #Create input and output scalers
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        #Build the model
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='lbfgs',
            alpha=alpha,  # L2 penalty (regularization term)
            batch_size=64,
            learning_rate='adaptive',  # Learning rate schedule
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=True,
            random_state=42,  # For reproducibility
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,

            n_iter_no_change=n_iter_no_change,  # Number of iterations with no improvement to wait before stopping
            verbose=False  # Set to True or an int for verbosity during training
        )

    def train(self, X, y, epochs=None, batch_size=None, validation_split=None, verbose=1):
        """
            Train the model on the provided dataset.

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y (numpy.ndarray): Target data (joint angles)
                epochs (int): (controlled by max_iter)
                batch_size (int):  (controlled by batch_size)
                validation_split (float):  (controlled by validation_fraction)
                verbose (int): Verbosity level
        :return:
            dict: Training history (not available in scikit-learn)
        """
        start_time = time.time()
        #Fit the scalers
        X_scaled = self.input_scaler.fit_transform(X)
        # MLPRegressor expects a 1D array for y if there's only one target,
        # or a 2D array if multiple targets. Ensure y_scaled is correctly shaped.
        if self.output_dimension == 1 and len(y.shape) > 1 and y.shape[1] == 1:
            y_scaled = self.output_scaler.fit_transform(y).ravel()
        else:
            y_scaled = self.output_scaler.fit_transform(y)

        #y_scaled = self.output_scaler.fit_transform(y)

        #Train the model

        self.model.fit(X_scaled, y_scaled)
        training_time = time.time() - start_time

        #print(f"scikit-learn model training completed in {training_time:.2f} seconds")

        if verbose:
            print(f"Scikit-learn model training completed in {training_time:.2f} seconds.")
            print(f"Number of iterations: {self.model.n_iter_}")
            print(f"Final loss: {self.model.loss_}")

            # Scikit-learn's MLPRegressor doesn't return a history dict like Keras.
            # We can return some information if needed.
        return {'loss': self.model.loss_, 'n_iter_': self.model.n_iter_}

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

        # Reshape if y_scaled_pred is 1D (single output target) to make inverse_transform work as expected
        if len(y_scaled_pred.shape) == 1:
            y_scaled_pred = y_scaled_pred.reshape(-1, 1)

        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)
        # print(f"Scikit-learn model inference for {X.shape[0]} samples: {inference_time*1000:.2f} ms")
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
        euclidean_error_joint_space = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        mean_euclidean_error_joint_space = np.mean(euclidean_error_joint_space)
        max_euclidean_error_joint_space = np.max(euclidean_error_joint_space)

        return {
            'mae_per_joint': mae_per_joint,
            'mae_overall': mae_overall,
            'mean_euclidean_error_joint_space': mean_euclidean_error_joint_space,
            'max_euclidean_error_joint_space': max_euclidean_error_joint_space,
            'score': self.model.score(self.input_scaler.transform(X), self.output_scaler.transform(y_true))  # R^2 score
        }

    def save(self, model_path):
        """
            Save the model to disk.

            Args:
                model_path (str): Path to save the model
        :return:
        """
        #Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model, model_path)
        print(f"Scikit-learn model saved to {model_path}")

        # Save scalers
        joblib.dump(self.input_scaler, os.path.join(model_dir, 'sklearn_input_scaler.joblib'))
        joblib.dump(self.output_scaler, os.path.join(model_dir, 'sklearn_output_scaler.joblib'))
        print(f"Scikit-learn scalers saved in {model_dir}")

    def load(self, model_path):
        """
            Load the model from disk.

            Args:
                model_path (str): Path to load the model from

        """
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
        """
        self.model = joblib.load(model_path)
        print(f"Scikit-learn model loaded from {model_path}")

        model_dir = os.path.dirname(model_path)
        self.input_scaler = joblib.load(os.path.join(model_dir, 'sklearn_input_scaler.joblib'))
        self.output_scaler = joblib.load(os.path.join(model_dir, 'sklearn_output_scaler.joblib'))
        print(f"Scikit-learn scalers loaded from {model_dir}")


if __name__ == "__main__":
    # Generate some random data to use in the models

    print("Running basic tests for neural network models...")
    np.random.seed(42)  # For reproducibility

    # Generate some synthetic data for a 4-DOF arm (3D position input, 4 joint angles output)
    num_samples = 2000  # Increased samples for a slightly more meaningful test
    input_dim = 3  # (x, y, z)
    output_dim = 4  # (q1, q2, q3, q4)

    # Simulate end-effector positions and corresponding joint angles
    # In a real scenario, this data would come from your forward kinematics calculations
    # or a dataset of robot movements.
    X_data = np.random.rand(num_samples, input_dim) * 200 - 100  # Example range for positions
    y_data = np.random.rand(num_samples, output_dim) * np.pi - (np.pi / 2)  # Example range for joint angles (radians)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # --- Test TensorFlowModel ---
    print("\n--- Testing TensorFlowModel ---")
    try:
        tf_model = TensorFlowModel(input_dimension=input_dim, output_dimension=output_dim,
                                   hidden_layers=(64, 32))  # Using a smaller test architecture
        print("Training TensorFlow model...")
        tf_history = tf_model.train(X_train, y_train, epochs=50, batch_size=32,
                                    verbose=0)  # Reduced epochs for quick test
        print("Evaluating TensorFlow model...")
        tf_metrics = tf_model.evaluate(X_test, y_test)

        print("\nTensorFlow Model Metrics:")
        print(f"  MAE per joint (joint space): {tf_metrics['mae_per_joint']}")
        print(f"  Overall MAE (joint space): {tf_metrics['mae_overall']:.4f}")
        print(f"  Mean Euclidean Error (joint space): {tf_metrics['mean_euclidean_error_joint_space']:.4f}")
        # Example of saving and loading
        tf_model.save("trained_models/tf_test_model/model.keras")
        tf_model_loaded = TensorFlowModel(input_dimension=input_dim, output_dimension=output_dim)
        tf_model_loaded.load("trained_models/tf_test_model/model.keras")
        tf_metrics_loaded = tf_model_loaded.evaluate(X_test, y_test)
        print(f"  Overall MAE (loaded model): {tf_metrics_loaded['mae_overall']:.4f}")


    except Exception as e:
        print(f"Error during TensorFlowModel test: {e}")

    # --- Test PyTorchModel ---
    print("\n--- Testing PyTorchModel ---")
    try:
        pt_model = PyTorchModel(input_dimension=input_dim, output_dimension=output_dim, hidden_layers=(64, 32))
        print("Training PyTorch model...")
        pt_history = pt_model.train(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        print("Evaluating PyTorch model...")
        pt_metrics = pt_model.evaluate(X_test, y_test)

        print("\nPyTorch Model Metrics:")
        print(f"  MAE per joint (joint space): {pt_metrics['mae_per_joint']}")
        print(f"  Overall MAE (joint space): {pt_metrics['mae_overall']:.4f}")
        print(f"  Mean Euclidean Error (joint space): {pt_metrics['mean_euclidean_error_joint_space']:.4f}")
        pt_model.save("trained_models/pt_test_model/model.pth")
        pt_model_loaded = PyTorchModel(input_dimension=input_dim, output_dimension=output_dim)
        pt_model_loaded.load("trained_models/pt_test_model/model.pth")
        pt_metrics_loaded = pt_model_loaded.evaluate(X_test, y_test)
        print(f"  Overall MAE (loaded model): {pt_metrics_loaded['mae_overall']:.4f}")

    except Exception as e:
        print(f"Error during PyTorchModel test: {e}")

    # --- Test ScikitLearnModel ---
    print("\n--- Testing ScikitLearnModel ---")
    try:
        sklearn_model = ScikitLearnModel(input_dimension=input_dim, output_dimension=output_dim,
                                         hidden_layer_sizes=(64, 32), n_iter_no_change=10)
        print("Training ScikitLearn model...")
        sklearn_history = sklearn_model.train(X_train, y_train, verbose=0)
        print("Evaluating ScikitLearn model...")
        sklearn_metrics = sklearn_model.evaluate(X_test, y_test)

        print("\nScikit-learn Model Metrics:")
        print(f"  MAE per joint (joint space): {sklearn_metrics['mae_per_joint']}")
        print(f"  Overall MAE (joint space): {sklearn_metrics['mae_overall']:.4f}")
        print(f"  Mean Euclidean Error (joint space): {sklearn_metrics['mean_euclidean_error_joint_space']:.4f}")
        print(f"  R^2 Score: {sklearn_metrics['score']:.4f}")  # R^2 score is a good indicator for regression
        sklearn_model.save("trained_models/sklearn_test_model/model.joblib")
        sklearn_model_loaded = ScikitLearnModel(input_dimension=input_dim, output_dimension=output_dim)
        sklearn_model_loaded.load("trained_models/sklearn_test_model/model.joblib")
        sklearn_metrics_loaded = sklearn_model_loaded.evaluate(X_test, y_test)
        print(f"  Overall MAE (loaded model): {sklearn_metrics_loaded['mae_overall']:.4f}")


    except Exception as e:
        print(f"Error during ScikitLearnModel test: {e}")

    print("\nBasic tests completed.")