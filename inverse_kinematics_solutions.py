"""
Implementation of 4-DOF and 3-DOF Inverse Kinematics Solutions
This module implements the specific solutions for 4-DOF and 3-DOF using ANN
for IK problem.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

from utils.Forward_kinematics import ForwardKinematics
from models.neural_network import TensorFlowModel
from utils.Forward_kinematics import ForwardKinematics
from utils.data_for_simulation import DataGenerator
from models.neural_network import TensorFlowModel, PyTorchModel, ScikitLearnModel


class InverseKinematics4DOF:
    """
        Implementation of IK for a 4-DOF robotic arm (R³ space).
    """

    def __init__(self, link_lengths=None, model_type='tensorflow'):
        """
            Initialize the 4-DOF inverse kinematics solver.

            Args:
                    link_lengths (list) : List of link lengths [lo, l1, l2, l3] in mm
                                            Default is [50, 100, 100, 100] for a typical small robotic arm
                                            model_type (str) : Type of neural model to use ('tensorflow', 'pytorch', 'sklearn')

        """

        self.link_lengths = link_lengths if link_lengths is not None else [50, 100, 100, 100]
        self.model_type = model_type

        #Initialize forward kinematics and data generator
        self.fk = ForwardKinematics(link_lengths=self.link_lengths)
        self.data_gen = DataGenerator(link_lengths=self.link_lengths)

        #Initialize neural netwrok model
        if model_type == 'tensorflow':
            self.model = TensorFlowModel(input_dimension=3, output_dimension=4)
        elif model_type == 'pytorch':
            self.model = PyTorchModel(input_dimension=3, output_dimension=4)
        elif model_type == 'sklearn':
            self.model = ScikitLearnModel(input_dimension=3, output_dimension=4)

        else:
            raise ValueError("Invalid model type")

        self.is_trained = False


    def generate_training_data(self, num_samples=1000):
        """
            Generate training data for the 4-DOF IK model.

            Args:
                num_samples (int) : Number of samples to generate

        :return:
            tuple: (X, y) where X is end-effector positions and y is joint angles

        """
        print(f"Generating {num_samples} training samples for 4-DOF model ...")
        return self.data_gen.generate_dataset_4dof(num_samples)

    def train(self, X=None, y=None, num_samples=1000, epochs=100, batch_size=32, validation_split=0.2,verbose=1):
        """
            Train the IK model.

             Args:
            X (numpy.ndarray): Input data (end-effector positions)
            y (numpy.ndarray): Target data (joint angles)
            num_samples (int): Number of samples to generate if X and y are not provided
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            verbose (int): Verbosity level
        :return:
            dict: Training history
        """
        #Generate data if not provided
        if X is None or y is None:
            X, y = self.generate_training_data(num_samples)

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training 4-DOF inverse kinematics model using {self.model_type}...")
        history = self.model.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                   validation_split=validation_split, verbose=verbose)

        #Evaluate the model
        metrics = self.model.evaluate(X_test, y_test)
        print("\nModel Evaluation:")
        print(f"MAE per joint: {metrics['mae_per_joint']}")
        print(f"Overall MAE: {metrics['mae_overall']:.4f}")
        print(f"Mean Euclidean Error: {metrics['mean_euclidean_error']:.4f}")

        self.is_trained = True

        return history, metrics


    def predict(self, end_effector_position):
        """
            Predict joint angles for a given end-effector position

            Args:
            end_effector_position (list or numpy.ndarray): End-effector position [x, y, z]

        :return:
            numpy.ndarray: Predicted joint angles [theta0, theta1, theta2, theta3]
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

            #Ensure input is a 2D array
        if len(np.array(end_effector_position).shape) == 1:
            end_effector_position = np.array(end_effector_position).reshape(1, -1)

            #Predict joint angles
        joint_angles = self.model.predict(end_effector_position)

        return joint_angles[0]

    def verify_accuracy(self, end_effector_position, predicted_angles):

        """
            Verify the accuracy of the predicted joint angles by calculating
            the FK and comparing with the target end-effector position.

            Args:
            end_effector_position (list or numpy.ndarray): Target end-effector position [x, y, z]
            predicted_angles (list or numpy.ndarray): Predicted joint angles [θ0, θ1, θ2, θ3]

        :return:
            float: Euclidean distance error in mm
        """

        #Calculate forward kinematics for the predicted angles
        x, y, z = self.fk.forward_kinematics_4dof(predicted_angles)

        #Calculate Euclidean distance error
        error = np.sqrt((end_effector_position[0] - x) ** 2 +
                        (end_effector_position[1] - y) ** 2 +
                        (end_effector_position[2] - z) ** 2)

        return error

    def save_model(self, model_path):
        """
            Save the trained model to disk.

            Args:
                model_path (str) : Path to save the model

        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
            Load a trained model from disk.

            Args:
                model_path (str) : Path to load the model from

        """
        self.model.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")

    def visualize_prediction(self, end_effector_position, predicted_angles):
        """
            Visualize the predicted joint configuration.
            Args:
            end_effector_position (list or numpy.ndarray): Target end-effector position [x, y, z]
            predicted_angles (list or numpy.ndarray): Predicted joint angles [θ0, θ1, θ2, θ3]

        :return:
            matplotlib.figure: Figure object with the visualization
        """
        #Calculate forward kinematics for the predicted angles
        x, y, z = self.fk.forward_kinematics_4dof(predicted_angles)

        #Calculate error
        error = self.verify_accuracy(end_effector_position, predicted_angles)

        #Visualize the arm configuration
        fig = self.fk.visualize_arm_4dof(predicted_angles)

        #Add target position
        ax = fig.axes[0]
        ax.scatter(end_effector_position[0], end_effector_position[1], end_effector_position[2],
                   color='red', marker='x', s=100, label='Target')

        #Add predicted position
        ax.scatter(x, y, z, color='green', marker='o', s=100, label='Predicted')

        #Update title with error information
        ax.set_title(f'4-DOF Arm Configuration\nError: {error:.2f} mm')
        ax.legend()

        return fig

class InverseKinematics3DOF:
    """
        Implementation of IK for a 3-DOF robotic arm (R² SPACE).
    """

    def __init__(self, link_lengths=None, model_type='tensorflow'):
        """
            Initialize the 3-DOF IK solver.

            Args:
            link_lengths (list): List of link lengths [l0, l1, l2] in mm
                                Default is [50, 100, 100] for a typical small robotic arm
            model_type (str): Type of neural network model to use ('tensorflow', 'pytorch', or 'sklearn')

        """
        self.link_lengths = link_lengths if link_lengths is not None else [50, 100, 100]
        self.model_type = model_type

        # Initialize forward kinematics and data generator
        self.fk = ForwardKinematics(link_lengths=self.link_lengths)
        self.data_gen = DataGenerator(link_lengths=self.link_lengths)

        # Initialize neural network model
        if model_type == 'tensorflow':
            self.model = TensorFlowModel(input_dimension=2, output_dimension=3)
        elif model_type == 'pytorch':
            self.model = PyTorchModel(input_dimension=2, output_dimension=3)
        elif model_type == 'sklearn':
            self.model = ScikitLearnModel(input_dimension=2, output_dimension=3)
        else:
            raise ValueError("Invalid model type. Choose 'tensorflow', 'pytorch', or 'sklearn'.")

        self.is_trained = False

    def generate_training_data(self, num_samples=10000, use_grid=False, grid_size=20):
        """
            Generate training data for the 3-DOF IK model.

            rgs:
            num_samples (int): Number of samples to generate
            use_grid (bool): Whether to use grid-based sampling
            grid_size (int): Size of the grid if use_grid is True

        :return:
            tuple: (X, y) where X is end-effector positions and y is joint angles

        """
        if use_grid:
            print(f"Generating grid-based training data with grid size {grid_size}...")
            return self.data_gen.generate_grid_dataset_3dof(grid_size)
        else:
            print(f"Generating {num_samples} random training samples for 3-DOF model...")
            return self.data_gen.generate_dataset_3dof(num_samples)

    def train(self, X=None, y=None, num_samples=10000, use_grid=False, grid_size=20,
              epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
            Train the IK model.
            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y (numpy.ndarray): Target data (joint angles)
                num_samples (int): Number of samples to generate if X and y are not provided
                use_grid (bool): Whether to use grid-based sampling
                grid_size (int): Size of the grid if use_grid is True
                epochs (int): Number of training epochs
                batch_size (int): Batch size for training
                validation_split (float): Fraction of data to use for validation
                verbose (int): Verbosity level
        :return:
            dict: Training history
        """
        #Generate data if not provided
        if X is None or y is None:
            X, y = self.generate_training_data(num_samples, use_grid, grid_size)

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training 3-DOF inverse kinematics model using {self.model_type}...")
        history = self.model.train(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                   validation_split=validation_split, verbose=verbose)

        #Evaluate the model
        metrics = self.model.evaluate(X_test, y_test)
        print("\nModel Evaluation:")
        print(f"MAE per joint: {metrics['mae_per_joint']}")
        print(f"Overall MAE: {metrics['mae_overall']:.4f}")
        print(f"Mean Euclidean Error: {metrics['mean_euclidean_error']:.4f}")

        self.is_trained = True
        return history, metrics

    def predict(self, end_effector_position):
            """
                Predict joint angles for a given end-effector position.
                Args:
                    end_effector_position (list or numpy.ndarray): End-effector position [x, y]

            :return:
                numpy.ndarray: Predicted joint angles [theta0, theta1, theta2]
            """
            if not self.is_trained:
                raise ValueError("Model is not trained. Call train() first.")

            #Ensure input is a 2D array
            if len(np.array(end_effector_position).shape) == 1:
                end_effector_position = np.array(end_effector_position).reshape(1, -1)

            #Predict joint angles
            joint_angles = self.model.predict(end_effector_position)

            return joint_angles[0]

    def verify_accuracy(self, end_effector_position, predicted_angles):
        """
            Verify the accuracy of the predicted joint angles by calculating
            the FK and comparing with the target end-effector position.
            Args:
                end_effector_position (list or numpy.ndarray): Target end-effector position [x, y]
                predicted_angles (list or numpy.ndarray): Predicted joint angles [θ0, θ1, θ2]

        :return:
            float: Euclidean distance error in mm
        """
        #Calculate forward kinematics for the predicted angles
        x, y = self.fk.forward_kinematics_3dof(predicted_angles)

        #Calculate Euclidean distance error
        error = np.sqrt((end_effector_position[0] - x) ** 2 +
                        (end_effector_position[1] - y) ** 2)

        return error

    def save_model(self, model_path):

        """
            Save the trained model to disk.
            Args:
                model_path (str): Path to save the model

        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
            Load a trained model from disk.
            Args:
                model_path (str): Path to load the model from

        """
        self.model.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")

    def visualize_prediction(self, end_effector_position, predicted_angles):
        """
            Visualize the predicted joint configuration.
            Args:
                end_effector_position (list or numpy.ndarray): Target end-effector position [x, y]
                predicted_angles (list or numpy.ndarray): Predicted joint angles [θ0, θ1, θ2]

        :return:
            matplotlib.figure: Figure object with the visualization
        """
        #Calculate forward kinematics for the predicted angles
        x, y = self.fk.forward_kinematics_3dof(predicted_angles)

        #Calculate error
        error = self.verify_accuracy(end_effector_position, predicted_angles)

        # Visualize the arm configuration
        fig = self.fk.visualize_arm_3dof(predicted_angles)

        #Add target position
        ax = fig.axes[0]
        ax.scatter(end_effector_position[0], end_effector_position[1],
                   color='red', marker='x', s=100, label='Target')

        #Add predicted position
        ax.scatter(x, y, color='green', marker='o', s=100, label='Predicted')

        #Update title with error information
        ax.set_title(f'3-DOF Arm Configuration\nError: {error:.2f} mm')
        ax.legend()

        return fig

if __name__ == "__main__":
    #Test 4-DOF inverse kinematics
    ik_4dof = InverseKinematics4DOF(model_type='tensorflow')

    #Generate small dataset for testing
    X_4dof, y_4dof = ik_4dof.generate_training_data(num_samples=1000)

    #Train the model with a small number of epochs for testing
    history_4dof, metrics_4dof = ik_4dof.train(X_4dof, y_4dof, epochs=10, verbose=1)

    #Test prediction
    target_position_4dof = [150, 100, 50]  # Example target position
    predicted_angles_4dof = ik_4dof.predict(target_position_4dof)

    print("\n4-DOF Prediction:")
    print(f"Target position: {target_position_4dof}")
    print(f"Predicted angles: {predicted_angles_4dof}")

    #Verify accuracy
    error_4dof = ik_4dof.verify_accuracy(target_position_4dof, predicted_angles_4dof)
    print(f"Prediction error: {error_4dof:.2f} mm")

    #Visualize prediction
    fig_4dof = ik_4dof.visualize_prediction(target_position_4dof, predicted_angles_4dof)

    #Test 3-DOF inverse kinematics
    ik_3dof = InverseKinematics3DOF(model_type='tensorflow')

    #Generate small dataset for testing
    X_3dof, y_3dof = ik_3dof.generate_training_data(num_samples=1000)

    #Train the model with a small number of epochs for testing
    history_3dof, metrics_3dof = ik_3dof.train(X_3dof, y_3dof, epochs=10, verbose=1)

    #Test prediction
    target_position_3dof = [150, 100]  # Example target position
    predicted_angles_3dof = ik_3dof.predict(target_position_3dof)

    print("\n3-DOF Prediction:")
    print(f"Target position: {target_position_3dof}")
    print(f"Predicted angles: {predicted_angles_3dof}")

    #Verify accuracy
    error_3dof = ik_3dof.verify_accuracy(target_position_3dof, predicted_angles_3dof)
    print(f"Prediction error: {error_3dof:.2f} mm")

    #Visualize prediction
    fig_3dof = ik_3dof.visualize_prediction(target_position_3dof, predicted_angles_3dof)

    plt.show()
