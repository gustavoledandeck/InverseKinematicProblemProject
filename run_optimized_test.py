"""
Optimized neural network model for inverse kinematics with improved accuracy.
This version focuses on achieving the 0.5mm accuracy target while maintaining
performance under 100ms on Raspberry Pi.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from utils.Forward_kinematics import ForwardKinematics
from utils.data_for_simulation import DataGenerator
from models.neural_network import TensorFlowModel, ScikitLearnModel, PyTorchModel

#Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

#Define results directory relative to the current script
results_dir = os.path.join(current_dir, "results")

#Define trained_models directory relative to the current script
trained_models_dir = os.path.join(current_dir, "trained_models")

class OptimizedInverseKinematics:
    """
        Optimized inverse kinematics implementation with focus on accuracy
    """

    def __init__(self, dof=3, framework='tensorflow', hidden_layers=None):
        """
        Initialize the optimized inverse kinematics model

        Args:
            dof (int): Degrees of freedom (3 or 4)
            framework (str): Neural network framework to use ('tensorflow' or 'sklearn')
            hidden_layers (list): Custom hidden layer configuration
        """
        self.dof = dof
        self.framework = framework
        self.fk = ForwardKinematics()

        #Set default hidden layers based on DOF if not provided
        if hidden_layers is None:
            if dof == 3:
                self.hidden_layers = [64, 128, 256, 128, 64]  # Deeper network for 3-DOF
            else:
                self.hidden_layers = [128, 256, 512, 256, 128]  # Even deeper for 4-DOF
        else:
            self.hidden_layers = hidden_layers

        #Initialize neural network model
        if dof == 3:
            input_dimension = 2
            output_dimension = 3
        else:
            input_dimension = 3
            output_dimension = 4

        if framework == 'tensorflow':
            self.model = TensorFlowModel(input_dimension=input_dimension, output_dimension=output_dimension,
                                         hidden_layers=self.hidden_layers)
        elif framework == 'sklearn':
            self.model = ScikitLearnModel(input_dimension=input_dimension, output_dimension=output_dimension,
                                         hidden_layers=self.hidden_layers)
        elif framework == 'pytorch':
            self.model = PyTorchModel(input_dimension=input_dimension, output_dimension=output_dimension,
                                         hidden_layers=self.hidden_layers)


        else:
            raise ValueError("Invalid framework. Choose 'tensorflow' or 'sklearn'")

        self.is_trained = False

    def generate_training_data(self, num_samples=10000, use_grid=True, grid_size=30):
        """
            Generate high-quality training data with grid-based sampling for better coverage

        Args:
            num_samples (int): Number of random samples to generate
            use_grid (bool): Whether to use grid-based sampling
            grid_size (int): Size of the grid for grid-based sampling

        Returns:
            tuple: (X, y) training data
        """
        data_gen = DataGenerator()

        if self.dof == 3:
            #For 3-DOF, use either grid-based or random sampling
            if use_grid:
                try:
                    X_grid, y_grid = data_gen.generate_grid_dataset_3dof(grid_size=grid_size)
                    if len(X_grid) > 0:
                        #If grid sampling worked, combine with random samples
                        X_random, y_random = data_gen.generate_dataset_3dof(num_samples)
                        X = np.vstack([X_grid, X_random])
                        y = np.vstack([y_grid, y_random])
                    else:
                        #Fallback to random sampling if grid is empty
                        print("Grid sampling returned empty dataset, falling back to random sampling")
                        X, y = data_gen.generate_dataset_3dof(num_samples)
                except Exception as e:
                    print(f"Error in grid sampling: {e}, falling back to random sampling")
                    X, y = data_gen.generate_dataset_3dof(num_samples)
            else:
                X, y = data_gen.generate_dataset_3dof(num_samples)
        else:
            X, y = data_gen.generate_dataset_4dof(num_samples)

        return X, y

    def train(self, X=None, y=None, num_samples=10000, epochs=100, batch_size=32, verbose=1):
        """
            Train the model with optimized parameters

            Args:
                X (numpy.ndarray): Input data (end-effector positions)
                y (numpy.ndarray): Target data (joint angles)
                num_samples (int): Number of samples to generate if X and y are not provided
                epochs (int): Number of training epochs
                batch_size (int): Batch size for training
                verbose (int): Verbosity level

            :return:
                tuple: (history, metrics) training history and evaluation metrics
        """
        #Generate data if not provided
        if X is None or y is None:
            X, y = self.generate_training_data(num_samples)

            # Add noise to inputs for robustness
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            X = np.vstack([X, X_noisy])
            y = np.vstack([y, y])

        #Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        print(f"Training optimized {self.dof}-DOF model using {self.framework}...")
        print(f"Network architecture: {self.hidden_layers}")

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]

        start_time = time.time()

        history = self.model.train(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbose,
            callbacks=callbacks
        )
        training_time = time.time() - start_time

        # Evaluate the model
        metrics = self.model.evaluate(X_test, y_test)

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"MAE per joint: {metrics['mae_per_joint']}")
        print(f"Overall MAE: {metrics['mae_overall']:.4f}")
        print(f"Mean Euclidean Error: {metrics['mean_euclidean_error']:.4f}")

        self.is_trained = True
        return history, metrics

    def predict(self, point):
        """
            Predict joint angles for a given end-effector position

            Args:
                point (list or numpy.ndarray): End-effector position

            :return:
                numpy.ndarray: Predicted joint angles
        """
        if not self.is_trained:
            raise ValueError("Model is not trained")

        # Ensure input is a 2D array
        if len(np.array(point).shape) == 1:
            point = np.array(point).reshape(1, -1)

        return self.model.predict(point)[0]

    def verify_accuracy(self, point, angles):
        """
            Verify the accuracy of the predicted joint angles

            Args:
                point (list or numpy.ndarray): Target end-effector position
                angles (list or numpy.ndarray): Predicted joint angles

            :return:
                float: Euclidean distance error in mm
        """
        if self.dof == 3:
            x, y = self.fk.forward_kinematics_3dof(angles)
            error = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        else:
            x, y, z = self.fk.forward_kinematics_4dof(angles)
            error = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2 + (point[2] - z) ** 2)

        return error

    def save_model(self, model_path):
        """
            Save the model with appropriate extension

            Args:
                model_path (str): Path to save the model

            :return:
                str: Path where the model was saved
        """
        if not self.is_trained:
            raise ValueError("Model is not trained")

        #Add appropriate extension based on framework
        if self.framework == 'tensorflow':
            if not (model_path.endswith('.keras') or model_path.endswith('.h5')):
                model_path = f"{model_path}.keras"
        elif self.framework == 'pytorch':
            if not model_path.endswith('.pt'):
                model_path = f"{model_path}.pt"
        elif self.framework == 'sklearn':
            if not model_path.endswith('.joblib'):
                model_path = f"{model_path}.joblib"

        #Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path):
        """
            Load a trained model

            Args:
                model_path (str): Path to the saved model
        """
        self.model.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")

    def visualize_prediction(self, point, angles=None):
        """
            Visualize the predicted joint configuration

            Args:
                point (list or numpy.ndarray): Target end-effector position
                angles (list or numpy.ndarray): Predicted joint angles (if None, will be predicted)

            :return:
                matplotlib.figure.Figure: Visualization figure
        """
        if angles is None:
            angles = self.predict(point)

        if self.dof == 3:
            x, y = self.fk.forward_kinematics_3dof(angles)
            fig = self.fk.visualize_arm_3dof(angles)
            ax = fig.axes[0]
            ax.scatter(point[0], point[1], color='red', marker='x', s=100, label='Target')
            ax.scatter(x, y, color='green', marker='o', s=100, label='Predicted')
        else:
            x, y, z = self.fk.forward_kinematics_4dof(angles)
            fig = self.fk.visualize_arm_4dof(angles)
            ax = fig.axes[0]
            ax.scatter(point[0], point[1], point[2], color='red', marker='x', s=100, label='Target')
            ax.scatter(x, y, z, color='green', marker='o', s=100, label='Predicted')

        error = self.verify_accuracy(point, angles)
        ax.set_title(f'{self.dof}-DOF Arm Configuration\nError: {error:.2f} mm')
        ax.legend()

        return fig


def run_optimized_test(framework='tensorflow', dof=3, num_samples=5000, epochs=50):
    """
        Run an optimized test with focus on achieving the 0.5mm accuracy target

        Args:
            framework (str): Neural network framework to use ('tensorflow' or 'sklearn')
            dof (int): Degrees of freedom (3 or 4)
            num_samples (int): Number of training samples
            epochs (int): Number of training epochs

        :return:
            dict: Performance metrics
    """
    print(f"\n=== Running optimized test for {dof}-DOF using {framework} ===\n")

    #Create and train model
    model = OptimizedInverseKinematics(dof=dof, framework=framework)

    #Generate training data
    X, y = model.generate_training_data(num_samples=num_samples)

    #Train model
    start_time = time.time()
    history, _ = model.train(X, y, epochs=epochs, batch_size=64, verbose=1)
    training_time = time.time() - start_time

    #Generate test points
    fk = ForwardKinematics()
    test_points = []

    for _ in range(10):  # 10 test points
        if dof == 4:
            theta0 = np.random.uniform(-np.pi, np.pi)
            theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta3 = np.random.uniform(-np.pi / 2, np.pi / 2)
            x, y, z = fk.forward_kinematics_4dof([theta0, theta1, theta2, theta3])
            test_points.append([x, y, z])
        else:
            theta0 = np.random.uniform(-np.pi, np.pi)
            theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)
            x, y = fk.forward_kinematics_3dof([theta0, theta1, theta2])
            test_points.append([x, y])

    #Test model
    errors = []
    inference_times = []

    for point in test_points:
        #Measure inference time
        start_time = time.time()
        predicted_angles = model.predict(point)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)

        #Calculate error
        error = model.verify_accuracy(point, predicted_angles)
        errors.append(error)

    #Calculate statistics
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    accuracy_percentage = np.mean([error < 0.5 for error in errors]) * 100
    mean_inference_time = np.mean(inference_times)

    #Save model
    model_path = f'{trained_models_dir}/optimized_model_{dof}dof_{framework}'
    saved_path = model.save_model(model_path)

    #Visualize one test case
    fig = model.visualize_prediction(test_points[0])
    plt.savefig(f'{results_dir}/optimized_{dof}dof_{framework}_test_case.png')
    plt.close(fig)

    # Print results
    print(f"\nResults for optimized {dof}-DOF {framework} model:")
    print(f"  Training time: {training_time:.2f} s")
    print(f"  Mean inference time: {mean_inference_time:.2f} ms")
    print(f"  Mean error: {mean_error:.4f} mm")
    print(f"  Max error: {max_error:.4f} mm")
    print(f"  Accuracy < 0.5mm: {accuracy_percentage:.2f}%")

    return {
        'DOF': dof,
        'Framework': framework,
        'Training Time (s)': training_time,
        'Inference Time (ms)': mean_inference_time,
        'Mean Error (mm)': mean_error,
        'Max Error (mm)': max_error,
        'Accuracy < 0.5mm (%)': accuracy_percentage
    }


if __name__ == "__main__":

    #Run optimized tests
    results = []

    #Test 3-DOF models with optimized parameters
    for framework in ['tensorflow', 'sklearn', 'pytorch']:
        result = run_optimized_test(framework=framework, dof=3, num_samples=5000, epochs=50)
        results.append(result)

    #Test 4-DOF models with optimized parameters
    for framework in ['tensorflow', 'sklearn', 'pytorch']:
        result = run_optimized_test(framework=framework, dof=4, num_samples=5000, epochs=50)
        results.append(result)

    #Create results dataframe
    results_df = pd.DataFrame(results)

    #Save results to CSV
    results_df.to_csv(f'{results_dir}/optimized_results.csv', index=False)

    #Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimized Inverse Kinematics Neural Network Performance', fontsize=16)

    #Filter results by DOF
    results_3dof = results_df[results_df['DOF'] == 3]
    results_4dof = results_df[results_df['DOF'] == 4]

    #Plot inference time
    axes[0, 0].bar(results_3dof['Framework'], results_3dof['Inference Time (ms)'], label='3-DOF')
    axes[0, 0].bar(results_4dof['Framework'], results_4dof['Inference Time (ms)'], alpha=0.7, label='4-DOF')
    axes[0, 0].set_title('Inference Time (ms)')
    axes[0, 0].set_ylabel('Milliseconds')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', label='Target (100ms)')
    axes[0, 0].legend()

    #Plot mean error
    axes[0, 1].bar(results_3dof['Framework'], results_3dof['Mean Error (mm)'], label='3-DOF')
    axes[0, 1].bar(results_4dof['Framework'], results_4dof['Mean Error (mm)'], alpha=0.7, label='4-DOF')
    axes[0, 1].set_title('Mean Error (mm)')
    axes[0, 1].set_ylabel('Millimeters')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5mm)')
    axes[0, 1].legend()

    #Plot accuracy percentage
    axes[1, 0].bar(results_3dof['Framework'], results_3dof['Accuracy < 0.5mm (%)'], label='3-DOF')
    axes[1, 0].bar(results_4dof['Framework'], results_4dof['Accuracy < 0.5mm (%)'], alpha=0.7, label='4-DOF')
    axes[1, 0].set_title('Accuracy < 0.5mm (%)')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend()

    #Plot training time
    axes[1, 1].bar(results_3dof['Framework'], results_3dof['Training Time (s)'], label='3-DOF')
    axes[1, 1].bar(results_4dof['Framework'], results_4dof['Training Time (s)'], alpha=0.7, label='4-DOF')
    axes[1, 1].set_title('Training Time (s)')
    axes[1, 1].set_ylabel('Seconds')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{results_dir}/optimized_performance_summary.png')

    print("\nOptimized testing completed. Results saved to results")
