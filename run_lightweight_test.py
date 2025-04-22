"""
Modified run_lightweight_test.py to handle model saving properly
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from utils.Forward_kinematics import ForwardKinematics
from utils.data_for_simulation import DataGenerator
from models.neural_network import TensorFlowModel, PyTorchModel, ScikitLearnModel


class SimpleInverseKinematics:
    """
        Simplified wrapper for inverse kinematics testing
    """

    def __init__(self, dof=3, framework='tensorflow'):
        """
            Initialize the inverse kinematics model

            Args:
                dof (int): Degrees of freedom (3 or 4)
                framework (str): Neural network framework to use
        """
        self.dof = dof
        self.framework = framework
        self.fk = ForwardKinematics()

        #Initialize neural network model
        if dof == 3:
            input_dimension = 2
            output_dimension = 3
        else:
            input_dimension = 3
            output_dimension = 4

        if framework == 'tensorflow':
            self.model = TensorFlowModel(input_dimension=input_dimension, output_dimension=output_dimension)
        elif framework == 'pytorch':
            self.model = PyTorchModel(input_dimension=input_dimension, output_dimension=output_dimension)
        elif framework == 'sklearn':
            self.model = ScikitLearnModel(input_dimension=input_dimension, output_dimension=output_dimension)

        else:
            raise ValueError("Invalid framework. Choose 'tensorflow', 'pytorch', or 'sklearn'.")

        self.is_trained = False

    def train(self, X, y, epochs=10, batch_size=32, verbose=1):
        """
            Train the model
            X (numpy.ndarray): Input data (end-effector positions)
            y (numpy.ndarray): Target data (joint angles)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        :return:
            dict: history
        """
        history = self.model.train(X, y, epochs=epochs, batch_size=batch_size,
                                   validation_split=0.2, verbose=verbose)
        self.is_trained = True
        return history

    def predict(self, point):
        """
            Predict joint angles for a given end-effector position
            Args:
                    point
        """
        if not self.is_trained:
            raise ValueError("Model is not trained")

        #Ensure input is a 2D array
        if len(np.array(point).shape) == 1:
            point = np.array(point).reshape(1, -1)

        return self.model.predict(point)[0]

    def verify_accuracy(self, point, angles):
        """
            Verify the accuracy of the predicted joint angles
            Args:
                    point
                    angles : joint angle in radian
        """
        if self.dof == 3:
            x, y = self.fk.forward_kinematics_3dof(angles)
            error = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        else:
            x, y, z = self.fk.forward_kinematics_4dof(angles)
            error = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2 + (point[2] - z) ** 2)

        return error

    def visualize_prediction(self, point, angles):
        """
            Visualize the predicted joint configuration
            Args:
                    point
                    angles : joint angle in radian
        """
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

    def save_model(self, model_path):
        """
            Save the model with appropriate extension

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

        #Create directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path


def run_lightweight_test(framework='tensorflow', dof=3, num_samples=1000, epochs=10):
    """
        Run a lightweight test of the inverse kinematics solution suitable for Raspberry Pi.

        Args:
            framework (str): Neural network framework to use ('tensorflow', 'pytorch', or 'sklearn')
            dof (int): Degrees of freedom (3 or 4)
            num_samples (int): Number of training samples
            epochs (int): Number of training epochs

    Returns:
        dict: Performance metrics
    """
    print(f"Running lightweight test for {dof}-DOF using {framework}...")

    #Create data generator
    data_gen = DataGenerator()

    #Generate training data
    if dof == 4:
        X, y = data_gen.generate_dataset_4dof(num_samples)
    else:  # dof == 3
        X, y = data_gen.generate_dataset_3dof(num_samples)

    #Create and train model
    model = SimpleInverseKinematics(dof=dof, framework=framework)

    #Train model with reduced parameters
    start_time = time.time()
    model.train(X, y, epochs=epochs, batch_size=32, verbose=1)
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
        else:  # dof == 3
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

    #Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #Define results directory relative to the current script
    results_dir = os.path.join(current_dir, "results")

    #Define trained_models directory relative to the current script
    trained_models_dir = os.path.join(current_dir, "trained_models")

    #Save model
    model_path = f'{trained_models_dir}/model_{dof}dof_{framework}_lightweight'
    saved_path = model.save_model(model_path)




    #Visualize one test case
    fig = model.visualize_prediction(test_points[0], model.predict(test_points[0]))
    plt.savefig(f'{results_dir}/{dof}dof_{framework}_test_case.png')
    plt.close(fig)

    #Print results
    print(f"\nResults for {dof}-DOF {framework} model:")
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
    #Run lightweight tests for all frameworks and DOFs
    results = []

    #Test 3-DOF models
    for framework in ['tensorflow', 'sklearn', 'pytorch']:
        result = run_lightweight_test(framework=framework, dof=3)
        results.append(result)

    #Test 4-DOF models
    for framework in ['tensorflow', 'sklearn', 'pytorch']:
        result = run_lightweight_test(framework=framework, dof=4)
        results.append(result)

    #Create results dataframe
    results_df = pd.DataFrame(results)

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define results directory relative to the current script
    results_dir = os.path.join(current_dir, "results")

    #Save results to CSV
    results_df.to_csv(f'{results_dir}/lightweight_results.csv', index=False)

    #Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inverse Kinematics Neural Network Performance', fontsize=16)

    # Filter results by DOF
    results_3dof = results_df[results_df['DOF'] == 3]
    results_4dof = results_df[results_df['DOF'] == 4]

    # Plot inference time
    axes[0, 0].bar(results_3dof['Framework'], results_3dof['Inference Time (ms)'], label='3-DOF')
    axes[0, 0].bar(results_4dof['Framework'], results_4dof['Inference Time (ms)'], alpha=0.7, label='4-DOF')
    axes[0, 0].set_title('Inference Time (ms)')
    axes[0, 0].set_ylabel('Milliseconds')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', label='Target (100ms)')
    axes[0, 0].legend()

    # Plot mean error
    axes[0, 1].bar(results_3dof['Framework'], results_3dof['Mean Error (mm)'], label='3-DOF')
    axes[0, 1].bar(results_4dof['Framework'], results_4dof['Mean Error (mm)'], alpha=0.7, label='4-DOF')
    axes[0, 1].set_title('Mean Error (mm)')
    axes[0, 1].set_ylabel('Millimeters')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Target (0.5mm)')
    axes[0, 1].legend()

    # Plot accuracy percentage
    axes[1, 0].bar(results_3dof['Framework'], results_3dof['Accuracy < 0.5mm (%)'], label='3-DOF')
    axes[1, 0].bar(results_4dof['Framework'], results_4dof['Accuracy < 0.5mm (%)'], alpha=0.7, label='4-DOF')
    axes[1, 0].set_title('Accuracy < 0.5mm (%)')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend()

    # Plot training time
    axes[1, 1].bar(results_3dof['Framework'], results_3dof['Training Time (s)'], label='3-DOF')
    axes[1, 1].bar(results_4dof['Framework'], results_4dof['Training Time (s)'], alpha=0.7, label='4-DOF')
    axes[1, 1].set_title('Training Time (s)')
    axes[1, 1].set_ylabel('Seconds')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{results_dir}/performance_summary.png')

    print("\nTesting completed. Results saved to results")
