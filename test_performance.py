"""
Performance Testing and Evaluation for Inverse Kinematics Neural Network Solutions

This script tests and evaluates the performance of the inverse kinematics solutions
for both 4-DOF (3D) and 3-DOF (2D) configurations using different neural network frameworks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.Forward_kinematics import ForwardKinematics
from utils.data_for_simulation import DataGenerator
from inverse_kinematics_solutions import InverseKinematics4DOF, InverseKinematics3DOF

#Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

#Define results directory relative to the current script
results_dir = os.path.join(current_dir, "results")

#Define trained_models directory relative to the current script
trained_models_dir = os.path.join(current_dir, "trained_models")

def generate_test_points(num_points=10):
    """
        Generate test points for evaluating the inverse kinematics solutions.

        Args:
            num_points (int): Number of test points to generate

        :return:
            tuple: (points_4dof, points_3dof) where each is a list of end-effector positions
    """
    #Create a forward kinematics model with default link lengths
    fk = ForwardKinematics()

    #Generate random joint angles for 4-DOF
    points_4dof = []
    for _ in range(num_points):
        theta0 = np.random.uniform(-np.pi, np.pi)
        theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta3 = np.random.uniform(-np.pi / 2, np.pi / 2)

        #Calculate forward kinematics
        x, y, z = fk.forward_kinematics_4dof([theta0, theta1, theta2, theta3])
        points_4dof.append([x, y, z])

    #Generate random joint angles for 3-DOF
    points_3dof = []
    for _ in range(num_points):
        theta0 = np.random.uniform(-np.pi, np.pi)
        theta1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        theta2 = np.random.uniform(-np.pi / 2, np.pi / 2)

        #Calculate forward kinematics
        x, y = fk.forward_kinematics_3dof([theta0, theta1, theta2])
        points_3dof.append([x, y])

    return points_4dof, points_3dof


def train_and_evaluate_models(train_epochs=100, num_samples=10000, test_points=10):
    """
        Train and evaluate all models for both 4-DOF and 3-DOF configurations.

        Args:
            train_epochs (int): Number of training epochs
            num_samples (int): Number of training samples
            test_points (int): Number of test points for evaluation

        :return:
            tuple: (results_4dof, results_3dof) where each is a pandas DataFrame with evaluation results
    """
    #Generate test points
    points_4dof, points_3dof = generate_test_points(test_points)

    #Create results dataframes
    results_4dof = pd.DataFrame(columns=[
        'Framework', 'Training Time (s)', 'Inference Time (ms)',
        'Mean Error (mm)', 'Max Error (mm)', 'Accuracy < 0.5mm (%)'
    ])

    results_3dof = pd.DataFrame(columns=[
        'Framework', 'Training Time (s)', 'Inference Time (ms)',
        'Mean Error (mm)', 'Max Error (mm)', 'Accuracy < 0.5mm (%)'
    ])

    #Create data generators
    data_gen_4dof = DataGenerator()
    data_gen_3dof = DataGenerator()

    #Generate training data
    print("Generating training data...")
    X_4dof, y_4dof = data_gen_4dof.generate_dataset_4dof(num_samples)
    X_3dof, y_3dof = data_gen_3dof.generate_dataset_3dof(num_samples)

    #Split data into training and testing sets
    X_train_4dof, X_test_4dof, y_train_4dof, y_test_4dof = train_test_split(
        X_4dof, y_4dof, test_size=0.2, random_state=42
    )

    X_train_3dof, X_test_3dof, y_train_3dof, y_test_3dof = train_test_split(
        X_3dof, y_3dof, test_size=0.2, random_state=42
    )

    #Create models directory
    #os.makedirs(f'{trained_models_dir}', exist_ok=True)

    #Train and evaluate 4-DOF models
    for framework in ['tensorflow', 'pytorch', 'sklearn']:
        print(f"\nTraining 4-DOF model using {framework}...")

        #Create and train model
        model_4dof = InverseKinematics4DOF(model_type=framework)

        #Measure training time
        start_time = time.time()
        history, metrics = model_4dof.train(
            X_train_4dof, y_train_4dof,
            epochs=train_epochs,
            batch_size=64,
            verbose=2
        )
        training_time = time.time() - start_time

        #Save model
        model_path = f'{trained_models_dir}/model_4dof_{framework}.keras'
        model_4dof.save_model(model_path)

        #Evaluate on test points
        errors = []
        inference_times = []

        for point in points_4dof:
            #Measure inference time
            start_time = time.time()
            predicted_angles = model_4dof.predict(point)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)

            #Calculate error
            error = model_4dof.verify_accuracy(point, predicted_angles)
            errors.append(error)

        #Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        accuracy_percentage = np.mean([error < 0.5 for error in errors]) * 100
        mean_inference_time = np.mean(inference_times)

        #Add to results
        results_4dof = pd.concat([results_4dof, pd.DataFrame([{
            'Framework': framework,
            'Training Time (s)': training_time,
            'Inference Time (ms)': mean_inference_time,
            'Mean Error (mm)': mean_error,
            'Max Error (mm)': max_error,
            'Accuracy < 0.5mm (%)': accuracy_percentage
        }])], ignore_index=True)

        print(f"4-DOF {framework} model evaluation:")
        print(f"  Training time: {training_time:.2f} s")
        print(f"  Mean inference time: {mean_inference_time:.2f} ms")
        print(f"  Mean error: {mean_error:.4f} mm")
        print(f"  Max error: {max_error:.4f} mm")
        print(f"  Accuracy < 0.5mm: {accuracy_percentage:.2f}%")

    #Train and evaluate 3-DOF models
    for framework in ['tensorflow', 'pytorch', 'sklearn']:
        print(f"\nTraining 3-DOF model using {framework}...")

        #Create and train model
        model_3dof = InverseKinematics3DOF(model_type=framework)

        #Measure training time
        start_time = time.time()
        history, metrics = model_3dof.train(
            X_train_3dof, y_train_3dof,
            epochs=train_epochs,
            batch_size=64,
            verbose=2
        )
        training_time = time.time() - start_time

        #Save model
        model_path = f'{trained_models_dir}/model_3dof_{framework}.keras'
        model_3dof.save_model(model_path)

        #Evaluate on test points
        errors = []
        inference_times = []

        for point in points_3dof:
            #Measure inference time
            start_time = time.time()
            predicted_angles = model_3dof.predict(point)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)

            #Calculate error
            error = model_3dof.verify_accuracy(point, predicted_angles)
            errors.append(error)

        #Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        accuracy_percentage = np.mean([error < 0.5 for error in errors]) * 100
        mean_inference_time = np.mean(inference_times)

        #Add to results
        results_3dof = pd.concat([results_3dof, pd.DataFrame([{
            'Framework': framework,
            'Training Time (s)': training_time,
            'Inference Time (ms)': mean_inference_time,
            'Mean Error (mm)': mean_error,
            'Max Error (mm)': max_error,
            'Accuracy < 0.5mm (%)': accuracy_percentage
        }])], ignore_index=True)

        print(f"3-DOF {framework} model evaluation:")
        print(f"  Training time: {training_time:.2f} s")
        print(f"  Mean inference time: {mean_inference_time:.2f} ms")
        print(f"  Mean error: {mean_error:.4f} mm")
        print(f"  Max error: {max_error:.4f} mm")
        print(f"  Accuracy < 0.5mm: {accuracy_percentage:.2f}%")

    #Save results to CSV
    results_4dof.to_csv(f'{results_dir}/results_4dof.csv', index=False)
    results_3dof.to_csv(f'{results_dir}/results_3dof.csv', index=False)

    return results_4dof, results_3dof


def plot_results(results_4dof, results_3dof):
    """
        Plot the evaluation results.

        Args:
            results_4dof (pandas.DataFrame): Results for 4-DOF models
            results_3dof (pandas.DataFrame): Results for 3-DOF models
    """
    #Create results directory
    os.makedirs(f'{results_dir}', exist_ok=True)

    #Set up figure for 4-DOF results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('4-DOF Model Performance Comparison', fontsize=16)

    #Plot training time
    axes[0, 0].bar(results_4dof['Framework'], results_4dof['Training Time (s)'])
    axes[0, 0].set_title('Training Time (s)')
    axes[0, 0].set_ylabel('Seconds')

    #Plot inference time
    axes[0, 1].bar(results_4dof['Framework'], results_4dof['Inference Time (ms)'])
    axes[0, 1].set_title('Inference Time (ms)')
    axes[0, 1].set_ylabel('Milliseconds')

    #Plot mean error
    axes[1, 0].bar(results_4dof['Framework'], results_4dof['Mean Error (mm)'])
    axes[1, 0].set_title('Mean Error (mm)')
    axes[1, 0].set_ylabel('Millimeters')

    #Plot accuracy percentage
    axes[1, 1].bar(results_4dof['Framework'], results_4dof['Accuracy < 0.5mm (%)'])
    axes[1, 1].set_title('Accuracy < 0.5mm (%)')
    axes[1, 1].set_ylabel('Percentage')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/4dof_results.png')

    #Set up figure for 3-DOF results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3-DOF Model Performance Comparison', fontsize=16)

    #Plot training time
    axes[0, 0].bar(results_3dof['Framework'], results_3dof['Training Time (s)'])
    axes[0, 0].set_title('Training Time (s)')
    axes[0, 0].set_ylabel('Seconds')

    #Plot inference time
    axes[0, 1].bar(results_3dof['Framework'], results_3dof['Inference Time (ms)'])
    axes[0, 1].set_title('Inference Time (ms)')
    axes[0, 1].set_ylabel('Milliseconds')

    #Plot mean error
    axes[1, 0].bar(results_3dof['Framework'], results_3dof['Mean Error (mm)'])
    axes[1, 0].set_title('Mean Error (mm)')
    axes[1, 0].set_ylabel('Millimeters')

    #Plot accuracy percentage
    axes[1, 1].bar(results_3dof['Framework'], results_3dof['Accuracy < 0.5mm (%)'])
    axes[1, 1].set_title('Accuracy < 0.5mm (%)')
    axes[1, 1].set_ylabel('Percentage')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/3dof_results.png')


def visualize_test_cases(num_cases=3):
    """
        Visualize a few test cases for both 4-DOF and 3-DOF models.

        Args:
            num_cases (int): Number of test cases to visualize
    """
    #Generate test points
    points_4dof, points_3dof = generate_test_points(num_cases)

    #Create results directory
    os.makedirs(f'{results_dir}/visualizations', exist_ok=True)

    #Load the best models (to do: increment a func to receive best MAE and load the model)

    model_4dof = InverseKinematics4DOF(model_type='pytorch')
    model_4dof.load_model(f'{trained_models_dir}/optimized_model_4dof_pytorch.pt')

    model_3dof = InverseKinematics3DOF(model_type='pytorch')
    model_3dof.load_model(f'{trained_models_dir}/optimized_model_3dof_pytorch.pt')

    #Visualize 4-DOF test cases
    for i, point in enumerate(points_4dof):
        predicted_angles = model_4dof.predict(point)
        error = model_4dof.verify_accuracy(point, predicted_angles)

        fig = model_4dof.visualize_prediction(point, predicted_angles)
        plt.savefig(f'{results_dir}/visualizations/4dof_case_{i + 1}.png')
        plt.close(fig)

        print(f"4-DOF Test Case {i + 1}:")
        print(f"  Target position: {point}")
        print(f"  Predicted angles: {predicted_angles}")
        print(f"  Error: {error:.4f} mm")

    # Visualize 3-DOF test cases
    for i, point in enumerate(points_3dof):
        predicted_angles = model_3dof.predict(point)
        error = model_3dof.verify_accuracy(point, predicted_angles)

        fig = model_3dof.visualize_prediction(point, predicted_angles)
        plt.savefig(f'{results_dir}/visualizations/3dof_case_{i + 1}.png')
        plt.close(fig)

        print(f"3-DOF Test Case {i + 1}:")
        print(f"  Target position: {point}")
        print(f"  Predicted angles: {predicted_angles}")
        print(f"  Error: {error:.4f} mm")




if __name__ == "__main__":
    # Create results directory
    os.makedirs(f'{results_dir}', exist_ok=True)

    # Train and evaluate models with reduced parameters for testing
    # In a real scenario, you would use more epochs and samples


    results_4dof, results_3dof = train_and_evaluate_models(
            train_epochs=300,
            num_samples=20000,
            test_points=100
        )

    """
        ik_solver = InverseKinematics4DOF()
        ik_solver.train(num_samples=10000)
        angles = ik_solver.predict([150, 100, 50])
        fig = ik_solver.visualize_prediction([150, 100, 50], angles)
        fig.savefig('arm_configuration.png')
    """

    #Plot results
    plot_results(results_4dof, results_3dof)


    #visualize_test_cases(num_cases=3)  # Reduced for testing
    #visualize_test_cases(num_cases=3)