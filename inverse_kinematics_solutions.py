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
from models.ikpy_visualizer import IKPyVisualizer
from utils.Forward_kinematics import ForwardKinematicsDH
from utils.data_for_simulation import DataGeneratorDH
from models.neural_network import TensorFlowModel, PyTorchModel, ScikitLearnModel


class InverseKinematicsBase:
    """Base class for IK solvers."""

    def __init__(self, fk_dh_model, num_model_dof, nn_input_dim, nn_output_dim,
                 model_type='tensorflow', nn_hidden_layers=(128, 64), nn_activation='relu'):
        self.fk_model = fk_dh_model
        self.num_model_dof = num_model_dof  # Actual DOF the NN predicts (3 for planar, 4 for spatial)
        self.nn_input_dim = nn_input_dim  # Dimension of target pose fed to NN (e.g., 3 for x,y,z)
        self.nn_output_dim = nn_output_dim  # Should match num_model_dof

        if self.nn_output_dim != self.num_model_dof:
            raise ValueError("NN output dimension must match the number of DOFs the model predicts.")

        self.model_type = model_type
        self.nn_hidden_layers = nn_hidden_layers
        self.nn_activation = nn_activation

        self._initialize_nn_model()
        self.is_trained = False
        self.data_generator = None  # To be set by subclasses

    def _initialize_nn_model(self):
        if self.model_type == 'tensorflow':
            self.model = TensorFlowModel(input_dimension=self.nn_input_dim, output_dimension=self.nn_output_dim,
                                         hidden_layers=self.nn_hidden_layers, activation=self.nn_activation)
        elif self.model_type == 'pytorch':
            self.model = PyTorchModel(input_dimension=self.nn_input_dim, output_dimension=self.nn_output_dim,
                                      hidden_layers=self.nn_hidden_layers, activation=self.nn_activation)
        elif self.model_type == 'sklearn':
            # Scikit-learn's MLPRegressor might need output_dimension for clarity if used elsewhere,
            # but its constructor doesn't take it directly.
            self.model = ScikitLearnModel(input_dimension=self.nn_input_dim, output_dimension=self.nn_output_dim,
                                          hidden_layer_sizes=self.nn_hidden_layers, activation=self.nn_activation)
        else:
            raise ValueError("Invalid model type specified.")

    def train(self, X_ee_poses, y_joint_angles, epochs=100, batch_size=32,
              validation_split_for_nn_train=0.1, verbose=1):
        """
        Trains the neural network model and evaluates Cartesian error on a test split.
        """
        if X_ee_poses.shape[0] == 0 or y_joint_angles.shape[0] == 0:
            print(f"Error: No training data provided for {self.num_model_dof}-DOF model. Aborting training.")
            return {}, {}
        if X_ee_poses.shape[0] != y_joint_angles.shape[0]:
            raise ValueError("X_ee_poses and y_joint_angles must have the same number of samples.")

        # Split data for final evaluation of this training run (Cartesian error)
        X_train_nn, X_test_eval, y_train_nn, y_test_eval = train_test_split(
            X_ee_poses, y_joint_angles, test_size=0.2, random_state=42
        )

        print(f"Training {self.num_model_dof}-DOF IK model ({self.model_type}) with {X_train_nn.shape[0]} samples...")
        nn_training_history = self.model.train(X_train_nn, y_train_nn, epochs=epochs, batch_size=batch_size,
                                               validation_split=validation_split_for_nn_train, verbose=verbose)

        print(f"\nEvaluating {self.num_model_dof}-DOF model on internal test set (Cartesian metrics)...")
        cartesian_errors_on_test_eval = []
        if X_test_eval.shape[0] > 0:
            for i in range(X_test_eval.shape[0]):
                target_cartesian_pos = X_test_eval[i]
                true_joint_angles_for_target = y_test_eval[i]  # For context, not directly used by verify_accuracy

                predicted_angles = self.predict(target_cartesian_pos)
                error_mm = self.verify_accuracy(target_cartesian_pos, predicted_angles, true_joint_angles_for_target)
                cartesian_errors_on_test_eval.append(error_mm)

            mean_err = np.mean(cartesian_errors_on_test_eval) if cartesian_errors_on_test_eval else float('nan')
            max_err = np.max(cartesian_errors_on_test_eval) if cartesian_errors_on_test_eval else float('nan')
            acc_lt_0_5 = np.mean(
                [err < 0.5 for err in cartesian_errors_on_test_eval]) * 100 if cartesian_errors_on_test_eval else 0.0

            eval_metrics = {
                'mean_cartesian_error_mm': mean_err,
                'max_cartesian_error_mm': max_err,
                'accuracy_lt_0.5mm_percent': acc_lt_0_5,
                'num_eval_samples': len(cartesian_errors_on_test_eval)
            }
            print(f"  Mean Cartesian Error: {mean_err:.4f} mm")
            print(f"  Max Cartesian Error: {max_err:.4f} mm")
            print(f"  Accuracy (<0.5mm): {acc_lt_0_5:.2f}% on {len(cartesian_errors_on_test_eval)} samples.")
        else:
            print(f"No samples in X_test_eval for {self.num_model_dof}-DOF Cartesian evaluation.")
            eval_metrics = {}

        self.is_trained = True
        return nn_training_history, eval_metrics

    def predict(self, target_end_effector_pose):
        """Predicts joint angles for a target end-effector pose."""
        pose_array = np.array(target_end_effector_pose)
        if pose_array.ndim == 1:
            pose_array = pose_array.reshape(1, -1)
        if pose_array.shape[1] != self.nn_input_dim:
            raise ValueError(f"NN Input dim mismatch. Expected {self.nn_input_dim}, got {pose_array.shape[1]}")
        return self.model.predict(pose_array)[0]  # Return first (and usually only) prediction

    def verify_accuracy(self, target_ee_pose_mm, predicted_angles_rad, true_angles_rad_for_context=None):
        """Must be implemented by subclasses based on their specific FK."""
        raise NotImplementedError

    def save_model(self, model_dir_path):
        """Saves the trained NN model."""
        model_filename = f"ik_model_{self.num_model_dof}dof_{self.model_type}"
        if self.model_type == 'tensorflow':
            model_filename += ".keras"
        elif self.model_type == 'pytorch':
            model_filename += ".pth"
        elif self.model_type == 'sklearn':
            model_filename += ".joblib"

        full_model_path = os.path.join(model_dir_path, model_filename)
        os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
        self.model.save(full_model_path)
        print(f"{self.num_model_dof}-DOF {self.model_type} model saved to {full_model_path}")

    def load_model(self, model_dir_path):
        """Loads a trained NN model."""
        model_filename = f"ik_model_{self.num_model_dof}dof_{self.model_type}"
        if self.model_type == 'tensorflow':
            model_filename += ".keras"
        elif self.model_type == 'pytorch':
            model_filename += ".pth"
        elif self.model_type == 'sklearn':
            model_filename += ".joblib"

        full_model_path = os.path.join(model_dir_path, model_filename)
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model file not found: {full_model_path}")
        self.model.load(full_model_path)
        self.is_trained = True
        print(f"{self.num_model_dof}-DOF {self.model_type} model loaded from {full_model_path}")


class InverseKinematics3DOFPlanar(InverseKinematicsBase):
    """
    IK for a 3-DOF planar arm (e.g., shoulder, elbow, wrist_pitch in a plane).
    The NN predicts 3 joint angles [q2, q3, q4] for a target [x, y, z] in that plane.
    The base rotation (q1) is considered fixed for this planar operation.
    """

    def __init__(self, fk_dh_model, fixed_base_rotation_rad=0.0, joint_angle_limits=None, **kwargs):
        # NN input is (x,y,z) of the planar target, output is 3 joint angles (q2,q3,q4)
        super().__init__(fk_dh_model, num_model_dof=3, nn_input_dim=3, nn_output_dim=3, **kwargs)
        self.fixed_base_rotation_rad = fixed_base_rotation_rad

        # Data generator for the 3 active DOFs
        # Joint angle limits should be for q2, q3, q4
        if joint_angle_limits is None:
            # Default limits for the 3 active joints (e.g. shoulder, elbow, wrist)
            default_limits_3dof = [(-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180), (-np.pi / 2, np.pi / 2)]
        else:
            default_limits_3dof = joint_angle_limits
        self.data_generator = DataGeneratorDH(fk_dh_model=self.fk_model, num_dof=3,
                                              joint_angle_limits=default_limits_3dof)
        print(f"IK3DOFPlanar using fixed base rotation: {np.degrees(self.fixed_base_rotation_rad):.1f} deg")

    def generate_training_data(self, num_samples=10000):
        print(f"Generating {num_samples} training samples for 3-DOF planar model...")
        # generate_data for 3-DOF in DataGeneratorDH takes fixed_base_rotation_rad
        X_ee_poses, y_3dof_angles = self.data_generator.generate_data(
            num_samples,
            fixed_base_rotation_for_3dof_rad=self.fixed_base_rotation_rad
        )
        print(f"Generated {X_ee_poses.shape[0]} 3-DOF planar samples.")
        return X_ee_poses, y_3dof_angles

    def verify_accuracy(self, target_ee_pose_mm, predicted_3dof_angles_rad, true_angles_rad_for_context=None):
        # FK for 3-DOF planar takes 3 angles (sh,elb,wr) + fixed base rotation
        x_pred, y_pred, z_pred = self.fk_model.forward_kinematics_3dof_planar(
            predicted_3dof_angles_rad,
            base_rotation_rad=self.fixed_base_rotation_rad
        )
        target_pos = np.array(target_ee_pose_mm).flatten()
        error = np.sqrt(
            (target_pos[0] - x_pred) ** 2 +
            (target_pos[1] - y_pred) ** 2 +
            (target_pos[2] - z_pred) ** 2
        )
        return error

    def visualize_prediction(self, target_ee_pose_mm, predicted_3dof_angles_rad):
        """Visualizes the 3-DOF planar arm. For this, we need all 4 D-H angles for the FK visualizer."""
        # Construct the full 4-angle set for visualization if using the 4-DOF visualizer
        full_angles_for_viz = [self.fixed_base_rotation_rad] + list(predicted_3dof_angles_rad)

        # Achieved position
        achieved_pos = self.fk_model.forward_kinematics_3dof_planar(predicted_3dof_angles_rad,
                                                                    self.fixed_base_rotation_rad)
        error = self.verify_accuracy(target_ee_pose_mm, predicted_3dof_angles_rad)

        fig = self.fk_model.visualize_arm_4dof_spatial(full_angles_for_viz, target_pos_mm=target_ee_pose_mm)
        ax = fig.axes[0]
        ax.set_title(
            f'3-DOF Planar Arm (Base Rot: {np.degrees(self.fixed_base_rotation_rad):.1f}°)\nTarget: {np.array(target_ee_pose_mm).round(1)} mm\nAchieved: {np.array(achieved_pos).round(1)} mm\nCartesian Error: {error:.3f} mm',
            fontsize=10)
        return fig


class InverseKinematics4DOFSpatial(InverseKinematicsBase):
    """
    IK for a 4-DOF spatial arm (base, shoulder, elbow, wrist_pitch).
    The NN predicts 4 joint angles [q1, q2, q3, q4] for a target [x, y, z].
    """

    def __init__(self, fk_dh_model, joint_angle_limits=None, **kwargs):
        # NN input is (x,y,z), output is 4 joint angles (q1,q2,q3,q4)
        super().__init__(fk_dh_model, num_model_dof=4, nn_input_dim=3, nn_output_dim=4, **kwargs)

        # Data generator for the 4 active DOFs
        if joint_angle_limits is None:
            default_limits_4dof = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180),
                                   (-np.pi / 2, np.pi / 2)]
        else:
            default_limits_4dof = joint_angle_limits
        self.data_generator = DataGeneratorDH(fk_dh_model=self.fk_model, num_dof=4,
                                              joint_angle_limits=default_limits_4dof)

    def generate_training_data(self, num_samples=20000):
        print(f"Generating {num_samples} training samples for 4-DOF spatial model...")
        X_ee_poses, y_4dof_angles = self.data_generator.generate_data(num_samples)
        print(f"Generated {X_ee_poses.shape[0]} 4-DOF spatial samples.")
        return X_ee_poses, y_4dof_angles

    def verify_accuracy(self, target_ee_pose_mm, predicted_4dof_angles_rad, true_angles_rad_for_context=None):
        # FK for 4-DOF spatial takes all 4 angles
        x_pred, y_pred, z_pred = self.fk_model.forward_kinematics_4dof_spatial(predicted_4dof_angles_rad)
        target_pos = np.array(target_ee_pose_mm).flatten()
        error = np.sqrt(
            (target_pos[0] - x_pred) ** 2 +
            (target_pos[1] - y_pred) ** 2 +
            (target_pos[2] - z_pred) ** 2
        )
        return error

    def newton_raphson_minimization(self, target_pos_mm, initial_angles_rad, max_iter=100, tol_mm=1e-3,
                                    learning_rate=0.1):
        """Refines joint angles using Newton-Raphson for 4-DOF spatial arm."""
        angles = np.array(initial_angles_rad, dtype=float)
        target_pos_mm = np.array(target_pos_mm, dtype=float)

        for iteration in range(max_iter):
            current_pos_mm = np.array(self.fk_model.forward_kinematics_4dof_spatial(angles))
            error_vec_mm = target_pos_mm - current_pos_mm
            current_error_mm = np.linalg.norm(error_vec_mm)

            if current_error_mm < tol_mm:
                break

            jacobian = np.zeros((3, self.num_model_dof))  # 3D position, N=num_model_dof angles
            epsilon = 1e-6
            for j in range(self.num_model_dof):  # Iterate through the 4 DOFs
                angles_perturbed = angles.copy()
                angles_perturbed[j] += epsilon
                pos_perturbed_mm = np.array(self.fk_model.forward_kinematics_4dof_spatial(angles_perturbed))
                jacobian[:, j] = (pos_perturbed_mm - current_pos_mm) / epsilon

            try:  # Damped Least Squares
                lambda_damping = 0.01
                delta_angles = np.linalg.inv(
                    jacobian.T @ jacobian + lambda_damping * np.eye(self.num_model_dof)) @ jacobian.T @ error_vec_mm
                angles += learning_rate * delta_angles
            except np.linalg.LinAlgError:
                angles += np.random.uniform(-0.005, 0.005, self.num_model_dof)  # Smaller random step
        else:  # No break from loop
            # print(f"NR 4DOF did not converge. Final error: {current_error_mm:.4f} mm")
            pass
        return angles

    def visualize_prediction(self, target_ee_pose_mm, predicted_4dof_angles_rad):
        achieved_pos = self.fk_model.forward_kinematics_4dof_spatial(predicted_4dof_angles_rad)
        error = self.verify_accuracy(target_ee_pose_mm, predicted_4dof_angles_rad)

        fig = self.fk_model.visualize_arm_4dof_spatial(predicted_4dof_angles_rad, target_pos_mm=target_ee_pose_mm)
        ax = fig.axes[0]
        ax.set_title(
            f'4-DOF Spatial Arm ({self.model_type})\nTarget: {np.array(target_ee_pose_mm).round(1)} mm\nAchieved: {np.array(achieved_pos).round(1)} mm\nCartesian Error: {error:.3f} mm',
            fontsize=10)
        return fig


if __name__ == '__main__':
    print("Running IK Solutions Main Script Example...")
    # CRITICAL: Ensure these link parameters match your ForwardKinematicsDH and your actual arm!
    fk_arm_model = ForwardKinematicsDH(d1=70.0, a2=100.0, a3=100.0, a4=60.0)

    # --- Test 3-DOF Planar ---
    print("\n--- Testing 3-DOF Planar IK ---")
    # Define joint limits for the 3 active joints (e.g. shoulder, elbow, wrist_pitch)
    limits_3dof_active = [(-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180), (-np.pi / 2, np.pi / 2)]
    ik_3dof = InverseKinematics3DOFPlanar(
        fk_dh_model=fk_arm_model,
        fixed_base_rotation_rad=np.deg2rad(0),  # Planar motion at 0 deg base rotation
        joint_angle_limits=limits_3dof_active,
        model_type='tensorflow',
        nn_hidden_layers=(64, 32)
    )
    X_3dof_data, y_3dof_data = ik_3dof.generate_training_data(num_samples=1000)  # Small set for example
    if X_3dof_data.shape[0] > 0:
        ik_3dof.train(X_3dof_data, y_3dof_data, epochs=20, verbose=1)
        if ik_3dof.is_trained:
            # Example target in the plane of motion (assuming base rotation = 0)
            # Calculate a reachable point using FK for a known set of 3dof angles
            sample_3dof_angles = [np.deg2rad(10), np.deg2rad(20), np.deg2rad(10)]
            target_3dof_ee_pos = fk_arm_model.forward_kinematics_3dof_planar(sample_3dof_angles,
                                                                             base_rotation_rad=ik_3dof.fixed_base_rotation_rad)
            print(f"3-DOF Target EE: {target_3dof_ee_pos}")

            pred_angles_3dof = ik_3dof.predict(target_3dof_ee_pos)
            error_3dof = ik_3dof.verify_accuracy(target_3dof_ee_pos, pred_angles_3dof)
            print(
                f"3-DOF Prediction for {np.around(target_3dof_ee_pos, 1)}: angles={np.degrees(pred_angles_3dof).round(1)} deg, error={error_3dof:.3f} mm")
            fig3 = ik_3dof.visualize_prediction(target_3dof_ee_pos, pred_angles_3dof)
    else:
        print("No data generated for 3-DOF test.")

    # --- Test 4-DOF Spatial ---
    print("\n--- Testing 4-DOF Spatial IK ---")
    # Define joint limits for all 4 active joints (base, shoulder, elbow, wrist_pitch)
    limits_4dof_active = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180), (-np.pi / 2, np.pi / 2)]
    ik_4dof = InverseKinematics4DOFSpatial(
        fk_dh_model=fk_arm_model,
        joint_angle_limits=limits_4dof_active,
        model_type='tensorflow',
        nn_hidden_layers=(128, 64, 32)
    )
    X_4dof_data, y_4dof_data = ik_4dof.generate_training_data(num_samples=1000)  # Small set
    if X_4dof_data.shape[0] > 0:
        ik_4dof.train(X_4dof_data, y_4dof_data, epochs=20, verbose=1)
        if ik_4dof.is_trained:
            # Example target
            sample_4dof_angles = [np.deg2rad(10), np.deg2rad(20), np.deg2rad(30), np.deg2rad(5)]
            target_4dof_ee_pos = fk_arm_model.forward_kinematics_4dof_spatial(sample_4dof_angles)
            print(f"4-DOF Target EE: {target_4dof_ee_pos}")

            pred_angles_nn_4dof = ik_4dof.predict(target_4dof_ee_pos)
            error_nn_4dof = ik_4dof.verify_accuracy(target_4dof_ee_pos, pred_angles_nn_4dof)
            print(
                f"4-DOF NN Pred for {np.around(target_4dof_ee_pos, 1)}: angles={np.degrees(pred_angles_nn_4dof).round(1)} deg, error={error_nn_4dof:.3f} mm")

            pred_angles_nr_4dof = ik_4dof.newton_raphson_minimization(target_4dof_ee_pos, pred_angles_nn_4dof,
                                                                      tol_mm=0.1)
            error_nr_4dof = ik_4dof.verify_accuracy(target_4dof_ee_pos, pred_angles_nr_4dof)
            print(
                f"4-DOF NR Pred for {np.around(target_4dof_ee_pos, 1)}: angles={np.degrees(pred_angles_nr_4dof).round(1)} deg, error={error_nr_4dof:.3f} mm")
            fig4 = ik_4dof.visualize_prediction(target_4dof_ee_pos, pred_angles_nr_4dof)
    else:
        print("No data generated for 4-DOF test.")

    plt.show()