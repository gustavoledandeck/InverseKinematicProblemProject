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


class NewtonRaphson_IK:
    def __init__(self, fk_function, jacobian_function, num_dof, joint_limits_rad=None,
                 orientation_dof_indices=None):
        self.fk_function = fk_function
        self.jacobian_function = jacobian_function
        self.num_dof = num_dof
        self.joint_limits_rad = np.array(joint_limits_rad) if joint_limits_rad is not None else None
        self.orientation_dof_indices = orientation_dof_indices if orientation_dof_indices is not None else []

    def _calculate_error(self, current_pose, target_pose):
        error = np.array(target_pose) - np.array(current_pose)
        for i in self.orientation_dof_indices:
            while error[i] > np.pi: error[i] -= 2 * np.pi
            while error[i] < -np.pi: error[i] += 2 * np.pi
        return error

    def solve(self, target_pose, initial_joint_angles_rad,
              max_iterations=100, position_tolerance=1e-4, orientation_tolerance_rad=1e-3,
              step_size_alpha=0.5, damping_lambda=0.01):
        q_current = np.array(initial_joint_angles_rad, dtype=float)
        target_pose = np.array(target_pose, dtype=float)

        if self.joint_limits_rad is not None:
            q_current = np.clip(q_current, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1])

        for i in range(max_iterations):
            current_pose = self.fk_function(q_current)
            error = self._calculate_error(current_pose, target_pose)

            pos_error_norm = np.linalg.norm(error[:3])
            orient_error_norm = 0.0
            if len(error) > 3 and self.orientation_dof_indices:
                orient_error_components = error[np.array(self.orientation_dof_indices)]
                orient_error_norm = np.linalg.norm(orient_error_components)
            final_error_norm = np.linalg.norm(error)

            if pos_error_norm < position_tolerance and \
                    (
                            len(error) <= 3 or not self.orientation_dof_indices or orient_error_norm < orientation_tolerance_rad):
                return q_current, True, i + 1, final_error_norm

            J = self.jacobian_function(q_current)
            try:
                if damping_lambda > 1e-9:  # Apply DLS
                    J_JT = J @ J.T
                    identity_matrix = np.eye(J_JT.shape[0])
                    # Ensure J_JT is invertible or regularized
                    J_pinv = J.T @ np.linalg.solve(J_JT + (damping_lambda ** 2) * identity_matrix,
                                                   np.eye(J_JT.shape[0]))
                else:  # Standard pseudo-inverse
                    J_pinv = np.linalg.pinv(J)
                delta_q = J_pinv @ error
            except np.linalg.LinAlgError:
                return q_current, False, i + 1, final_error_norm

            q_new = q_current + step_size_alpha * delta_q
            if self.joint_limits_rad is not None:
                q_new = np.clip(q_new, self.joint_limits_rad[:, 0], self.joint_limits_rad[:, 1])

            if np.linalg.norm(q_new - q_current) < 1e-6 * self.num_dof:
                current_pose_final = self.fk_function(q_current)
                error_final = self._calculate_error(current_pose_final, target_pose)
                return q_current, True, i + 1, np.linalg.norm(error_final)
            q_current = q_new

        return q_current, False, max_iterations, final_error_norm

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
            fixed_q1_base_rotation_rad=self.fixed_base_rotation_rad
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

        self.nr_solver_instance = None  # Initialize to None
        self.initialize_nr_solver(joint_angle_limits)  # Call initialization

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

    def initialize_nr_solver(self, active_joint_limits_rad):
        """Initializes the NR solver instance."""
        if active_joint_limits_rad is None: # Fallback if not passed during __init__
            # Get from data_generator if available, or use defaults
            if hasattr(self, 'data_generator') and self.data_generator:
                active_joint_limits_rad = self.data_generator.limits
            else: # Absolute fallback, should be defined better
                active_joint_limits_rad = [(-np.pi, np.pi)] * self.num_model_dof
                print("Warning: NR using default wide joint limits.")

        self.nr_solver_instance = NewtonRaphson_IK(
            fk_function=self.fk_model.forward_kinematics_4dof_spatial,
            jacobian_function=self.fk_model.jacobian_4dof_spatial, # CRITICAL: This must be analytical
            num_dof=self.num_model_dof, # Should be 4
            joint_limits_rad=active_joint_limits_rad,
            orientation_dof_indices=[] # If your 4-DOF IK targets only [x,y,z] for the NN
        )
        print(f"4-DOF NR solver initialized. Ensure fk_model.jacobian_4dof_spatial is implemented analytically.")

    def refined_predict_with_nr(self, target_pos_mm, initial_nn_angles_rad,
                                max_iter=50, tol_mm=0.1, learning_rate_nr=0.5, damping_nr=0.01):
        if self.nr_solver_instance is None:
            print("Error: NR solver not initialized. Call initialize_nr_solver first or ensure it's called in __init__.")
            return initial_nn_angles_rad # Fallback

        q_refined, success, iters, final_err = self.nr_solver_instance.solve(
            target_pose=np.array(target_pos_mm),
            initial_joint_angles_rad=np.array(initial_nn_angles_rad),
            max_iterations=max_iter,
            position_tolerance=tol_mm,
            orientation_tolerance_rad=1e-3, # Adjust if orientation is part of target_pose
            step_size_alpha=learning_rate_nr,
            damping_lambda=damping_nr
        )
        return q_refined

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