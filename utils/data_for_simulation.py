"""
Data Generator for Inverse Kinematics Neural Network Training

This module generates training data for neural network models by sampling
random joint configurations and calculating their corresponding end-effector positions.
"""

import numpy as np
import os
import pandas as pd
from utils.Forward_kinematics import ForwardKinematicsDH


class DataGeneratorDH:
    """
    Generates training data (End-Effector Poses, Joint Angles) using D-H based FK.
    """

    def __init__(self, fk_dh_model, num_dof, joint_angle_limits=None):
        """
        Args:
            fk_dh_model (ForwardKinematicsDH): An instance of the D-H FK model.
            num_dof (int): Number of degrees of freedom to generate data for (3 or 4).
            joint_angle_limits (list of tuples): [(min_rad, max_rad), ...] for each joint.
                                                 Length must match num_dof.
        """
        self.fk_model = fk_dh_model
        self.num_dof = num_dof

        if joint_angle_limits is None:

            if self.num_dof == 3:  # Shoulder, Elbow, Wrist_Pitch
                self.limits = [
                    (-np.pi / 4, np.pi / 4),  # q2_shoulder (e.g., -45 to +45 deg)
                    (0, np.pi * (120 / 180)),  # q3_elbow (e.g., 0 to 120 deg)
                    (-np.pi * (60 / 180), np.pi * (60 / 180))  # q4_wrist_pitch
                ]
            elif self.num_dof == 4:  # Base, Shoulder, Elbow, Wrist_Pitch
                self.limits = [
                    (0, np.pi * (120 / 180)),  # q1_base (e.g., 0 to +120 deg)
                    (-np.pi / 4, np.pi / 4),  # q2_shoulder
                    (0, np.pi * (120 / 180)),  # q3_elbow
                    (-np.pi * (60 / 180), np.pi * (60 / 180))  # q4_wrist_pitch
                ]
            else:
                raise ValueError("num_dof must be 3 or 4 for this generator.")
        else:
            if len(joint_angle_limits) != self.num_dof:
                raise ValueError(
                    f"Length of joint_angle_limits ({len(joint_angle_limits)}) must match num_dof ({self.num_dof}).")
            self.limits = joint_angle_limits

        print(f"DataGeneratorDH initialized for {self.num_dof}-DOF with joint limits (radians):")
        for i, lim in enumerate(self.limits):
            print(
                f"  Joint {i + 1}: Min={lim[0]:.2f} rad ({np.degrees(lim[0]):.1f} deg), Max={lim[1]:.2f} rad ({np.degrees(lim[1]):.1f} deg)")

    def _generate_random_joint_angles(self):
        """Generates a single set of random joint angles within specified limits."""
        return np.array([np.random.uniform(low, high) for low, high in self.limits])

    def generate_data(self, num_samples, fixed_base_rotation_for_3dof_rad=None):
        """
        Generates dataset (X: EE poses, y: joint angles).

        Args:
            num_samples (int): Number of data points to generate.
            fixed_base_rotation_for_3dof_rad (float, optional):
                If num_dof is 3, this specifies the fixed base rotation (q1) to use.
                If None, and num_dof is 3, an error will be raised if the FK expects it.
                The `forward_kinematics_3dof_planar` method in `ForwardKinematicsDH`
                takes `base_rotation_rad` as an argument.

        Returns:
            tuple: (X_data, y_data)
                   X_data: numpy array of shape (num_samples, 3) for EE positions (x,y,z).
                   y_data: numpy array of shape (num_samples, num_dof) for joint angles.
        """

        if self.num_dof == 3 and fixed_base_rotation_for_3dof_rad is None:
            print(
                "Warning: Generating 3-DOF data. `fixed_base_rotation_for_3dof_rad` is None. Assuming 0.0 for FK calculation if needed by FK method.")
            fixed_base_rotation_for_3dof_rad = 0.0

        X_data = []
        y_data = []

        for _ in range(num_samples):
            joint_angles = self._generate_random_joint_angles()

            if self.num_dof == 3:
                # FK for 3-DOF planar takes 3 angles (sh,elb,wr) + fixed base rotation
                ee_pos = self.fk_model.forward_kinematics_3dof_planar(
                    joint_angles,  # These are q2, q3, q4
                    fixed_q1_base_rotation_rad=fixed_base_rotation_for_3dof_rad
                )
                y_data.append(joint_angles)  # Store the 3 active joint angles
            elif self.num_dof == 4:
                # FK for 4-DOF takes all 4 angles (base,sh,elb,wr)
                ee_pos = self.fk_model.forward_kinematics_4dof_spatial(joint_angles)
                y_data.append(joint_angles)  # Store all 4 active joint angles
            else:
                continue  # Should not happen with initial checks

            X_data.append(ee_pos)
            #dataset = []
            #dataset.append(X_data)
            #dataset.append(y_data)
            #current_script_dir = os.path.dirname(os.path.abspath(__file__))
            #dataset_dir = os.path.join(current_script_dir, "dataset")
            #final_dataset_df = pd.concat(dataset, ignore_index=True)
            #summary_csv_path = os.path.join(dataset_dir, "dataset.csv")
            #final_dataset_df.to_csv(summary_csv_path, index=False)

        return np.array(X_data), np.array(y_data)


# Example Usage
if __name__ == '__main__':
    # Convert degrees to radians for the generator
    # Base (q1): 0 to 180 deg; Shoulder (q2): -45 to +45 deg; Elbow (q3): 0 to 120 deg; Wrist (q4): -90 to +90 deg
    realistic_limits_4dof_deg = [(0, 180), (-45, 45), (0, 120), (-90, 90)]
    realistic_limits_4dof_rad = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in realistic_limits_4dof_deg]

    # For 3-DOF (e.g., q2, q3, q4 of the 4-DOF structure):
    realistic_limits_3dof_deg = [(-45, 45), (0, 120), (-90, 90)]  # Shoulder, Elbow, Wrist
    realistic_limits_3dof_rad = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in realistic_limits_3dof_deg]

    # CRITICAL: Use the same link parameters as in ForwardKinematicsDH for consistency
    fk_dh = ForwardKinematicsDH(d1=70.0, a2=100.0, a3=100.0, a4=60.0)

    # --- 3-DOF Planar Data Generation ---
    # Define joint limits for the 3 active joints (shoulder, elbow, wrist_pitch)
    # Example: q2_shoulder, q3_elbow, q4_wrist
    limits_3dof = [
        (-np.pi / 4, np.pi / 4),  # Shoulder pitch
        (0, np.pi * (120 / 180)),  # Elbow pitch (e.g. 0 to 150 deg)
        (-np.pi / 2, np.pi / 2)  # Wrist pitch
    ]
    data_gen_3dof = DataGeneratorDH(fk_dh_model=fk_dh, num_dof=3, joint_angle_limits=realistic_limits_3dof_rad)
    # For 3-DOF planar, specify the fixed base rotation (q1)
    X3, y3 = data_gen_3dof.generate_data(num_samples=5, fixed_base_rotation_for_3dof_rad=np.deg2rad(0))
    print("\n--- 3-DOF Planar Data (Base fixed at 0 deg) ---")
    for i in range(len(X3)):
        print(f"Joints (deg): {[f'{np.degrees(a):.1f}' for a in y3[i]]} -> EE Pos (mm): {[f'{p:.2f}' for p in X3[i]]}")
    """
    # --- 4-DOF Spatial Data Generation ---
    # Define joint limits for the 4 active joints (base, shoulder, elbow, wrist_pitch)
    # Example: q1_base, q2_shoulder, q3_elbow, q4_wrist
    limits_4dof = [
        (0, np.pi * (120 / 180)),  # Base rotation (full circle)
        (-np.pi / 4, np.pi / 4),  # Shoulder pitch
        (0, np.pi * (120 / 180)),  # Elbow pitch
        (-np.pi / 2, np.pi / 2)  # Wrist pitch
    ]
    """
    data_gen_4dof = DataGeneratorDH(fk_dh_model=fk_dh, num_dof=4, joint_angle_limits=realistic_limits_4dof_rad)
    X4, y4 = data_gen_4dof.generate_data(num_samples=5)
    print("\n--- 4-DOF Spatial Data ---")
    for i in range(len(X4)):
        print(f"Joints (deg): {[f'{np.degrees(a):.1f}' for a in y4[i]]} -> EE Pos (mm): {[f'{p:.2f}' for p in X4[i]]}")