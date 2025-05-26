"""
Forward Kinematics Implementation for Robotic Arm

This module implements forward kinematics for 4-DOF and 3-DOF robotic arms.
Based on the mathematical foundations provided in literature about.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

LINK_LENGTH_BASE_TO_SHOULDER_Z_OFFSET = 70.0  # Vertical distance from base to shoulder joint axis
LINK_LENGTH_SHOULDER_TO_ELBOW = 100.0  # Length of the upper arm link
LINK_LENGTH_ELBOW_TO_WRIST = 100.0  # Length of the forearm link
LINK_LENGTH_WRIST_TO_EE = 60.0  # Length from wrist pitch joint to EE center


class ForwardKinematicsDH:
    """
    Forward Kinematics using Denavit-Hartenberg (D-H) parameters.
    Assumes a common hobbyist 6-DOF arm structure, used here for 3-DOF and 4-DOF.

    Joints used:
    - q1 (theta1): Base rotation (around Z0-axis). Used for 4-DOF.
    - q2 (theta2): Shoulder pitch (around Y1-axis, after Z0 rotation and d1 offset).
    - q3 (theta3): Elbow pitch (around Y2-axis).
    - q4 (theta4): Wrist pitch (around Y3-axis).
    """

    def __init__(self,
                 d1=LINK_LENGTH_BASE_TO_SHOULDER_Z_OFFSET,
                 a2=LINK_LENGTH_SHOULDER_TO_ELBOW,
                 a3=LINK_LENGTH_ELBOW_TO_WRIST,
                 a4=LINK_LENGTH_WRIST_TO_EE):  # a4 is length along X of final frame
        """
        Initializes with arm link parameters.
        Args:
            d1 (float): D-H 'd' for link 1 (offset along Z0 to frame 1, at J2).
            a2 (float): D-H 'a' for link 2 (length of link from J2 to J3, along X2).
            a3 (float): D-H 'a' for link 3 (length of link from J3 to J4, along X3).
            a4 (float): D-H 'a' for link 4 (length of link from J4 to EE, along X4).
        """
        self.d1 = float(d1)
        self.a2 = float(a2)
        self.a3 = float(a3)
        self.a4 = float(a4)  # Length from wrist pitch to EE point

        # Standard D-H parameters for this type of arm (alpha, a, d, theta)
        # Link | alpha_{i-1} | a_{i-1} | d_i     | theta_i
        # -----|---------------|-----------|-----------|-----------
        # 1    | 0             | 0         | self.d1   | q1 (base)
        # 2    | pi/2          | 0         | 0         | q2 (shoulder)
        # 3    | 0             | self.a2   | 0         | q3 (elbow)
        # 4    | 0             | self.a3   | 0         | q4 (wrist_pitch)
        # EE   | 0             | self.a4   | 0         | 0 (fixed relative to frame 4)

    def _dh_transformation_matrix(self, alpha, a, d, theta):
        """Computes a single D-H transformation matrix."""
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)

        T = np.array([
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0, s_alpha, c_alpha, d],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics_3dof_planar(self, joint_angles_rad, base_rotation_rad=0.0):
        """
        Calculates FK for the 3-DOF planar configuration (shoulder, elbow, wrist pitch)
        operating in a vertical plane, potentially rotated by a fixed base_rotation.

        Args:
            joint_angles_rad (list/np.array): [q2_shoulder, q3_elbow, q4_wrist_pitch] in radians.
            base_rotation_rad (float): Fixed base rotation (q1) in radians.

        Returns:
            tuple: (x, y, z) end-effector coordinates in mm.
        """
        if len(joint_angles_rad) != 3:
            raise ValueError("3-DOF planar FK requires 3 joint angles (shoulder, elbow, wrist).")

        q1_base = base_rotation_rad
        q2_shoulder, q3_elbow, q4_wrist = joint_angles_rad

        T01 = self._dh_transformation_matrix(0, 0, self.d1, q1_base)
        T12 = self._dh_transformation_matrix(np.pi / 2, 0, 0, q2_shoulder)
        T23 = self._dh_transformation_matrix(0, self.a2, 0, q3_elbow)
        T34 = self._dh_transformation_matrix(0, self.a3, 0, q4_wrist)
        # Transformation from frame 4 to End-Effector point
        # Assuming EE is simply offset along X-axis of frame 4 by a4
        T4_EE = self._dh_transformation_matrix(0, self.a4, 0, 0)

        T0_EE = T01 @ T12 @ T23 @ T34 @ T4_EE

        x = T0_EE[0, 3]
        y = T0_EE[1, 3]
        z = T0_EE[2, 3]
        return x, y, z

    def forward_kinematics_4dof_spatial(self, joint_angles_rad):
        """
        Calculates FK for the 4-DOF spatial configuration (base + shoulder, elbow, wrist pitch).

        Args:
            joint_angles_rad (list/np.array): [q1_base, q2_shoulder, q3_elbow, q4_wrist_pitch] in radians.

        Returns:
            tuple: (x, y, z) end-effector coordinates in mm.
        """
        if len(joint_angles_rad) != 4:
            raise ValueError("4-DOF spatial FK requires 4 joint angles.")

        q1_base, q2_shoulder, q3_elbow, q4_wrist = joint_angles_rad

        # D-H transformations
        T01 = self._dh_transformation_matrix(0, 0, self.d1, q1_base)
        T12 = self._dh_transformation_matrix(np.pi / 2, 0, 0,
                                             q2_shoulder)  # alpha is pi/2 due to Z0 to Y1 rotation convention
        T23 = self._dh_transformation_matrix(0, self.a2, 0, q3_elbow)
        T34 = self._dh_transformation_matrix(0, self.a3, 0, q4_wrist)
        T4_EE = self._dh_transformation_matrix(0, self.a4, 0, 0)  # EE offset from frame 4

        T0_EE = T01 @ T12 @ T23 @ T34 @ T4_EE

        x = T0_EE[0, 3]
        y = T0_EE[1, 3]
        z = T0_EE[2, 3]
        return x, y, z

    def get_all_joint_positions_4dof(self, joint_angles_rad):
        """
            Calculates the 3D position of each joint origin and the EE for visualization.
            Args:
                P0 = origin of frame 0 (base)
                P1 = origin of frame 1 (at J2 - shoulder)
                P2 = origin of frame 2 (at J3 - elbow)
                P3 = origin of frame 3 (at J4 - wrist)
                P_EE = end-effector point

            Return:
                p_base = T0 J1 origin list
                p_shoulder = T0 J2 origin list
                p_elbow = T0 J3 origin list
                p_wrist = T0 J4 origin list
                p_ee = T0 EE pt list
        """
        if len(joint_angles_rad) != 4:
            raise ValueError("Requires 4 joint angles.")
        q1, q2, q3, q4 = joint_angles_rad

        T00 = np.eye(4)  # Base frame
        T01 = self._dh_transformation_matrix(0, 0, self.d1, q1)
        T12 = self._dh_transformation_matrix(np.pi / 2, 0, 0, q2)
        T23 = self._dh_transformation_matrix(0, self.a2, 0, q3)
        T34 = self._dh_transformation_matrix(0, self.a3, 0, q4)
        T4_EE = self._dh_transformation_matrix(0, self.a4, 0, 0)

        T0_J1_origin = T00  # J1 (base rotation) is at origin of T00
        T0_J2_origin = T01  # J2 (shoulder) is at origin of T01
        T0_J3_origin = T01 @ T12  # J3 (elbow) is at origin of T02 (T01 @ T12)
        T0_J4_origin = T01 @ T12 @ T23  # J4 (wrist) is at origin of T03
        T0_EE_pt = T01 @ T12 @ T23 @ T34 @ T4_EE  # EE point

        p_base = T0_J1_origin[:3, 3]
        p_shoulder = T0_J2_origin[:3, 3]
        p_elbow = T0_J3_origin[:3, 3]
        p_wrist = T0_J4_origin[:3, 3]
        p_ee = T0_EE_pt[:3, 3]

        return p_base, p_shoulder, p_elbow, p_wrist, p_ee

    def visualize_arm_4dof_spatial(self, joint_angles_rad, target_pos_mm=None):
        """Visualizes the 4-DOF arm in 3D space."""
        p_base, p_shoulder, p_elbow, p_wrist, p_ee = self.get_all_joint_positions_4dof(joint_angles_rad)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot links
        xs = [p_base[0], p_shoulder[0], p_elbow[0], p_wrist[0], p_ee[0]]
        ys = [p_base[1], p_shoulder[1], p_elbow[1], p_wrist[1], p_ee[1]]
        zs = [p_base[2], p_shoulder[2], p_elbow[2], p_wrist[2], p_ee[2]]
        ax.plot(xs, ys, zs, 'o-', color='dodgerblue', linewidth=3, markersize=7, label='Arm Links')

        # Highlight joints
        ax.scatter(p_base[0], p_base[1], p_base[2], c='black', s=100, label='Base (J1 axis)')
        ax.scatter(p_shoulder[0], p_shoulder[1], p_shoulder[2], c='red', s=80, label='Shoulder (J2 axis)')
        ax.scatter(p_elbow[0], p_elbow[1], p_elbow[2], c='green', s=80, label='Elbow (J3 axis)')
        ax.scatter(p_wrist[0], p_wrist[1], p_wrist[2], c='purple', s=80, label='Wrist (J4 axis)')
        ax.scatter(p_ee[0], p_ee[1], p_ee[2], c='orange', marker='X', s=120, label='End-Effector')

        if target_pos_mm is not None:
            ax.scatter(target_pos_mm[0], target_pos_mm[1], target_pos_mm[2],
                       c='magenta', marker='*', s=150, label='Target Position')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('4-DOF Spatial Robotic Arm Configuration (D-H Based)')

        # Auto-scaling plot limits
        all_x = xs + ([target_pos_mm[0]] if target_pos_mm is not None else [])
        all_y = ys + ([target_pos_mm[1]] if target_pos_mm is not None else [])
        all_z = zs + ([target_pos_mm[2]] if target_pos_mm is not None else [])

        ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
        ax.set_ylim(min(all_y) - 20, max(all_y) + 20)
        ax.set_zlim(min(all_z) - 20, max(all_z) + 20)
        ax.legend()
        ax.view_init(elev=20., azim=-135)  # Adjust view angle
        # ax.set_aspect('equal') # May cause issues if scales are very different, use with caution
        return fig


# Example Usage
if __name__ == '__main__':
    # CRITICAL: Replace these with your arm's actual dimensions!
    fk_arm = ForwardKinematicsDH(
        d1=70.0,  # Base to Shoulder Z offset
        a2=100.0,  # Shoulder to Elbow length
        a3=100.0,  # Elbow to Wrist length
        a4=60.0  # Wrist to EE length
    )

    # Test 3-DOF Planar (fixed base rotation, e.g., 0 radians)
    q_3dof_planar = [np.deg2rad(30), np.deg2rad(45), np.deg2rad(-30)]  # shoulder, elbow, wrist_pitch
    x_3dof, y_3dof, z_3dof = fk_arm.forward_kinematics_3dof_planar(q_3dof_planar, base_rotation_rad=np.deg2rad(0))
    print(f"3-DOF Planar (base at 0 deg): EE Pos (x,y,z) = ({x_3dof:.2f}, {y_3dof:.2f}, {z_3dof:.2f}) mm")

    # Test 4-DOF Spatial
    q_4dof_spatial = [np.deg2rad(45), np.deg2rad(30), np.deg2rad(45),
                      np.deg2rad(-30)]  # base, shoulder, elbow, wrist_pitch
    x_4dof, y_4dof, z_4dof = fk_arm.forward_kinematics_4dof_spatial(q_4dof_spatial)
    print(f"4-DOF Spatial: EE Pos (x,y,z) = ({x_4dof:.2f}, {y_4dof:.2f}, {z_4dof:.2f}) mm")

    fig4dof = fk_arm.visualize_arm_4dof_spatial(q_4dof_spatial, target_pos_mm=[x_4dof, y_4dof, z_4dof])
    plt.show()