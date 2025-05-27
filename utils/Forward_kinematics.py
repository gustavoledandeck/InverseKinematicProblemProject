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
        self.a4 = float(a4)
        self.dh_params = [
            {'theta_offset': 0, 'd': float(d1), 'a': 0, 'alpha': np.pi / 2},  # Link 1 (J1 to J2)
            {'theta_offset': 0, 'd': 0, 'a': float(a2), 'alpha': 0},  # Link 2 (J2 to J3)
            {'theta_offset': 0, 'd': 0, 'a': float(a3), 'alpha': 0},  # Link 3 (J3 to J4)
            {'theta_offset': 0, 'd': 0, 'a': float(a4), 'alpha': 0}  # Link 4 (J4 to EEP)
        ]

    def _dh_transformation_matrix(self, theta, d, a, alpha):
        """Computes a single D-H transformation matrix."""
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_alpha = np.cos(alpha)
        s_alpha = np.sin(alpha)

        T = np.array([
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0,         s_alpha,            c_alpha,                   d],
            [0,             0,                  0,                     1]
        ])
        return T

    def _calculate_T0_N(self, joint_angles_rad_full_dof):
        """
                    Helper to calculate T0_N for the full 4-DOF arm.
            Args: joint_angles_rad_full_dof: [q1, q2, q3, q4]

            Return: ndarray (T_0_N)

        """
        T_0_N = np.eye(4)
        current_angles = joint_angles_rad_full_dof

        # Iterate through the main DH links
        for i in range(len(self.dh_params)):
            params = self.dh_params[i]

            A_i = self._dh_transformation_matrix(
                current_angles[i] + params['theta_offset'],  # Theta_i = q_i + offset
                params['d'],  # d_i
                params['a'],  # a_i
                params['alpha']  # alpha_i
            )

            T_0_N = T_0_N @ A_i

        return T_0_N

    def forward_kinematics_3dof_planar(self, active_joint_angles_rad, fixed_q1_base_rotation_rad=0.0):
        """
        Calculates FK for the 3-DOF planar configuration (shoulder, elbow, wrist pitch)
        operating in a vertical plane, potentially rotated by a fixed base_rotation.

        Args:
            joint_angles_rad (list/np.array): [q2_shoulder, q3_elbow, q4_wrist_pitch] in radians.
            base_rotation_rad (float): Fixed base rotation (q1) in radians.

        Returns:
            tuple: (x, y, z) end-effector coordinates in mm.
        """

        if len(active_joint_angles_rad) != 3:
            raise ValueError("3-DOF planar FK requires 3 joint angles (for shoulder, elbow, wrist).")

        q1_base_actual = fixed_q1_base_rotation_rad
        q2_shoulder, q3_elbow, q4_wrist = active_joint_angles_rad



        T_0_N = np.eye(4)

        # Apply J1 transformation (base)
        params_j1 = self.dh_params[0]
        A1 = self._dh_transformation_matrix(
            q1_base_actual + params_j1['theta_offset'],
            params_j1['d'], params_j1['a'], params_j1['alpha']
        )
        T_0_N = T_0_N @ A1

        # Apply J2 transformation (shoulder)
        params_j2 = self.dh_params[1]
        A2 = self._dh_transformation_matrix(
            q2_shoulder + params_j2['theta_offset'],
            params_j2['d'], params_j2['a'], params_j2['alpha']
        )
        T_0_N = T_0_N @ A2

        # Apply J3 transformation (elbow)
        params_j3 = self.dh_params[2]
        A3 = self._dh_transformation_matrix(
            q3_elbow + params_j3['theta_offset'],
            params_j3['d'], params_j3['a'], params_j3['alpha']
        )
        T_0_N = T_0_N @ A3


        params_j4 = self.dh_params[3]  # This uses the 4th set of DH params
        A4_EE = self._dh_transformation_matrix(
            q4_wrist + params_j4['theta_offset'],
            params_j4['d'], params_j4['a'], params_j4['alpha']  # a4 is effectively the EE link
        )
        T_0_N = T_0_N @ A4_EE

        return T_0_N[:3, 3]  # Return x, y, z position

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
        #T01 = self._dh_transformation_matrix(0, 0, self.d1, q1_base)
        #T01 = self._dh_transformation_matrix(q1_base, self.d1, 0, 0)
        #T12 = self._dh_transformation_matrix(np.pi / 2, 0, 0, q2_shoulder)
        #T12 = self._dh_transformation_matrix(q2_shoulder, 0, 0, np.pi/2)
        #T23 = self._dh_transformation_matrix(0, self.a2, 0, q3_elbow)
        #T23 = self._dh_transformation_matrix(q3_elbow, 0, self.a2, 0)
        #T34 = self._dh_transformation_matrix(0, self.a3, 0, q4_wrist)
        #T34 = self._dh_transformation_matrix(q4_wrist, 0, self.a3, 0)
        #T4_EE = self._dh_transformation_matrix(0, self.a4, 0, 0)
        #T4_EE = self._dh_transformation_matrix(0, 0, self.a4,0)

        #T0_EE = T01 @ T12 @ T23 @ T34 @ T4_EE


        #return T0_EE[:3, 3]

        T0_EE = self._calculate_T0_N(joint_angles_rad)
        return T0_EE[:3, 3]


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
            raise ValueError("Requires 4 joint angles for get_all_joint_positions_4dof.")

        q1, q2, q3, q4 = joint_angles_rad

        P0 = np.array([0, 0, 0])  # Origin of base frame

        T_0_1 = self._dh_transformation_matrix(q1 + self.dh_params[0]['theta_offset'], self.dh_params[0]['d'],
                                               self.dh_params[0]['a'], self.dh_params[0]['alpha'])
        P1 = T_0_1[:3, 3]  # Origin of frame 1 (location of J2)

        T_1_2 = self._dh_transformation_matrix(q2 + self.dh_params[1]['theta_offset'], self.dh_params[1]['d'],
                                               self.dh_params[1]['a'], self.dh_params[1]['alpha'])
        T_0_2 = T_0_1 @ T_1_2
        P2 = T_0_2[:3, 3]  # Origin of frame 2 (location of J3)

        T_2_3 = self._dh_transformation_matrix(q3 + self.dh_params[2]['theta_offset'], self.dh_params[2]['d'],
                                               self.dh_params[2]['a'], self.dh_params[2]['alpha'])
        T_0_3 = T_0_2 @ T_2_3
        P3 = T_0_3[:3, 3]  # Origin of frame 3 (location of J4)

        T_3_4 = self._dh_transformation_matrix(q4 + self.dh_params[3]['theta_offset'], self.dh_params[3]['d'],
                                               self.dh_params[3]['a'], self.dh_params[3]['alpha'])
        T_0_4_EE = T_0_3 @ T_3_4  # This T_0_4 is actually T_0_EE if a4 is the final link to EE
        P_EE = T_0_4_EE[:3, 3]

        return P0, P1, P2, P3, P_EE

    def visualize_arm_4dof_spatial(self, joint_angles_rad, target_pos_mm=None, ax=None):
        """Visualizes the 4-DOF arm in 3D space based on calculated joint positions."""
        p0, p1, p2, p3, p_ee = self.get_all_joint_positions_4dof(joint_angles_rad)

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            created_fig = fig
        else:
            created_fig = ax.get_figure()

        # Plot links
        xs = [p0[0], p1[0], p2[0], p3[0], p_ee[0]]
        ys = [p0[1], p1[1], p2[1], p3[1], p_ee[1]]
        zs = [p0[2], p1[2], p2[2], p3[2], p_ee[2]]
        ax.plot(xs, ys, zs, 'o-', color='dodgerblue', linewidth=3, markersize=7, label='Arm Links')

        # Highlight joints
        ax.scatter(p0[0], p0[1], p0[2], c='black', s=100, label='Base Origin (J1 Axis)')
        ax.scatter(p1[0], p1[1], p1[2], c='red', s=80, label='Shoulder (J2 Axis)')
        ax.scatter(p2[0], p2[1], p2[2], c='green', s=80, label='Elbow (J3 Axis)')
        ax.scatter(p3[0], p3[1], p3[2], c='purple', s=80, label='Wrist (J4 Axis)')
        ax.scatter(p_ee[0], p_ee[1], p_ee[2], c='orange', marker='X', s=120, label='End-Effector')

        if target_pos_mm is not None:
            ax.scatter(target_pos_mm[0], target_pos_mm[1], target_pos_mm[2],
                       c='magenta', marker='*', s=150, label='Target Position')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('4-DOF Spatial Robotic Arm Configuration (D-H Based)')

        all_coords = np.array([p0, p1, p2, p3, p_ee])
        if target_pos_mm is not None:
            all_coords = np.vstack([all_coords, target_pos_mm])

        min_vals = all_coords.min(axis=0)
        max_vals = all_coords.max(axis=0)
        ranges = max_vals - min_vals
        centers = (max_vals + min_vals) / 2
        plot_radius = max(ranges) * 0.6 + 20  # 20mm padding

        ax.set_xlim(centers[0] - plot_radius, centers[0] + plot_radius)
        ax.set_ylim(centers[1] - plot_radius, centers[1] + plot_radius)
        ax.set_zlim(0, centers[2] + plot_radius if centers[
                                                       2] + plot_radius > 0 else plot_radius)  # Ensure z starts at 0 or reasonable
        ax.legend()
        ax.view_init(elev=20., azim=-135)
        return created_fig

    def jacobian_4dof_spatial(self, joint_angles_rad):
        """
        Computes the analytical Jacobian for the 4-DOF arm's EEP position.
        J = [J_p1, J_p2, J_p3, J_p4] where J_pi is a 3x1 vector.
        For a revolute joint i: J_pi = Z_{i-1} x (P_EE - P_{i-1})
        Z_{i-1} is the axis of rotation of joint i (3rd column of R_0_{i-1})
        P_EE is the end-effector position.
        P_{i-1} is the origin of frame i-1.
        """
        if len(joint_angles_rad) != 4:
            raise ValueError("Jacobian calculation requires 4 joint angles.")
        q1, q2, q3, q4 = joint_angles_rad

        J = np.zeros((3, 4))  # 3D position (x,y,z) vs 4 joint angles

        # Transformation matrices up to each joint
        T_0_0 = np.eye(4)  # Base frame
        T_0_1 = T_0_0 @ self._dh_transformation_matrix(q1 + self.dh_params[0]['theta_offset'], self.dh_params[0]['d'],
                                                       self.dh_params[0]['a'], self.dh_params[0]['alpha'])
        T_0_2 = T_0_1 @ self._dh_transformation_matrix(q2 + self.dh_params[1]['theta_offset'], self.dh_params[1]['d'],
                                                       self.dh_params[1]['a'], self.dh_params[1]['alpha'])
        T_0_3 = T_0_2 @ self._dh_transformation_matrix(q3 + self.dh_params[2]['theta_offset'], self.dh_params[2]['d'],
                                                       self.dh_params[2]['a'], self.dh_params[2]['alpha'])
        T_0_EE = T_0_3 @ self._dh_transformation_matrix(q4 + self.dh_params[3]['theta_offset'], self.dh_params[3]['d'],
                                                        self.dh_params[3]['a'], self.dh_params[3]['alpha'])

        P_EE = T_0_EE[:3, 3]

        # Z axes of rotation (3rd column of rotation matrix part)
        Z0 = T_0_0[:3, 2]  # Axis for q1
        Z1 = T_0_1[:3, 2]  # Axis for q2
        Z2 = T_0_2[:3, 2]  # Axis for q3
        Z3 = T_0_3[:3, 2]  # Axis for q4

        # Origins of frames
        P0 = T_0_0[:3, 3]  # Origin for q1
        P1 = T_0_1[:3, 3]  # Origin for q2
        P2 = T_0_2[:3, 3]  # Origin for q3
        P3 = T_0_3[:3, 3]  # Origin for q4

        # Jacobian columns for position
        J[:, 0] = np.cross(Z0, P_EE - P0)
        J[:, 1] = np.cross(Z1, P_EE - P1)
        J[:, 2] = np.cross(Z2, P_EE - P2)
        J[:, 3] = np.cross(Z3, P_EE - P3)

        return J


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