"""
Forward Kinematics Implementation for Robotic Arm

This module implements forward kinematics for 4-DOF and 3-DOF robotic arms.
Based on the mathematical foundations provided in literature about.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class ForwardKinematics:
    """
    Forward Kinematics implementation for robotic arms with different DOF configurations.
    """

    def __init__(self, link_lengths=None):
        """
            Initialize the FK model with link lengths

            Args:
                    link_lengths (list): List of link lengths [l0, l1, l2, l3, ...] in mm
                                                Default is [50, 100, 100, 100] for a typical small robotic arm
        """
        self.link_lengths = link_lengths if link_lengths is not None else [50, 100, 100, 100]

    def forward_kinematics_4dof(self, theta):
        """
            Calculate forward kinematics for a 4-DOF robotic arm R³ space
            Args:
                theta (list) : List of joint angles [Theta0, Theta1, Theta2, Theta3] in radians
        :return:
            tuple: (x,y,z) coordinates of the end effector
        """

        #Ensure we have 4 angles
        assert len(theta) == 4, "4-DOF forward kinematics requires 4 joint angles"

        #Extract link lengths and joint angles
        l0, l1, l2, l3 = self.link_lengths[:4]
        theta0, theta1, theta2, theta3 = theta

        #Calculate position based on the equations of l and theta
        #Base rotation
        x = (l0 * np.cos(theta0) +
                l1 * np.cos(theta0 + theta1) +
                l2 * np.cos(theta0 + theta1 + theta2) +
                l3 * np.cos(theta0 + theta1 + theta2 + theta3))
        y = (l0 * np.sin(theta0) +
                l1 * np.sin(theta0 + theta1) +
                l2 * np.sin(theta0 + theta1 + theta2) +
                l3 * np.sin(theta0 + theta1 + theta2 + theta3))
        #Calculate z based on the angles
        z = l0 + l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)

        return x, y, z

    def forward_kinematics_3dof(self, theta):
        """
            Calculate forward kinematics for a 3-DOF robotic arm R² space
            In 3-DOF we have a Planar robot with 3 revolute joints to control.
            Args:
                theta (list) : List of joint angles [Theta0, theta1, theta2] in radians
        :return:
            tuple: (x, y) coordinates of the end effector
        """
        # Ensure we have 3 angles
        assert len(theta) == 3, "3-DOF forward kinematics requires 3 joint angles"

        # Extract link lengths and joint angles
        l0, l1, l2 = self.link_lengths[:3]
        theta0, theta1, theta2 = theta

        # Calculate position in 2D plane (x, y)
        x = (l0 * np.cos(theta0) +
             l1 * np.cos(theta0 + theta1) +
             l2 * np.cos(theta0 + theta1 + theta2))

        y = (l0 * np.sin(theta0) +
             l1 * np.sin(theta0 + theta1) +
             l2 * np.sin(theta0 + theta1 + theta2))

        return x, y
    def visualize_arm_4dof(self, theta):
        """
            Visualize the 4-DOF robotic arm configuration (R³ space)
            Args:
                theta (list) : List of joint angles [theta0, theta1, theta2, theta3] in radians

        :return:
            matplotlib.figure: Figure object with the visualization
        """

        #Ensure we have 4 angles
        assert len(theta) == 4, "4-DOF visualization requires 4 joint angles"

         #Extract link lengths
        l0, l1, l2, l3 = self.link_lengths[:4]
        theta0, theta1, theta2, theta3 = theta

        # Calculate joint positions
        joint0 = [0, 0, 0]  # Base

        # First joint position after base rotation
        joint1 = [
            l0 * np.cos(theta0),
            l0 * np.sin(theta0),
            0
        ]

        # Second joint position
        joint2 = [
            joint1[0] + l1 * np.cos(theta0 + theta1),
            joint1[1] + l1 * np.sin(theta0 + theta1),
            l1 * np.sin(theta1)
        ]

        # Third joint position
        joint3 = [
            joint2[0] + l2 * np.cos(theta0 + theta1 + theta2),
            joint2[1] + l2 * np.sin(theta0 + theta1 + theta2),
            joint2[2] + l2 * np.sin(theta1 + theta2)
        ]

        # End effector position
        end_effector = [
            joint3[0] + l3 * np.cos(theta0 + theta1 + theta2 + theta3),
            joint3[1] + l3 * np.sin(theta0 + theta1 + theta2 + theta3),
            joint3[2] + l3 * np.sin(theta1 + theta2 + theta3)
        ]

        #Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        #Plot the arm segments
        joints = np.array([joint0, joint1, joint2, joint3, end_effector])
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'o-', linewidth=2, markersize=8)

        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('4-DOF Robotic Arm Configuration')

        # Set equal aspect ratio
        max_range = np.max([
            np.ptp(joints[:, 0]),
            np.ptp(joints[:, 1]),
            np.ptp(joints[:, 2])
        ])
        mid_x = np.mean([np.min(joints[:, 0]), np.max(joints[:, 0])])
        mid_y = np.mean([np.min(joints[:, 1]), np.max(joints[:, 1])])
        mid_z = np.mean([np.min(joints[:, 2]), np.max(joints[:, 2])])
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        return fig
    def visualize_arm_3dof(self, theta):
        """
            Visualize the 4-DOF robotic arm configuration (R³ space)
            Args:
                theta (list) : List of joint angles in radians
        :return:
            matplotlib.figure
        """

        #Ensure we have 3 angles
        assert len(theta) == 3, "3-DOF visualization requires 3 joint angles"

        #Extract link lengths
        l0, l1, l2 = self.link_lengths[:3]
        theta0, theta1, theta2 = theta

        #Calculate joint positions
        joint0 = [0, 0]  # Base

        #First joint position
        joint1 = [
            l0 * np.cos(theta0),
            l0 * np.sin(theta0)
        ]

        #Second joint position
        joint2 = [
            joint1[0] + l1 * np.cos(theta0 + theta1),
            joint1[1] + l1 * np.sin(theta0 + theta1)
        ]

        #End effector position
        end_effector = [
            joint2[0] + l2 * np.cos(theta0 + theta1 + theta2),
            joint2[1] + l2 * np.sin(theta0 + theta1 + theta2)
        ]

        #Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        #Plot the arm segments
        joints = np.array([joint0, joint1, joint2, end_effector])
        ax.plot(joints[:, 0], joints[:, 1], 'o-', linewidth=2, markersize=8)

        #Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('3-DOF Robotic Arm Configuration (2D)')
        ax.grid(True)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Set limits with some padding
        max_range = np.max([
            np.ptp(joints[:, 0]),
            np.ptp(joints[:, 1])
        ])
        mid_x = np.mean([np.min(joints[:, 0]), np.max(joints[:, 0])])
        mid_y = np.mean([np.min(joints[:, 1]), np.max(joints[:, 1])])
        ax.set_xlim(mid_x - max_range / 1.5, mid_x + max_range / 1.5)
        ax.set_ylim(mid_y - max_range / 1.5, mid_y + max_range / 1.5)

        return fig


if __name__ == '__main__':
    # Create a forward kinematics model with default link lengths
    fk = ForwardKinematics()

    # Test 4-DOF forward kinematics
    theta_4dof = [np.pi/4, np.pi/6, np.pi/4, np.pi/6]  # Example joint angles
    x, y, z = fk.forward_kinematics_4dof(theta_4dof)
    print(f"4-DOF End Effector Position: ({x:.2f}, {y:.2f}, {z:.2f}) mm")

    # Test 3-DOF forward kinematics
    theta_3dof = [np.pi/4, np.pi/6, np.pi/4]  # Example joint angles
    x, y = fk.forward_kinematics_3dof(theta_3dof)
    print(f"3-DOF End Effector Position: ({x:.2f}, {y:.2f}) mm")

    # Visualize the arm configurations
    fig_4dof = fk.visualize_arm_4dof(theta_4dof)
    fig_3dof = fk.visualize_arm_3dof(theta_3dof)

    plt.show()