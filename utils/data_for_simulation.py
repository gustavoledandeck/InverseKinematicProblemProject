"""
Data Generator for Inverse Kinematics Neural Network Training

This module generates training data for neural network models by sampling
random joint configurations and calculating their corresponding end-effector positions.
"""

import numpy as np

from utils.Forward_kinematics import ForwardKinematics


class DataGenerator:
    """
        Generate training and validation data for inverse kinematics neural networks.
    """

    def __init__(self, link_lengths=None, angle_limits=None):
        """
            Initialize the data generator with link lengths and joint angle limits.

        Args:
            link_lengths (list): List of link lengths [l0, l1, l2, l3, ...] in mm
                                Default is [50, 100, 100, 100] for a typical small robotic arm
            angle_limits (list): List of tuples [(min_angle, max_angle)] for each joint in radians
                                Default limits are set to reasonable ranges for each joint
        """
        self.link_lengths = link_lengths if link_lengths is not None else [50, 100, 100, 100]

        # Default angle limits if not provided
        if angle_limits is None:
            # Format: [(min_angle_joint0, max_angle_joint0), (min_angle_joint1, max_angle_joint1), ...]
            self.angle_limits = [
                (-np.pi, np.pi),  # Base rotation (full 360 degrees)
                (-np.pi / 2, np.pi / 2),  # Shoulder joint (180 degrees)
                (-np.pi / 2, np.pi / 2),  # Elbow joint (180 degrees)
                (-np.pi / 2, np.pi / 2),  # Wrist joint (180 degrees)
                (-np.pi / 2, np.pi / 2),  # Wrist rotation (180 degrees)
                (-np.pi / 2, np.pi / 2)  # Gripper (180 degrees)
            ]
        else:
            self.angle_limits = angle_limits

        self.fk = ForwardKinematics(link_lengths=self.link_lengths)

    def generate_random_angles(self, dof):
        """
            Generate random joint angles within the specified limits.

        Args:
            dof (int): Degrees of freedom (number of joints)

        Returns:
            numpy.ndarray: Array of random joint angles
        """
        angles = np.zeros(dof)
        for i in range(dof):
            min_angle, max_angle = self.angle_limits[i]
            angles[i] = np.random.uniform(min_angle, max_angle)
        return angles

    def generate_dataset_4dof(self, num_samples):
        """
        Generate dataset for 4-DOF robotic arm (3D space).

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            tuple: (X, y) where X is end-effector positions and y is joint angles
        """
        # Initialize arrays to store data
        X = np.zeros((num_samples, 3))  # End-effector positions (x, y, z)
        y = np.zeros((num_samples, 4))  # Joint angles (θ0, θ1, θ2, θ3)

        for i in range(num_samples):
            # Generate random joint angles
            angles = self.generate_random_angles(4)
            y[i] = angles

            # Calculate forward kinematics
            x, y_pos, z = self.fk.forward_kinematics_4dof(angles)

            X[i] = [x, y_pos, z]

        return X, y

    def generate_dataset_3dof(self, num_samples):
        """
        Generate dataset for 3-DOF robotic arm (2D space).

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            tuple: (X, y) where X is end-effector positions and y is joint angles
        """
        # Initialize arrays to store data
        X = np.zeros((num_samples, 2))  # End-effector positions (x, y)
        y = np.zeros((num_samples, 3))  # Joint angles (θ0, θ1, θ2)

        for i in range(num_samples):
            # Generate random joint angles
            angles = self.generate_random_angles(3)
            y[i] = angles

            # Calculate forward kinematics
            x, y_pos = self.fk.forward_kinematics_3dof(angles)
            X[i] = [x, y_pos]

        return X, y

    def generate_grid_dataset_3dof(self, grid_size=20):
        """
        Generate a grid-based dataset for 3-DOF robotic arm to ensure coverage of workspace.

        Args:
            grid_size (int): Number of points along each dimension of the grid

        Returns:
            tuple: (X, y) where X is end-effector positions and y is joint angles
        """
        # Calculate the total arm length to determine workspace boundaries
        total_length = sum(self.link_lengths[:3])

        # Create a grid of points in the 2D workspace
        x_range = np.linspace(-total_length, total_length, grid_size)
        y_range = np.linspace(-total_length, total_length, grid_size)

        # Initialize arrays to store valid points
        X_valid = []
        y_valid = []

        # For each grid point, try to find a valid joint configuration
        for x in x_range:
            for y in y_range:
                # Check if the point is within the reachable workspace
                distance = np.sqrt(x ** 2 + y ** 2)
                if distance <= total_length and distance >= abs(self.link_lengths[0] - self.link_lengths[1] - self.link_lengths[2]):
                    # Try multiple random configurations to find one that reaches this point
                    for i in range(10):  # Try up to 10 random starting points
                        angles = self.generate_random_angles(3)

                        # Use numerical optimization to refine the angles
                        # This is a simplified approach; in practice, you might use more sophisticated IK methods
                        refined_angles = self.refine_angles_3dof(angles, [x, y], max_iterations=100)

                        # Calculate the end effector position with the refined angles
                        x_pos, y_pos = self.fk.forward_kinematics_3dof(refined_angles)

                        # Check if the refined solution is close enough to the target
                        error = np.sqrt((x - x_pos) ** 2 + (y - y_pos) ** 2)
                        if error < 1.0:  # 1mm tolerance
                            X_valid.append([x, y])
                            y_valid.append(refined_angles)
                            break

        return np.array(X_valid), np.array(y_valid)

    def refine_angles_3dof(self, initial_angles, target_position, max_iterations=100, learning_rate=0.01):
        """
            Refine joint angles to reach a target position using gradient descent.

            Args:
                initial_angles (list): Initial joint angles [θ0, θ1, θ2]
                target_position (list): Target end-effector position [x, y]
                max_iterations (int): Maximum number of iterations
                learning_rate (float): Learning rate for gradient descent

        Returns:
            numpy.ndarray: Refined joint angles
        """
        angles = np.array(initial_angles)
        target_x, target_y = target_position

        for _ in range(max_iterations):
            # Calculate current end effector position
            x, y = self.fk.forward_kinematics_3dof(angles)

            # Calculate error
            error_x = target_x - x
            error_y = target_y - y
            error = np.sqrt(error_x ** 2 + error_y ** 2)

            # If error is small enough, return the angles
            if error < 0.1:  # 0.1mm tolerance
                break

            # Calculate Jacobian numerically
            jacobian = np.zeros((2, 3))
            epsilon = 1e-6

            for j in range(3):
                # Perturb each angle slightly
                angles_perturbed = angles.copy()
                angles_perturbed[j] += epsilon

                # Calculate perturbed position
                x_perturbed, y_perturbed = self.fk.forward_kinematics_3dof(angles_perturbed)

                # Calculate partial derivatives
                jacobian[0, j] = (x_perturbed - x) / epsilon
                jacobian[1, j] = (y_perturbed - y) / epsilon

            # Calculate pseudo-inverse of Jacobian
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)

                # Update angles using the Jacobian
                delta_angles = jacobian_pinv @ np.array([error_x, error_y])
                angles += learning_rate * delta_angles

                # Ensure angles stay within limits
                for j in range(3):
                    angles[j] = np.clip(angles[j], self.angle_limits[j][0], self.angle_limits[j][1])
            except np.linalg.LinAlgError:
                # If matrix inversion fails, make a small random adjustment
                angles += np.random.uniform(-0.1, 0.1, 3)

        return angles


# Example usage
if __name__ == "__main__":
    # Create a data generator with default parameters
    data_gen = DataGenerator()

    # Generate datasets

    X_4dof, y_4dof = data_gen.generate_dataset_4dof(1000)
    X_3dof, y_3dof = data_gen.generate_dataset_3dof(1000)

    print(f"4-DOF dataset: {X_4dof.shape}, {y_4dof.shape}")
    print(f"3-DOF dataset: {X_3dof.shape}, {y_3dof.shape}")

    # Generate grid-based dataset for 3-DOF
    X_grid, y_grid = data_gen.generate_grid_dataset_3dof(grid_size=10)
    print(f"3-DOF grid dataset: {X_grid.shape}, {y_grid.shape}")
