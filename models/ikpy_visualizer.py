import numpy as np
import matplotlib.pyplot as plt
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


class IKPyVisualizer:
    def __init__(self, link_lengths):
        """Initialize with arm segment lengths in mm"""
        self.link_lengths = np.array(link_lengths) / 1000  # Convert to meters

        # Build the kinematic chain
        self.chain = Chain(
            name="robot_arm",
            links=[
                OriginLink(),
                *[
                    URDFLink(
                        name=f"link_{i}",
                        bounds=(-np.pi, np.pi),
                        origin_translation=[length, 0, 0],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 0, 1],
                        joint_type="revolute"
                    )
                    for i, length in enumerate(self.link_lengths)
                ]
            ]
        )

    def plot_arm(self, joint_angles, target_pos=None, ax=None):
        """
        Plot arm configuration with optional target

        Args:
            joint_angles: List of angles in radians (excluding base)
            target_pos: Optional [x,y,z] target in mm
            ax: Existing matplotlib axis (optional)

        Returns:
            matplotlib.figure.Figure
        """
        # Prepend 0 for base rotation
        ikpy_angles = [0] + list(joint_angles)

        # Create figure if none provided
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        # Convert target to meters if provided
        target_meters = None
        if target_pos is not None:
            target_meters = np.array(target_pos) / 1000

        # Plot the arm
        self.chain.plot(
            ikpy_angles,
            ax=ax,
            target=target_meters,
            show=False,
            links_colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
            target_color='#d62728',
            linewidth=6
        )

        # Configure view
        max_reach = sum(self.link_lengths)
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach * 1.2])

        return fig