import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

class IKPyVisualizer:
    def __init__(self, link_lengths):
        self.link_lengths = np.array(link_lengths) / 1000  # Convert to meters
        self.chain = Chain(
            name="robot_arm",
            links=[
                OriginLink(),
                *[URDFLink(
                    name=f"link_{i}",
                    bounds=(-np.pi, np.pi),
                    origin_translation=[length, 0, 0],
                    origin_orientation=[0, 0, 0],
                    rotation=[0, 0, 1],
                    joint_type="revolute"
                ) for i, length in enumerate(self.link_lengths)]
            ]
        )

    def visualize(self, angles, target=None):
        ikpy_angles = [0] + list(angles)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        self.chain.plot(
            ikpy_angles,
            ax=ax,
            target=target/1000 if target is not None else None,
            show=False,
            links_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            target_color='#ff0000',
            linewidth=6
        )
        max_reach = sum(self.link_lengths)
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach * 1.2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # Add workspace boundary
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = max_reach * np.outer(np.cos(u), np.sin(v))
        y = max_reach * np.outer(np.sin(u), np.sin(v))
        z = max_reach * np.outer(np.ones(np.size(u)), np.cos(v))
        z[z < 0] = 0  # Only show upper hemisphere
        ax.plot_surface(x, y, z, color='gray', alpha=0.1)
        if target is not None:
            ax.text(target[0]/1000, target[1]/1000, target[2]/1000, f'Target\n({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) mm',
                    color='red', fontsize=10)
        return fig

