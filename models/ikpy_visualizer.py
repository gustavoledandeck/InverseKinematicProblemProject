# In ikpy_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
from ikpy.chain import Chain
from ikpy.link import OriginLink, DHLink


class IKPyVisualizer:
    def __init__(self, fk_dh_model_instance, num_dof_to_visualize, realistic_joint_limits_rad):
        self.fk_model = fk_dh_model_instance
        self.num_dof = num_dof_to_visualize
        self.dh_params_from_fk = self.fk_model.dh_params  # List of param dicts

        links = [OriginLink()]

        if num_dof_to_visualize > len(self.dh_params_from_fk):
            raise ValueError("num_dof_to_visualize exceeds available DH parameters in fk_model")

        # Create active links
        for i in range(num_dof_to_visualize):
            params = self.dh_params_from_fk[i]
            min_angle, max_angle = realistic_joint_limits_rad[i]


            links.append(DHLink(
                name=f'joint_{i + 1}',
                d=params['d'] / 1000.0,
                a=params['a'] / 1000.0,
                alpha=params['alpha'],

                bounds=(min_angle, max_angle)
            ))



        if len(self.dh_params_from_fk) == 4 and num_dof_to_visualize == 4:
            # The 4 links are already added as active.
            pass
        elif len(self.dh_params_from_fk) > num_dof_to_visualize:

            for i in range(num_dof_to_visualize, len(self.dh_params_from_fk)):
                params = self.dh_params_from_fk[i]

                links.append(DHLink(
                    name=f'fixed_structural_link_{i + 1}',
                    d=params['d'] / 1000.0,
                    a=params['a'] / 1000.0,
                    alpha=params['alpha']
                ))

        active_links_mask = [False] + [True] * num_dof_to_visualize + \
                            [False] * (len(links) - 1 - num_dof_to_visualize)

        self.chain = Chain(links, active_links_mask=active_links_mask)

    def visualize(self, active_joint_angles_rad, target_pos_mm=None, ax=None):
        if len(active_joint_angles_rad) != self.num_dof:
            raise ValueError(f"Expected {self.num_dof} active joint angles, got {len(active_joint_angles_rad)}")

        ikpy_formatted_angles = [0.0] * len(self.chain.links)

        active_link_chain_indices = [i for i, active in enumerate(self.chain.active_links_mask) if active]

        if len(active_link_chain_indices) != self.num_dof:
            raise ValueError(
                f"Mismatch between active links in chain ({len(active_link_chain_indices)}) "
                f"and expected active DOFs ({self.num_dof}). Check chain construction."
            )

        # Set angles for active links, incorporating theta_offset
        for i in range(self.num_dof):
            chain_link_idx = active_link_chain_indices[i]

            angle_to_use = active_joint_angles_rad[i] + self.dh_params_from_fk[i].get('theta_offset', 0.0)
            ikpy_formatted_angles[chain_link_idx] = angle_to_use

        # Set angles for any "fixed" DH links defined after the active ones

        fixed_link_start_idx_in_fk_params = self.num_dof
        current_chain_link_idx = 1 + self.num_dof  # Start after OriginLink and active links

        for i in range(fixed_link_start_idx_in_fk_params, len(self.dh_params_from_fk)):
            if current_chain_link_idx < len(ikpy_formatted_angles) and not self.chain.active_links_mask[
                current_chain_link_idx]:
                params = self.dh_params_from_fk[i]
                ikpy_formatted_angles[current_chain_link_idx] = params.get('theta_offset', 0.0)
                current_chain_link_idx += 1
            elif current_chain_link_idx >= len(ikpy_formatted_angles):
                break

        if ax is None:
            fig_internal = plt.figure(figsize=(8, 7))
            ax_internal = fig_internal.add_subplot(111, projection='3d')
        else:
            ax_internal = ax
            fig_internal = ax.get_figure()

        target_m = np.array(target_pos_mm) / 1000.0 if target_pos_mm is not None else None

        try:
            fk_result_matrices = self.chain.forward_kinematics(ikpy_formatted_angles, full_kinematics=True)
            ee_pos_m_ikpy = fk_result_matrices[-1][:3, 3]
            self.chain.plot(ikpy_formatted_angles, ax_internal, target=target_m)
        except Exception as e:
            print(f"Error during IKPy plotting or FK: {e}")
            print(f"Chain links: {len(self.chain.links)}")
            print(f"Formatted angles: {ikpy_formatted_angles} (length {len(ikpy_formatted_angles)})")
            print(f"Active links mask: {self.chain.active_links_mask}")


            if hasattr(self.fk_model, 'visualize_arm_4dof_spatial') and self.num_dof == 4:
                print("Falling back to FK_DH_MODEL visualization due to IKPy error.")

                if self.num_dof == 4:
                    angles_for_fk_viz = active_joint_angles_rad
                elif self.num_dof == 3 and hasattr(self.fk_model,
                                                   'fixed_base_rotation_rad_for_3dof_viz'):

                    print("Fallback for 3DOF to FKDH visualizer needs fixed base angle context.")
                    return fig_internal
                else:
                    print("Cannot fallback to FKDH visualization for current DOF setup.")
                    return fig_internal

                self.fk_model.visualize_arm_4dof_spatial(angles_for_fk_viz, target_pos_mm=target_pos_mm, ax=ax_internal)
            return fig_internal

        ax_internal.set_xlabel("X (m)")
        ax_internal.set_ylabel("Y (m)")
        ax_internal.set_zlabel("Z (m)")
        ax_internal.set_title(f"{self.num_dof}-DOF Arm Visualization (IKPy)")

        all_points_m = [fk_matrix[:3, 3] for fk_matrix in fk_result_matrices]
        if target_m is not None:
            all_points_m.append(target_m)

        all_points_m_np = np.array(all_points_m)
        min_vals = all_points_m_np.min(axis=0)
        max_vals = all_points_m_np.max(axis=0)
        centers = (max_vals + min_vals) / 2
        plot_radius = np.max(max_vals - min_vals) * 0.6 + 0.05

        ax_internal.set_xlim(centers[0] - plot_radius, centers[0] + plot_radius)
        ax_internal.set_ylim(centers[1] - plot_radius, centers[1] + plot_radius)
        z_min_plot = min(0, min_vals[2] - 0.05)
        ax_internal.set_zlim(z_min_plot, max(max_vals[2] + 0.05, z_min_plot + plot_radius * 0.5))

        if target_pos_mm is not None:
            ax_internal.text(target_m[0], target_m[1], target_m[2] + 0.02,
                             f'Tgt:({target_pos_mm[0]:.0f}, {target_pos_mm[1]:.0f}, {target_pos_mm[2]:.0f})mm',
                             color='red', fontsize=8, ha='center')

        ax_internal.text(ee_pos_m_ikpy[0], ee_pos_m_ikpy[1], ee_pos_m_ikpy[2] - 0.03,
                         f'EEP:({ee_pos_m_ikpy[0] * 1000:.0f}, {ee_pos_m_ikpy[1] * 1000:.0f}, {ee_pos_m_ikpy[2] * 1000:.0f})mm',
                         color='blue', fontsize=8, ha='center')

        return fig_internal