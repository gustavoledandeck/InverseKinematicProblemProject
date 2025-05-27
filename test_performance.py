"""
Performance Testing and Evaluation for Inverse Kinematics Neural Network Solutions

This script tests and evaluates the performance of the inverse kinematics solutions
for both 4-DOF (3D) and 3-DOF (2D) configurations using different neural network frameworks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.Forward_kinematics import ForwardKinematicsDH
from inverse_kinematics_solutions import InverseKinematics3DOFPlanar, InverseKinematics4DOFSpatial
#from utils.Forward_kinematics import ForwardKinematics
#from utils.data_for_simulation import DataGenerator
#from inverse_kinematics_solutions import InverseKinematics4DOF, InverseKinematics3DOF
from models.ikpy_visualizer import IKPyVisualizer

current_script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir_perf = os.path.join(current_script_dir, "results_perf_dh_test")
os.makedirs(results_dir_perf, exist_ok=True)
trained_models_base_perf_dir = os.path.join(current_script_dir, "trained_models_perf_dh_test")
os.makedirs(trained_models_base_perf_dir, exist_ok=True)


D1_BASE_SHOULDER_Z = 40.0
A2_SHOULDER_ELBOW = 100.0
A3_ELBOW_WRIST = 130.0
A4_WRIST_EE = 30.0
FK_DH_MODEL_GLOBAL = ForwardKinematicsDH(d1=D1_BASE_SHOULDER_Z, a2=A2_SHOULDER_ELBOW, a3=A3_ELBOW_WRIST, a4=A4_WRIST_EE)

# Define joint limits (radians) - MUST MATCH YOUR ARM
# For 3-DOF Planar (q_shoulder, q_elbow, q_wrist_pitch)
#JOINT_LIMITS_3DOF_ACTIVE = [(-np.pi / 4, np.pi / 4), (0, np.pi * (120 / 180)), (-np.pi / 2, np.pi / 2)]
# For 4-DOF Spatial (q_base, q_shoulder, q_elbow, q_wrist_pitch)
#JOINT_LIMITS_4DOF_ACTIVE = [(0, np.pi * (120 / 180)), (-np.pi / 4, np.pi / 4), (0, np.pi * (120 / 180)), (-np.pi / 2, np.pi / 2)]

# Base (q1): 0 to 180 deg; Shoulder (q2): -45 to +45 deg; Elbow (q3): 0 to 120 deg; Wrist (q4): -90 to +90 deg
JOINT_LIMITS_4DOF_ACTIVE_DEG = [(0, 60), (-45, 45), (0, 120), (-90, 90)]
JOINT_LIMITS_4DOF_ACTIVE_RAD = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in JOINT_LIMITS_4DOF_ACTIVE_DEG]

# For 3-DOF Planar (e.g., q_shoulder, q_elbow, q_wrist_pitch - corresponding to J2,J3,J4 of 4-DOF structure)
JOINT_LIMITS_3DOF_ACTIVE_DEG = [(-45, 45), (0, 120), (-90, 90)] # Shoulder, Elbow, Wrist
JOINT_LIMITS_3DOF_ACTIVE_RAD = [(np.deg2rad(lim[0]), np.deg2rad(lim[1])) for lim in JOINT_LIMITS_3DOF_ACTIVE_DEG]


def run_evaluation_pipeline(
        arm_dof_type,  # "3dof_planar" or "4dof_spatial"
        fk_dh_model_instance,
        active_joint_limits_rad,  # Pass radians directly
        fixed_base_rot_for_3dof_rad=None,  # Radians, only for 3dof_planar
        train_epochs=50,
        num_master_samples=5000,
        nn_arch_config=(64, 32),
        run_nr=False,
        nr_tol_mm=0.1,  # NR tolerance in mm
        nr_max_iter=30,
        visualize_sample_count=1  # Number of test samples to visualize with IKPy
):
    """
        Runs the full training and evaluation pipeline for a given arm configuration.
    """
    frameworks = ['tensorflow', 'pytorch', 'sklearn']
    all_results = []

    # Initialize IKPy visualizer for the current DOF type
    # This visualizer will be used for a few sample plots
    ikpy_viz = None
    if arm_dof_type == "4dof_spatial":
        ikpy_viz = IKPyVisualizer(fk_dh_model_instance, num_dof_to_visualize=4,
                                  realistic_joint_limits_rad=active_joint_limits_rad)
    elif arm_dof_type == "3dof_planar":

        pass

    print(f"\n\n=== EVALUATING {arm_dof_type.upper()} MODELS ===")

    # 1. Generate Master Dataset
    if arm_dof_type == "3dof_planar":
        ik_solver_prototype = InverseKinematics3DOFPlanar(
            fk_dh_model=fk_dh_model_instance,
            fixed_base_rotation_rad=fixed_base_rot_for_3dof_rad,
            joint_angle_limits=active_joint_limits_rad,  # Pass rad
            model_type='tensorflow'
        )
        print(f"Generating master dataset for {arm_dof_type} ({num_master_samples} samples)...")
        X_master, y_master = ik_solver_prototype.generate_training_data(num_master_samples)
    elif arm_dof_type == "4dof_spatial":
        ik_solver_prototype = InverseKinematics4DOFSpatial(
            fk_dh_model=fk_dh_model_instance,
            joint_angle_limits=active_joint_limits_rad,  # Pass rad
            model_type='tensorflow'
        )
        print(f"Generating master dataset for {arm_dof_type} ({num_master_samples} samples)...")
        X_master, y_master = ik_solver_prototype.generate_training_data(num_master_samples)
    else:
        raise ValueError("Invalid arm_dof_type")

    if X_master.shape[0] == 0:
        print(f"No data generated for {arm_dof_type}. Skipping.")
        return pd.DataFrame()

    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(
        X_master, y_master, test_size=0.25, random_state=42
    )
    print(
        f"{arm_dof_type} Master Dataset: {X_master.shape[0]} samples. Using {X_test_main.shape[0]} for global testing.")

    for framework in frameworks:
        print(f"\n--- Testing {arm_dof_type} with {framework} ---")
        if arm_dof_type == "3dof_planar":
            ik_solver = InverseKinematics3DOFPlanar(
                fk_dh_model=fk_dh_model_instance,
                fixed_base_rotation_rad=fixed_base_rot_for_3dof_rad,
                joint_angle_limits=active_joint_limits_rad,  # Pass rad
                model_type=framework, nn_hidden_layers=nn_arch_config
            )
        else:  # 4dof_spatial
            ik_solver = InverseKinematics4DOFSpatial(
                fk_dh_model=fk_dh_model_instance,
                joint_angle_limits=active_joint_limits_rad,  # Pass rad
                model_type=framework, nn_hidden_layers=nn_arch_config
            )
            # Ensure NR solver is initialized for 4DOF if we plan to use it
            if run_nr and hasattr(ik_solver, 'initialize_nr_solver'):
                ik_solver.initialize_nr_solver(active_joint_limits_rad=active_joint_limits_rad)

        model_save_dir = os.path.join(trained_models_base_perf_dir, f"{arm_dof_type}_{framework}")

        print(f"Training {arm_dof_type} {framework} model...")
        t_start_train = time.time()
        nn_history, train_cart_metrics = ik_solver.train(
            X_train_main.copy(), y_train_main.copy(), epochs=train_epochs, verbose=0
        )
        time_train_s = time.time() - t_start_train
        ik_solver.save_model(model_save_dir)
        print(f"  Training complete. Internal test metrics from training: {train_cart_metrics}")

        errors_nn_mm, times_nn_ms = [], []
        errors_nr_mm, times_nr_ms = [], []  # Initialize for NR results

        if X_test_main.shape[0] > 0:
            for i in range(X_test_main.shape[0]):
                target_ee_pos_mm = X_test_main[i]  # Target EE pos in mm
                true_angles_for_context = y_test_main[i]  # For verify_accuracy if needed

                t_s_nn = time.time()
                angles_nn_rad = ik_solver.predict(target_ee_pos_mm)
                times_nn_ms.append((time.time() - t_s_nn) * 1000)
                err_nn_mm = ik_solver.verify_accuracy(target_ee_pos_mm, angles_nn_rad, true_angles_for_context)
                errors_nn_mm.append(err_nn_mm)

                # NR Refinement and Data Collection
                if run_nr and isinstance(ik_solver, InverseKinematics4DOFSpatial) and hasattr(ik_solver,
                                                                                              'refined_predict_with_nr'):
                    t_s_nr = time.time()
                    angles_nr_rad = ik_solver.refined_predict_with_nr(
                        target_ee_pos_mm, angles_nn_rad,  # Pass NN prediction as initial guess
                        max_iter=nr_max_iter, tol_mm=nr_tol_mm
                    )
                    times_nr_ms.append((time.time() - t_s_nr) * 1000)
                    err_nr_mm = ik_solver.verify_accuracy(target_ee_pos_mm, angles_nr_rad, true_angles_for_context)
                    errors_nr_mm.append(err_nr_mm)
                elif run_nr:  # If run_nr is true but not 4DOF or method missing, fill with NaN or skip
                    errors_nr_mm.append(float('nan'))
                    times_nr_ms.append(float('nan'))

                # IKPy Visualization for a few samples
                if i < visualize_sample_count and ikpy_viz is not None and arm_dof_type == "4dof_spatial":
                    print(f"  Visualizing sample {i + 1} for {framework} ({arm_dof_type})...")
                    fig_nn = ikpy_viz.visualize(angles_nn_rad, target_pos_mm=target_ee_pos_mm)
                    fig_nn.suptitle(
                        f"IKPy Viz: {framework} - NN Only\nTarget: {np.round(target_ee_pos_mm, 1)}mm, Error: {err_nn_mm:.2f}mm",
                        fontsize=10)
                    plt.savefig(os.path.join(results_dir_perf, f"viz_{arm_dof_type}_{framework}_nn_sample{i}.png"))
                    plt.close(fig_nn)

                    if run_nr and isinstance(ik_solver, InverseKinematics4DOFSpatial) and not np.isnan(err_nr_mm):
                        fig_nr = ikpy_viz.visualize(angles_nr_rad, target_pos_mm=target_ee_pos_mm)
                        fig_nr.suptitle(
                            f"IKPy Viz: {framework} - NN+NR\nTarget: {np.round(target_ee_pos_mm, 1)}mm, Error: {err_nr_mm:.2f}mm",
                            fontsize=10)
                        plt.savefig(os.path.join(results_dir_perf, f"viz_{arm_dof_type}_{framework}_nr_sample{i}.png"))
                        plt.close(fig_nr)
                elif i < visualize_sample_count and arm_dof_type == "3dof_planar":
                    # Visualization for 3DOF planar (using its own visualize_prediction)
                    if hasattr(ik_solver, 'visualize_prediction'):
                        print(f"  Visualizing sample {i + 1} for {framework} ({arm_dof_type})...")
                        # This uses the FK-based visualizer in InverseKinematics3DOFPlanar
                        fig3_nn = ik_solver.visualize_prediction(target_ee_pos_mm, angles_nn_rad)
                        plt.savefig(
                            os.path.join(results_dir_perf, f"viz_{arm_dof_type}_{framework}_nn_sample{i}_fk.png"))
                        plt.close(fig3_nn)
                        # NR not typically run/visualized for 3DOF in this setup, but could be added if refined_predict_with_nr was implemented for it

            row = {
                'Framework': framework, 'DOF_Type': arm_dof_type,
                'Train Time (s)': round(time_train_s, 2),
                'Mean Infer NN (ms)': round(np.mean(times_nn_ms) if times_nn_ms else 0, 3),
                'Mean Err NN (mm)': round(np.mean(errors_nn_mm) if errors_nn_mm else float('inf'), 3),
                'Max Err NN (mm)': round(np.max(errors_nn_mm) if errors_nn_mm else float('inf'), 3),
                'Acc NN <0.5mm (%)': round(
                    np.mean([e < 0.5 for e in errors_nn_mm if not np.isnan(e)]) * 100 if errors_nn_mm else 0, 2),
            }
            # Add NR results to the row if they were collected
            if run_nr and isinstance(ik_solver, InverseKinematics4DOFSpatial):  # Only add if NR was applicable and run
                row.update({
                    'Mean Infer NR (ms)': round(
                        np.mean(times_nr_ms) if times_nr_ms and not all(np.isnan(times_nr_ms)) else 0, 3),
                    'Mean Err NR (mm)': round(
                        np.mean(errors_nr_mm) if errors_nr_mm and not all(np.isnan(errors_nr_mm)) else float('inf'), 3),
                    'Max Err NR (mm)': round(
                        np.max(errors_nr_mm) if errors_nr_mm and not all(np.isnan(errors_nr_mm)) else float('inf'), 3),
                    'Acc NR <0.5mm (%)': round(
                        np.mean([e < 0.5 for e in errors_nr_mm if not np.isnan(e)]) * 100 if errors_nr_mm else 0, 2),
                })
            all_results.append(row)
            print(f"  Global Test ({framework}): NN Err={row['Mean Err NN (mm)']:.3f}mm", end="")
            if 'Mean Err NR (mm)' in row:  # Check if NR metrics are in the row
                print(f", NR Err={row['Mean Err NR (mm)']:.3f}mm")
            else:
                print("")

    return pd.DataFrame(all_results)


def plot_summary_results(df_results, save_dir):
    if df_results.empty:
        print("No results to plot.")
        return

    for dof_type in df_results['DOF_Type'].unique():
        df_plot = df_results[df_results['DOF_Type'] == dof_type].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Define metrics always present
        metrics_nn_error = ['Mean Err NN (mm)', 'Max Err NN (mm)']
        metrics_nn_acc = ['Acc NN <0.5mm (%)']
        metrics_nn_time = ['Mean Infer NN (ms)', 'Train Time (s)']

        # Define NR metrics - check if columns exist
        nr_error_col = 'Mean Err NR (mm)'
        nr_acc_col = 'Acc NR <0.5mm (%)'
        nr_time_col = 'Mean Infer NR (ms)'

        has_nr_data = nr_error_col in df_plot.columns and not df_plot[nr_error_col].isnull().all()

        # How many sets of bars: NN, (optional) NR
        num_metric_sets = 1
        if has_nr_data:
            num_metric_sets = 2

        # Create figure with enough subplots
        # We'll plot Error, Accuracy, and Time separately for clarity
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)  # Share X axis (Frameworks)
        fig.suptitle(f"Performance Summary: {dof_type.replace('_', ' ').title()}", fontsize=16, y=0.99)

        bar_width = 0.35
        framework_indices = np.arange(len(df_plot['Framework'].unique()))

        # --- Plot 1: Mean and Max Error (NN and optionally NR) ---
        ax_error = axes[0]
        ax_error.bar(framework_indices - bar_width / 2, df_plot['Mean Err NN (mm)'], width=bar_width,
                     label='Mean Err NN (mm)', color='skyblue')
        ax_error.bar(framework_indices - bar_width / 2, df_plot['Max Err NN (mm)'] - df_plot['Mean Err NN (mm)'],
                     bottom=df_plot['Mean Err NN (mm)'], width=bar_width, label='Max Err NN (mm) (additional)',
                     color='lightsteelblue', alpha=0.7)

        if has_nr_data:
            ax_error.bar(framework_indices + bar_width / 2, df_plot[nr_error_col], width=bar_width,
                         label='Mean Err NR (mm)', color='salmon')
            if 'Max Err NR (mm)' in df_plot.columns:
                ax_error.bar(framework_indices + bar_width / 2, df_plot['Max Err NR (mm)'] - df_plot[nr_error_col],
                             bottom=df_plot[nr_error_col], width=bar_width, label='Max Err NR (mm) (additional)',
                             color='lightcoral', alpha=0.7)

        ax_error.set_ylabel("Error (mm)")
        ax_error.set_title("Prediction Error (Lower is Better)")
        ax_error.legend(loc='upper right')
        ax_error.grid(axis='y', linestyle='--')

        # --- Plot 2: Accuracy < 0.5mm (NN and optionally NR) ---
        ax_acc = axes[1]
        ax_acc.bar(framework_indices - (bar_width / 2 if has_nr_data else 0), df_plot['Acc NN <0.5mm (%)'],
                   width=bar_width, label='Acc NN <0.5mm (%)', color='lightgreen')
        if has_nr_data:
            ax_acc.bar(framework_indices + bar_width / 2, df_plot[nr_acc_col], width=bar_width,
                       label='Acc NR <0.5mm (%)', color='palegreen')

        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy (Higher is Better)")
        ax_acc.set_ylim(0, 105)
        ax_acc.legend(loc='upper right')
        ax_acc.grid(axis='y', linestyle='--')

        # --- Plot 3: Inference and Train Time ---
        ax_time = axes[2]
        ax_time.bar(framework_indices - bar_width / 2, df_plot['Mean Infer NN (ms)'], width=bar_width,
                    label='Mean Infer NN (ms)', color='gold')
        if has_nr_data and nr_time_col in df_plot.columns:  # Check if NR time column exists
            ax_time.bar(framework_indices + bar_width / 2, df_plot[nr_time_col], width=bar_width,
                        label='Mean Infer NR (ms)', color='khaki')

        # Secondary Y-axis for Train Time (can have very different scale)
        ax_time2 = ax_time.twinx()
        ax_time2.plot(framework_indices, df_plot['Train Time (s)'], color='grey', linestyle='--', marker='o',
                      label='Train Time (s) - Right Axis')
        ax_time2.set_ylabel("Train Time (s)", color='grey')
        ax_time2.tick_params(axis='y', labelcolor='grey')

        ax_time.set_ylabel("Inference Time (ms)")
        ax_time.set_title("Computational Time (Lower is Better)")

        # Combine legends from both time axes
        lines, labels = ax_time.get_legend_handles_labels()
        lines2, labels2 = ax_time2.get_legend_handles_labels()
        ax_time.legend(lines + lines2, labels + labels2, loc='upper right')
        ax_time.grid(axis='y', linestyle='--')

        # Set X-axis labels
        ax_time.set_xticks(framework_indices)
        ax_time.set_xticklabels(df_plot['Framework'].unique(), rotation=15, ha="right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Adjust layout to make space for suptitle and x-labels
        plt.savefig(os.path.join(save_dir, f"summary_plot_{dof_type}_detailed.png"))
        print(f"Detailed plot saved: summary_plot_{dof_type}_detailed.png")
        plt.close(fig)


if __name__ == "__main__":
    print("Starting D-H Based Performance Evaluation Script...")
    # --- Parameters for the main test run ---
    EPOCHS = 500
    NUM_SAMPLES = 100000

    # NN Architectures (can be tuned)
    NN_ARCH_3DOF = (64, 64, 32)
    NN_ARCH_4DOF = (128, 128, 64, 32)

    RUN_NEWTON_RAPHSON = True  # Set True to test refinement
    NR_TOLERANCE_MM = 0.05  # Target precision for NR in mm
    NR_MAX_ITERATIONS = 50
    VISUALIZE_SAMPLES = 1  # Visualize 1 sample per framework/DOF type

    all_run_results_list = []

    # --- 3-DOF Planar Evaluation ---
    print("\n--- Running 3-DOF Planar Evaluation ---")
    results_3dof_df = run_evaluation_pipeline(
        arm_dof_type="3dof_planar",
        fk_dh_model_instance=FK_DH_MODEL_GLOBAL,
        active_joint_limits_rad=JOINT_LIMITS_3DOF_ACTIVE_RAD,
        fixed_base_rot_for_3dof_rad=np.deg2rad(0.0),
        train_epochs=EPOCHS,
        num_master_samples=NUM_SAMPLES,
        nn_arch_config=NN_ARCH_3DOF,
        run_nr=False,  # NR is primarily set up for 4DOF in InverseKinematics4DOFSpatial
        nr_tol_mm=NR_TOLERANCE_MM,
        nr_max_iter=NR_MAX_ITERATIONS,
        visualize_sample_count=VISUALIZE_SAMPLES
    )
    if not results_3dof_df.empty:
        all_run_results_list.append(results_3dof_df)

    # --- 4-DOF Spatial Evaluation ---
    print("\n--- Running 4-DOF Spatial Evaluation ---")
    results_4dof_df = run_evaluation_pipeline(
        arm_dof_type="4dof_spatial",
        fk_dh_model_instance=FK_DH_MODEL_GLOBAL,
        active_joint_limits_rad=JOINT_LIMITS_4DOF_ACTIVE_RAD,
        train_epochs=EPOCHS,
        num_master_samples=NUM_SAMPLES,
        nn_arch_config=NN_ARCH_4DOF,
        run_nr=RUN_NEWTON_RAPHSON,
        nr_tol_mm=NR_TOLERANCE_MM,
        nr_max_iter=NR_MAX_ITERATIONS,
        visualize_sample_count=VISUALIZE_SAMPLES
    )
    if not results_4dof_df.empty:
        all_run_results_list.append(results_4dof_df)

    if all_run_results_list:
        final_summary_df = pd.concat(all_run_results_list, ignore_index=True)
        print("\n\n--- FINAL SUMMARY OF ALL RUNS ---")
        print(final_summary_df.to_string())
        summary_csv_path = os.path.join(results_dir_perf, "final_performance_summary_dh.csv")
        final_summary_df.to_csv(summary_csv_path, index=False)
        print(f"Final summary CSV saved to: {summary_csv_path}")
        plot_summary_results(final_summary_df, results_dir_perf)
    else:
        print("No results generated from any pipeline.")

    print("\nD-H Based Performance Evaluation Script Finished.")