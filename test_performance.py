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

current_script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir_perf = os.path.join(current_script_dir, "results_perf_dh_test")
os.makedirs(results_dir_perf, exist_ok=True)
trained_models_base_perf_dir = os.path.join(current_script_dir, "trained_models_perf_dh_test")
os.makedirs(trained_models_base_perf_dir, exist_ok=True)

# These MUST be accurate for your physical arm.
D1_BASE_SHOULDER_Z = 70.0
A2_SHOULDER_ELBOW = 100.0
A3_ELBOW_WRIST = 100.0
A4_WRIST_EE = 60.0
FK_DH_MODEL_GLOBAL = ForwardKinematicsDH(d1=D1_BASE_SHOULDER_Z, a2=A2_SHOULDER_ELBOW, a3=A3_ELBOW_WRIST, a4=A4_WRIST_EE)

# Define joint limits (radians) - MUST MATCH YOUR ARM
# For 3-DOF Planar (q_shoulder, q_elbow, q_wrist_pitch)
JOINT_LIMITS_3DOF_ACTIVE = [(-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180), (-np.pi / 2, np.pi / 2)]
# For 4-DOF Spatial (q_base, q_shoulder, q_elbow, q_wrist_pitch)
JOINT_LIMITS_4DOF_ACTIVE = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (0, np.pi * 150 / 180), (-np.pi / 2, np.pi / 2)]


def run_evaluation_pipeline(
        arm_dof_type,  # "3dof_planar" or "4dof_spatial"
        fk_dh_model_instance,
        active_joint_limits,
        fixed_base_rot_for_3dof=None,  # Radians, only for 3dof_planar
        train_epochs=50,
        num_master_samples=5000,
        nn_arch_config=(64, 32),
        run_nr=False,
        nr_tol=0.1,
        nr_iter=30
):
    """
    Runs the full training and evaluation pipeline for a given arm configuration.
    """
    frameworks = ['tensorflow', 'pytorch', 'sklearn']
    all_results = []

    print(f"\n\n=== EVALUATING {arm_dof_type.upper()} MODELS ===")

    # 1. Generate Master Dataset
    if arm_dof_type == "3dof_planar":
        ik_solver_prototype = InverseKinematics3DOFPlanar(
            fk_dh_model=fk_dh_model_instance,
            fixed_base_rotation_rad=fixed_base_rot_for_3dof,
            joint_angle_limits=active_joint_limits,
            model_type='tensorflow'  # Placeholder for data gen
        )
        print(f"Generating master dataset for {arm_dof_type} ({num_master_samples} samples)...")
        X_master, y_master = ik_solver_prototype.generate_training_data(num_master_samples)
    elif arm_dof_type == "4dof_spatial":
        ik_solver_prototype = InverseKinematics4DOFSpatial(
            fk_dh_model=fk_dh_model_instance,
            joint_angle_limits=active_joint_limits,
            model_type='tensorflow'  # Placeholder for data gen
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
                fixed_base_rotation_rad=fixed_base_rot_for_3dof,
                joint_angle_limits=active_joint_limits,
                model_type=framework, nn_hidden_layers=nn_arch_config
            )
        else:  # 4dof_spatial
            ik_solver = InverseKinematics4DOFSpatial(
                fk_dh_model=fk_dh_model_instance,
                joint_angle_limits=active_joint_limits,
                model_type=framework, nn_hidden_layers=nn_arch_config
            )

        model_save_dir = os.path.join(trained_models_base_perf_dir, f"{arm_dof_type}_{framework}")

        print(f"Training {arm_dof_type} {framework} model...")
        t_start_train = time.time()
        nn_history, train_cart_metrics = ik_solver.train(
            X_train_main.copy(), y_train_main.copy(), epochs=train_epochs, verbose=0  # Less verbose during perf test
        )
        time_train_s = time.time() - t_start_train
        ik_solver.save_model(model_save_dir)
        print(f"  Training complete. Internal test metrics: {train_cart_metrics}")

        # Evaluate on Global Test Set
        errors_nn_mm, times_nn_ms = [], []
        errors_nr_mm, times_nr_ms = [], []

        if X_test_main.shape[0] > 0:
            for i in range(X_test_main.shape[0]):
                target_ee_pos = X_test_main[i]

                t_s_nn = time.time()
                angles_nn = ik_solver.predict(target_ee_pos)
                times_nn_ms.append((time.time() - t_s_nn) * 1000)
                err_nn = ik_solver.verify_accuracy(target_ee_pos, angles_nn)
                errors_nn_mm.append(err_nn)

                if run_nr and hasattr(ik_solver, 'newton_raphson_minimization'):
                    t_s_nr = time.time()
                    angles_nr = ik_solver.newton_raphson_minimization(
                        target_ee_pos, angles_nn, max_iter=nr_iter, tol_mm=nr_tol
                    )
                    times_nr_ms.append((time.time() - t_s_nr) * 1000)
                    err_nr = ik_solver.verify_accuracy(target_ee_pos, angles_nr)
                    errors_nr_mm.append(err_nr)

            row = {
                'Framework': framework, 'DOF_Type': arm_dof_type, 'Train Time (s)': time_train_s,
                'Mean Infer NN (ms)': np.mean(times_nn_ms) if times_nn_ms else 0,
                'Mean Err NN (mm)': np.mean(errors_nn_mm) if errors_nn_mm else float('inf'),
                'Max Err NN (mm)': np.max(errors_nn_mm) if errors_nn_mm else float('inf'),
                'Acc NN <0.5mm (%)': np.mean([e < 0.5 for e in errors_nn_mm]) * 100 if errors_nn_mm else 0,
            }
            if run_nr and hasattr(ik_solver, 'newton_raphson_minimization'):
                row.update({
                    'Mean Infer NR (ms)': np.mean(times_nr_ms) if times_nr_ms else 0,
                    'Mean Err NR (mm)': np.mean(errors_nr_mm) if errors_nr_mm else float('inf'),
                    'Max Err NR (mm)': np.max(errors_nr_mm) if errors_nr_mm else float('inf'),
                    'Acc NR <0.5mm (%)': np.mean([e < 0.5 for e in errors_nr_mm]) * 100 if errors_nr_mm else 0,
                })
            all_results.append(row)
            print(f"  Global Test ({framework}): NN Err={row['Mean Err NN (mm)']:.3f}mm", end="")
            if run_nr and 'Mean Err NR (mm)' in row:
                print(f", NR Err={row['Mean Err NR (mm)']:.3f}mm")
            else:
                print("")  # Newline

    return pd.DataFrame(all_results)


def plot_summary_results(df_results, save_dir):
    if df_results.empty: return
    for dof_type in df_results['DOF_Type'].unique():
        df_plot = df_results[df_results['DOF_Type'] == dof_type]
        metrics_nn = ['Mean Err NN (mm)', 'Acc NN <0.5mm (%)']
        metrics_nr = ['Mean Err NR (mm)', 'Acc NR <0.5mm (%)']

        has_nr_data = all(col in df_plot.columns for col in metrics_nr)

        fig, axes = plt.subplots(1, 2 if has_nr_data else 1, figsize=(12 if has_nr_data else 7, 6), squeeze=False)
        fig.suptitle(f"Performance Summary: {dof_type.replace('_', ' ').title()}", fontsize=16)

        df_plot.set_index('Framework')[metrics_nn].plot(kind='bar', ax=axes[0, 0], rot=15, title="NN-Only Performance")
        axes[0, 0].set_ylabel("Error (mm) / Accuracy (%)")
        axes[0, 0].grid(axis='y', linestyle='--')

        if has_nr_data:
            df_plot.set_index('Framework')[metrics_nr].plot(kind='bar', ax=axes[0, 1], rot=15,
                                                            title="NN + Newton-Raphson")
            axes[0, 1].set_ylabel("Error (mm) / Accuracy (%)")
            axes[0, 1].grid(axis='y', linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"summary_plot_{dof_type}.png"))
        print(f"Plot saved: summary_plot_{dof_type}.png")
        plt.close(fig)


if __name__ == "__main__":
    print("Starting D-H Based Performance Evaluation Script...")
    # --- Parameters for the main test run ---
    EPOCHS = 1000  # Increase for better NN training (e.g., 200-1000)
    NUM_SAMPLES = 20000  # Increase for robust training (e.g., 50k-200k)

    # NN Architectures (can be tuned)
    NN_ARCH_3DOF = (64, 64, 32)
    NN_ARCH_4DOF = (128, 128, 64, 32)

    RUN_NEWTON_RAPHSON = True  # Set True to test refinement
    NR_TOLERANCE = 0.01  # Target precision for NR (mm)
    NR_MAX_ITER = 100

    all_run_results = []

    # --- 3-DOF Planar Evaluation ---
    # For planar, the base rotation is fixed. Let's test at 0 degrees.
    results_3dof = run_evaluation_pipeline(
        arm_dof_type="3dof_planar",
        fk_dh_model_instance=FK_DH_MODEL_GLOBAL,
        active_joint_limits=JOINT_LIMITS_3DOF_ACTIVE,
        fixed_base_rot_for_3dof=np.deg2rad(0.0),  # Base fixed at 0 degrees
        train_epochs=EPOCHS,
        num_master_samples=NUM_SAMPLES,
        nn_arch_config=NN_ARCH_3DOF,
        run_nr=RUN_NEWTON_RAPHSON,  # NR not typically used/needed for simpler planar
        nr_tol=NR_TOLERANCE,
        nr_iter=NR_MAX_ITER
    )
    if not results_3dof.empty:
        all_run_results.append(results_3dof)

    # --- 4-DOF Spatial Evaluation ---
    results_4dof = run_evaluation_pipeline(
        arm_dof_type="4dof_spatial",
        fk_dh_model_instance=FK_DH_MODEL_GLOBAL,
        active_joint_limits=JOINT_LIMITS_4DOF_ACTIVE,
        train_epochs=EPOCHS,
        num_master_samples=NUM_SAMPLES,
        nn_arch_config=NN_ARCH_4DOF,
        run_nr=RUN_NEWTON_RAPHSON,
        nr_tol=NR_TOLERANCE,
        nr_iter=NR_MAX_ITER
    )
    if not results_4dof.empty:
        all_run_results.append(results_4dof)

    if all_run_results:
        final_summary_df = pd.concat(all_run_results, ignore_index=True)
        print("\n\n--- FINAL SUMMARY OF ALL RUNS ---")
        print(final_summary_df.to_string())
        summary_csv_path = os.path.join(results_dir_perf, "final_performance_summary_dh.csv")
        final_summary_df.to_csv(summary_csv_path, index=False)
        print(f"Final summary CSV saved to: {summary_csv_path}")
        plot_summary_results(final_summary_df, results_dir_perf)
    else:
        print("No results generated from any pipeline.")

    print("\nD-H Based Performance Evaluation Script Finished.")
    # plt.show() # If you want to see plots interactively