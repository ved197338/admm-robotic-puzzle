import os
import numpy as np
import matplotlib.pyplot as plt

import config
from puzzle_env import PuzzleEnvironment
from arm_models import RoboticArm
from consensus_admm import ConsensusADMM
from ml_model import MLCoordinator

def run_experiment(env, pref1, pref2, use_ml=False, ml_coordinator=None):
    """
    Runs the ADMM loop with or without ML initialization.
    """
    arm1 = RoboticArm(config.R1_mask, pref1)
    arm2 = RoboticArm(config.R2_mask, pref2)
    
    admm = ConsensusADMM(arm1, arm2, config.R12_mask)
    
    if use_ml and ml_coordinator is not None:
        initial_z = ml_coordinator.predict_initial_z(pref1, pref2, env)
        admm.z = initial_z
        
    history = admm.run(config.max_iter, config.tol)
    global_state = admm.get_global_state()
    
    return history, global_state

def main():
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Setup Environment and Ground Truth
    # Fixed seed for generating the testing environment instance
    np.random.seed(config.SEED)
    env = PuzzleEnvironment()
    _, pref1, pref2 = env.make_example_labels()
    
    print("--- Running Consensus ADMM WITHOUT ML ---")
    hist_no_ml, global_no_ml = run_experiment(env, pref1, pref2, use_ml=False)
    
    print("\n--- Training ML Model ---")
    ml_coord = MLCoordinator()
    # ML model trains on varying instances so it generalizes
    ml_coord.train(env, n_samples=50)
    
    print("\n--- Running Consensus ADMM WITH ML ---")
    # Evaluate ML run on the same fixed initial preference state
    hist_ml, global_ml = run_experiment(env, pref1, pref2, use_ml=True, ml_coordinator=ml_coord)
    
    # --- Logging / Terminal Summary ---
    print("\n=== Summary ===")
    print("Arm 1 and Arm 2 agree on the shared region; the puzzle is now consistent.")
    print(f"Iterations to converge (No ML): {len(hist_no_ml['primal_res'])}")
    print(f"Iterations to converge (With ML): {len(hist_ml['primal_res'])}")
    
    # --- Plotting ---
    
    # 1. Residual Convergence Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_no_ml['primal_res'], label='Primal Residual (No ML)', color='blue')
    plt.plot(hist_ml['primal_res'], label='Primal Residual (ML)', color='blue', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.title('Primal Residuals Convergence')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_no_ml['dual_res'], label='Dual Residual (No ML)', color='orange')
    plt.plot(hist_ml['dual_res'], label='Dual Residual (ML)', color='orange', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.title('Dual Residuals Convergence')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('results/plots/residuals.png')
    plt.close()
    
    # 2. Total Objective Value Plot
    plt.figure(figsize=(8, 5))
    plt.plot(hist_no_ml['obj_total'], label='Total Objective (No ML)', color='green')
    plt.plot(hist_ml['obj_total'], label='Total Objective (ML)', color='green', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Local Cost Sum')
    plt.title('Objective Value over Iterations')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/plots/objective.png')
    plt.close()
    
    # 3. Final Consistent Global State Visualization
    fig = env.plot_global_state(global_ml, title="Final Consistent Global Puzzle (With ML)")
    fig.savefig('results/plots/final_puzzle.png')
    plt.close(fig)
    
    # 4. Initial Disagreement State Visualization
    naive_comb = np.zeros((config.H, config.W))
    naive_comb[config.R1_mask] = pref1[config.R1_mask]
    naive_comb[config.R2_mask] = pref2[config.R2_mask]
    naive_comb[config.R12_mask] = 0.5 * (pref1[config.R12_mask] + pref2[config.R12_mask])
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(pref1, cmap='viridis', interpolation='nearest')
    plt.title("Arm 1 Local View")
    plt.colorbar(label="Label")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pref2, cmap='viridis', interpolation='nearest')
    plt.title("Arm 2 Local View")
    plt.colorbar(label="Label")
    
    plt.subplot(1, 3, 3)
    plt.imshow(naive_comb, cmap='viridis', interpolation='nearest')
    plt.title("Naive Overlay (Disagreement at R12)")
    plt.colorbar(label="Label")
    plt.axvline(x=(config.W//2 - 1) - 0.5, color='red', linestyle='--')
    plt.axvline(x=(config.W//2) + 0.5, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/plots/initial_disagreement.png')
    plt.close()
    
    print("\nVisualizations successfully saved in 'results/plots/'.")

if __name__ == '__main__':
    main()
