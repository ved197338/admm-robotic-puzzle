# Consensus ADMM Robotic Puzzle Simulator

This repository contains a decentralized optimization simulation where two independent robotic arms must coordinate to assemble a shared puzzle boundary using **Consensus ADMM** (Alternating Direction Method of Multipliers).

## Problem Overview
- **Robotic Arm 1** controls the left half of the grid.
- **Robotic Arm 2** controls the right half of the grid.
- The arms overlap exactly in the center (Region 12).

Initially, the arms have strongly conflicting preferences for what the shared boundary should look like. Consensus ADMM forces them to negotiate a mathematical middle ground, ensuring a flawlessly consistent final global puzzle while minimizing the penalty for deviating from their original plans.

## Key Features
- **Mathematical Optimization**: Computes a fast, closed-form quadratic algebraic local update (the 'x-update') for both agents.
- **Decentralized Negotiation**: Utilizes Primal and Dual variables to track errors and mathematically penalize disagreements.
- **Machine Learning Accelerator**: Incorporates a lightweight `scikit-learn` Ridge Regressor that predicts the optimal initial consensus variable ($z$). This "psychic" prediction significantly reduces the number of ADMM iterations required to achieve convergence.
- **Visual Validation**: Dynamically generates learning curves for Primal/Dual Residuals and renders colored heatmaps of the final unified puzzle state.

## Project Structure
- `config.py`: Defines matrix dimensions, ADMM hyperparameters ($\rho$, $max\_iter$), and regional boolean jurisdiction masks.
- `puzzle_env.py`: Sets up the noisy environment, generates conflicting ground truths, and handles matplotlib rendering.
- `arm_models.py`: Encapsulates the local algebraic logic and cost functions for the individual robotic agents.
- `consensus_admm.py`: The global coordinator that executes the $z$-updates, $u$-updates, and tracks residual convergence.
- `ml_model.py`: The predictive Ridge Regression model used to warm-start the ADMM algorithm.
- `main.py`: The main execution script that benchmarks standard ADMM against ML-accelerated ADMM.
- `variable_usage.txt`: Detailed technical documentation of all codebase variables, vectors, and matrices.
- `results/plots/`: Output directory containing generated convergence charts and puzzle imagery.

## Setup & Execution

You can run the simulation using the provided automation script, which will automatically handle the setup of a localized Python virtual environment and download the necessary dependencies (`numpy`, `scikit-learn`, `matplotlib`).

```bash
# Clone the repository
git clone https://github.com/ved197338/admm-robotic-puzzle.git
cd admm-robotic-puzzle

# Give the setup script permission to run, and execute it
chmod +x run.sh
./run.sh
```

Once completed, the script will output the structural convergence logs to the terminal and save the rendered plots to the `/results` folder.
