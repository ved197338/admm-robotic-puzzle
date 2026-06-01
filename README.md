# Consensus ADMM Distributed Optimization

*Project by Vedanth Vaidya and Sameer Reddy*

This repository implements a decentralized optimization protocol. It coordinates two independent computing nodes over a shared grid utilizing **Consensus ADMM** (Alternating Direction Method of Multipliers).

## Project Overview
- **Node 1** processes the left partition of the grid matrix.
- **Node 2** processes the right partition of the grid matrix.
- Both nodes share an overlapping boundary region (Region 12).

Initially, both nodes initialize with strictly conflicting local preferences regarding the overlapping boundary. Consensus ADMM executes a rigorous mathematical negotiation, resolving the conflict through distributed alternating updates. This guarantees absolute convergence without necessitating a centralized processor to maintain the full global state.

## Core Implementation Features
- **Primal Optimization**: The protocol utilizes a closed-form quadratic update (the `x-update`) allowing nodes to analytically compute their local minima in $O(1)$ temporal complexity per iteration.
- **Dual Constraints**: Discrepancies at the boundary are aggressively penalized via a dual variable ($u$), preventing instability and enforcing strict mathematical consensus.
- **High-Resolution Verification**: The algorithm executes over a dense 100x200 matrix ($H=100$, $W=200$) with a strict convergence tolerance ($\epsilon = 10^{-5}$) and an Augmented Lagrangian penalty ($\rho = 1.5$).
- **Metric Exporting**: The execution pipeline automatically generates high-fidelity convergence plots (Primal/Dual Residuals and Objective Costs) and spatial heatmaps.

## Codebase Architecture
- `config.py`: Defines the matrix dimensionality, hyperparameter values ($\rho$, $tol$, $max\_iter$), and the boolean masks used to restrict node memory limits.
- `puzzle_env.py`: Generates the underlying boundary states, injects synthetic regional noise, and manages `matplotlib` rendering.
- `arm_models.py`: Encapsulates the local computation blocks (`x-update` matrices) for the individual nodes.
- `consensus_admm.py`: The central orchestrator. It manages the $z$-updates, $u$-updates, and tracks the residual convergence threshold.
- `main.py`: The primary execution script that initializes the nodes, executes the ADMM optimization cycle, and plots the tracked data.
- `variable_usage.txt`: Comprehensive documentation defining the dimensionality and functional purpose of all mathematical variables.
- `results/plots/`: Directory for exported visualization artifacts.

## Execution

A shell script is provided to automate environment initialization and dependency resolution (`numpy`, `matplotlib`).

```bash
# Clone the repository
git clone https://github.com/ved197338/admm-robotic-puzzle.git
cd admm-robotic-puzzle

# Run the setup script
chmod +x run.sh
./run.sh
```

Upon execution, the terminal will log the optimization iterations until absolute convergence is reached. Output artifacts will be saved automatically to the `/results/plots` directory.
