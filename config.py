"""
config.py

Configuration parameters for the Consensus ADMM robotic puzzle simulation.
Defines grid dimensions, regional boolean masks for the robotic arms,
and the optimization hyperparameters.
"""
import numpy as np

# ---------------------------------------------------------
# Puzzle Dimensions
# ---------------------------------------------------------
# Height and width of the global puzzle grid
H = 8
W = 16

# ---------------------------------------------------------
# Regional Masks
# ---------------------------------------------------------
# Arm 1 controls the left half of the puzzle (columns 0 to W//2)
R1_mask = np.zeros((H, W), dtype=bool)
R1_mask[:, :W//2 + 1] = True

# Arm 2 controls the right half of the puzzle (columns W//2 - 1 to W-1)
R2_mask = np.zeros((H, W), dtype=bool)
R2_mask[:, W//2 - 1:] = True

# The shared boundary (Region 12) where the arms overlap and must negotiate
R12_mask = np.zeros((H, W), dtype=bool)
R12_mask[:, W//2 - 1:W//2 + 1] = True

# ---------------------------------------------------------
# ADMM Optimization Hyperparameters
# ---------------------------------------------------------
rho = 1.0       # Augmented Lagrangian penalty parameter
max_iter = 100  # Maximum number of ADMM iterations to prevent infinite loops
tol = 1e-4      # Convergence tolerance for primal and dual residuals

# Random seed for reproducible dataset generation
SEED = 42
