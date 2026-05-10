import numpy as np

# Grid size
H = 8
W = 16

# Masks for regions
# Arm 1 controls columns 0 to W//2 (inclusive)
R1_mask = np.zeros((H, W), dtype=bool)
R1_mask[:, :W//2 + 1] = True

# Arm 2 controls columns W//2 - 1 to W-1 (inclusive)
R2_mask = np.zeros((H, W), dtype=bool)
R2_mask[:, W//2 - 1:] = True

# The overlap (shared strip R12) is columns W//2 - 1 to W//2 (inclusive)
R12_mask = np.zeros((H, W), dtype=bool)
R12_mask[:, W//2 - 1:W//2 + 1] = True

# Consensus ADMM Hyperparameters
rho = 1.0
max_iter = 100
tol = 1e-4

# Random seed for reproducibility
SEED = 42
