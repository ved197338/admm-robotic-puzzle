"""
arm_models.py

Contains the local optimization logic for each robotic arm.
Implements the local x-update step of the ADMM algorithm.
"""
import numpy as np

class RoboticArm:
    """
    Represents an independent robotic arm operating on a localized region of the puzzle.
    """
    def __init__(self, region_mask: np.ndarray, local_pref: np.ndarray):
        """
        Initializes the arm with its defined active region and preferred configuration.
        
        Args:
            region_mask (np.ndarray): Boolean mask indicating the arm's active area.
            local_pref (np.ndarray): The arm's ideal local state (matrix of preferences).
        """
        self.region_mask = region_mask
        self.local_pref = local_pref
        self.H, self.W = region_mask.shape
        
        # The working state variable 'x' (starts off as the preferred state)
        self.x = np.copy(local_pref)
        
    def local_cost(self, x: np.ndarray) -> float:
        """
        Computes the quadratic penalty for deviating from the initial preference.
        Mathematically: Cost = 0.5 * || x - local_pref ||_2^2
        
        Args:
            x (np.ndarray): The current state of the arm.
            
        Returns:
            float: The scalar cost indicating structural deviation.
        """
        diff = (x - self.local_pref)[self.region_mask]
        return 0.5 * np.sum(diff**2)
        
    def update_local(self, z: np.ndarray, u: np.ndarray, rho: float, shared_mask: np.ndarray) -> np.ndarray:
        """
        Executes the local ADMM 'x-update' step. 
        Minimizes the local cost while being pulled towards the global consensus 'z'.
        
        x^(k+1) = argmin_x [ local_cost(x) + (rho/2) * || x_R12 - z + u ||_2^2 ]
        
        Args:
            z (np.ndarray): The global consensus variable.
            u (np.ndarray): The dual variable (penalty accumulator).
            rho (float): ADMM penalty parameter.
            shared_mask (np.ndarray): Boolean mask of the overlap region.
            
        Returns:
            np.ndarray: The newly updated local state 'x'.
        """
        new_x = np.copy(self.local_pref)
        
        # Extract the variables only for the shared overlap region
        pref_shared = self.local_pref[shared_mask]
        z_shared = z[shared_mask]
        u_shared = u[shared_mask]
        
        # Since the cost function is purely quadratic, we compute the closed-form algebraic solution:
        # x_shared = (pref_shared + rho * (z_shared - u_shared)) / (1 + rho)
        new_x[shared_mask] = (pref_shared + rho * (z_shared - u_shared)) / (1.0 + rho)
        
        # Zero-out anything outside the arm's assigned active region for safety
        new_x[~self.region_mask] = 0.0
        
        self.x = new_x
        return new_x
        
    def get_boundary_labels(self, shared_mask: np.ndarray) -> np.ndarray:
        """
        Extracts the arm's current state specifically over the shared boundary.
        
        Args:
            shared_mask (np.ndarray): Boolean mask for the overlap region.
            
        Returns:
            np.ndarray: The 1D array of values at the boundary.
        """
        return self.x[shared_mask]
