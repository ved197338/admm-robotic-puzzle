"""
consensus_admm.py

The central orchestrator for the Consensus ADMM process.
Manages the global variables (z) and dual variables (u), and tracks convergence.
"""
import numpy as np
import config

class ConsensusADMM:
    """
    Coordinates the decentralized negotiation between the two robotic arms.
    """
    def __init__(self, arm1, arm2, shared_mask: np.ndarray):
        """
        Initializes the ADMM solver.
        
        Args:
            arm1: The first RoboticArm instance.
            arm2: The second RoboticArm instance.
            shared_mask (np.ndarray): Boolean mask indicating the overlap region.
        """
        self.arms = [arm1, arm2]
        self.shared_mask = shared_mask
        self.H, self.W = shared_mask.shape
        
        # Global consensus variable 'z' (the middle ground)
        self.z = np.zeros((self.H, self.W))
        
        # Dual variables 'u1', 'u2' (the penalty trackers for each arm)
        self.u1 = np.zeros((self.H, self.W))
        self.u2 = np.zeros((self.H, self.W))
        
        self.rho = config.rho
        
        # Tracking metrics for proof of convergence
        self.history = {
            'primal_res': [],
            'dual_res': [],
            'obj1': [],
            'obj2': [],
            'obj_total': []
        }
        
    def run(self, max_iter: int, tol: float) -> dict:
        """
        Executes the Consensus ADMM optimization loop until convergence or max_iter.
        
        Args:
            max_iter (int): Maximum number of iterations to allow.
            tol (float): Tolerance threshold for residuals to declare convergence.
            
        Returns:
            dict: The history dictionary containing tracked metrics over time.
        """
        z_prev = np.copy(self.z)
        
        for k in range(max_iter):
            # Step 1: Local Primal Updates (Each arm optimizes its own state independently)
            x1 = self.arms[0].update_local(self.z, self.u1, self.rho, self.shared_mask)
            x2 = self.arms[1].update_local(self.z, self.u2, self.rho, self.shared_mask)
            
            # Extract boundary values for consensus calculation
            x1_b = self.arms[0].get_boundary_labels(self.shared_mask)
            x2_b = self.arms[1].get_boundary_labels(self.shared_mask)
            u1_b = self.u1[self.shared_mask]
            u2_b = self.u2[self.shared_mask]
            
            # Step 2: Global Consensus Update (z)
            # Analytically derived as the average of primal and dual variables
            z_new_b = 0.5 * (x1_b + x2_b + u1_b + u2_b)
            self.z[self.shared_mask] = z_new_b
            
            # Step 3: Dual Variables Update (u)
            # Accumulate the error to penalize lack of consensus
            self.u1[self.shared_mask] = u1_b + (x1_b - z_new_b)
            self.u2[self.shared_mask] = u2_b + (x2_b - z_new_b)
            
            # Step 4: Calculate Residuals for Convergence Checking
            # Primal residual measures spatial disagreement between arms
            primal_res1 = np.linalg.norm(x1_b - z_new_b)
            primal_res2 = np.linalg.norm(x2_b - z_new_b)
            primal_res = np.sqrt(primal_res1**2 + primal_res2**2)
            
            # Dual residual measures the drift in the consensus variable
            dual_res = np.linalg.norm(self.rho * (self.z[self.shared_mask] - z_prev[self.shared_mask])) * np.sqrt(2)
            z_prev = np.copy(self.z)
            
            # Compute objectives (costs)
            obj1 = self.arms[0].local_cost(self.arms[0].x)
            obj2 = self.arms[1].local_cost(self.arms[1].x)
            
            # Record tracking metrics
            self.history['primal_res'].append(primal_res)
            self.history['dual_res'].append(dual_res)
            self.history['obj1'].append(obj1)
            self.history['obj2'].append(obj2)
            self.history['obj_total'].append(obj1 + obj2)
            
            # Check stopping criteria
            if primal_res < tol and dual_res < tol:
                print(f"[ADMM] Converged successfully at iteration {k+1}")
                print(f"[ADMM] Final Primal Residual: {primal_res:.6f} | Final Dual Residual: {dual_res:.6f}")
                break
                
        print(f"[ADMM] Final Combined Objective Value: {self.history['obj_total'][-1]:.6f}")
        return self.history
        
    def get_global_state(self) -> np.ndarray:
        """
        Stitches the disjoint arms' views and the consensus boundary into a single, seamless global matrix.
        
        Returns:
            np.ndarray: The fully assembled global puzzle state.
        """
        global_state = np.zeros((self.H, self.W))
        
        # Isolate the exclusive, non-shared regions for both arms
        r1_only = self.arms[0].region_mask & ~self.shared_mask
        r2_only = self.arms[1].region_mask & ~self.shared_mask
        
        # Populate the exclusive areas
        global_state[r1_only] = self.arms[0].x[r1_only]
        global_state[r2_only] = self.arms[1].x[r2_only]
        
        # Use the negotiated consensus variable 'z' for the shared boundary
        global_state[self.shared_mask] = self.z[self.shared_mask]
        
        return global_state
