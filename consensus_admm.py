import numpy as np
import config

class ConsensusADMM:
    def __init__(self, arm1, arm2, shared_mask):
        """
        Initializes the Consensus ADMM solver with two arms.
        """
        self.arms = [arm1, arm2]
        self.shared_mask = shared_mask
        
        self.H, self.W = shared_mask.shape
        
        # Shared consensus variable z
        self.z = np.zeros((self.H, self.W))
        
        # Dual variables u1, u2 for the consensus constraint
        self.u1 = np.zeros((self.H, self.W))
        self.u2 = np.zeros((self.H, self.W))
        
        self.rho = config.rho
        
        self.history = {
            'primal_res': [],
            'dual_res': [],
            'obj1': [],
            'obj2': [],
            'obj_total': []
        }
        
    def run(self, max_iter, tol):
        """
        Executes the Consensus ADMM optimization loop.
        """
        z_prev = np.copy(self.z)
        
        for k in range(max_iter):
            # 1. Local Primal Updates
            x1 = self.arms[0].update_local(self.z, self.u1, self.rho, self.shared_mask)
            x2 = self.arms[1].update_local(self.z, self.u2, self.rho, self.shared_mask)
            
            # 2. Consensus Update (z)
            x1_b = self.arms[0].get_boundary_labels(self.shared_mask)
            x2_b = self.arms[1].get_boundary_labels(self.shared_mask)
            u1_b = self.u1[self.shared_mask]
            u2_b = self.u2[self.shared_mask]
            
            # Analytical average for z
            z_new_b = 0.5 * (x1_b + x2_b + u1_b + u2_b)
            self.z[self.shared_mask] = z_new_b
            
            # 3. Dual Variables Update
            self.u1[self.shared_mask] = u1_b + (x1_b - z_new_b)
            self.u2[self.shared_mask] = u2_b + (x2_b - z_new_b)
            
            # 4. Primal and Dual Residuals
            primal_res1 = np.linalg.norm(x1_b - z_new_b)
            primal_res2 = np.linalg.norm(x2_b - z_new_b)
            primal_res = np.sqrt(primal_res1**2 + primal_res2**2)
            
            # Dual residual: || rho * (z - z_prev) ||_2 * sqrt(2)
            dual_res = np.linalg.norm(self.rho * (self.z[self.shared_mask] - z_prev[self.shared_mask])) * np.sqrt(2)
            z_prev = np.copy(self.z)
            
            # Costs
            obj1 = self.arms[0].local_cost(self.arms[0].x)
            obj2 = self.arms[1].local_cost(self.arms[1].x)
            
            self.history['primal_res'].append(primal_res)
            self.history['dual_res'].append(dual_res)
            self.history['obj1'].append(obj1)
            self.history['obj2'].append(obj2)
            self.history['obj_total'].append(obj1 + obj2)
            
            # Check stopping criteria
            if primal_res < tol and dual_res < tol:
                print(f"Converged at iteration {k+1}")
                print(f"Primal residual: {primal_res:.6f}, Dual residual: {dual_res:.6f}")
                break
                
        print(f"Final Total Objective: {self.history['obj_total'][-1]:.6f}")
        return self.history
        
    def get_global_state(self):
        """
        Stitches the arms' variables and the consensus boundary into a global matrix.
        """
        global_state = np.zeros((self.H, self.W))
        
        # Isolate the non-shared regions
        r1_only = self.arms[0].region_mask & ~self.shared_mask
        r2_only = self.arms[1].region_mask & ~self.shared_mask
        
        global_state[r1_only] = self.arms[0].x[r1_only]
        global_state[r2_only] = self.arms[1].x[r2_only]
        
        # Use z for the shared boundary
        global_state[self.shared_mask] = self.z[self.shared_mask]
        
        return global_state
