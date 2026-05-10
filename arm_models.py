import numpy as np

class RoboticArm:
    def __init__(self, region_mask, local_pref):
        """
        Initializes the arm with its defined region and preferred configuration.
        """
        self.region_mask = region_mask
        self.local_pref = local_pref
        self.H, self.W = region_mask.shape
        self.x = np.copy(local_pref)
        
    def local_cost(self, x):
        """
        Computes quadratic deviation from the local preference in the region.
        Cost = 0.5 * || x - local_pref ||_2^2
        """
        diff = (x - self.local_pref)[self.region_mask]
        return 0.5 * np.sum(diff**2)
        
    def update_local(self, z, u, rho, shared_mask):
        """
        Local ADMM Update:
        x^(k+1) = argmin_x [ local_cost(x) + (rho/2) * || x_R12 - z + u ||_2^2 ]
        """
        new_x = np.copy(self.local_pref)
        
        # Extract variables for the shared region
        pref_shared = self.local_pref[shared_mask]
        z_shared = z[shared_mask]
        u_shared = u[shared_mask]
        
        # Analytical solution for quadratic cost over the shared boundary:
        # x_shared = (pref_shared + rho * (z_shared - u_shared)) / (1 + rho)
        new_x[shared_mask] = (pref_shared + rho * (z_shared - u_shared)) / (1.0 + rho)
        
        # Zero-out anything outside the assigned region
        new_x[~self.region_mask] = 0.0
        
        self.x = new_x
        return new_x
        
    def get_boundary_labels(self, shared_mask):
        """
        Returns the arm's boundary labels on R12.
        """
        return self.x[shared_mask]
