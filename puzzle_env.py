"""
puzzle_env.py

Defines the simulation environment for the puzzle assembly.
Handles the generation of ground truth matrices, the addition of noise
to create conflicting local preferences, and plotting capabilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import config

class PuzzleEnvironment:
    """
    Simulates the state of the puzzle and the arms' local observations.
    """
    def __init__(self, seed: int = None):
        """
        Initializes the environment with specific grid masks.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        self.H = config.H
        self.W = config.W
        self.R1_mask = config.R1_mask
        self.R2_mask = config.R2_mask
        self.R12_mask = config.R12_mask
        
        if seed is not None:
            np.random.seed(seed)
        
    def make_example_labels(self):
        """
        Generates a ground truth puzzle and the corresponding noisy/biased
        local label matrices for Arm 1 and Arm 2 to simulate a disagreement.
        
        Returns:
            tuple: (ground_truth, pref1, pref2) where pref1 and pref2 are the local views.
        """
        x_coords, y_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        # Introduce random phase shifts to generate diverse continuous patterns
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        ground_truth = np.sin(x_coords * 0.5 + phase1) + np.cos(y_coords * 0.5 + phase2)
        
        # Inject Gaussian noise to simulate sensor inaccuracies or subjective preferences
        noise1 = np.random.normal(0, 0.5, (self.H, self.W))
        noise2 = np.random.normal(0, 0.5, (self.H, self.W))
        
        pref1 = ground_truth + noise1
        pref2 = ground_truth + noise2
        
        # Ensure arms only hold data for their respective assigned regions
        pref1[~self.R1_mask] = 0.0
        pref2[~self.R2_mask] = 0.0
        
        # Artificially force a heavy disagreement exactly on the shared boundary
        pref1[self.R12_mask] += 2.0
        pref2[self.R12_mask] -= 2.0
        
        return ground_truth, pref1, pref2

    def plot_global_state(self, global_labels: np.ndarray, title: str = "Global State"):
        """
        Visualizes the global puzzle state, highlighting the shared boundary.
        
        Args:
            global_labels (np.ndarray): The full H x W matrix representing the puzzle.
            title (str): Title for the plot.
            
        Returns:
            matplotlib.figure.Figure: The generated figure object.
        """
        fig = plt.figure(figsize=(8, 4))
        plt.imshow(global_labels, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Label Value')
        plt.title(title)
        
        # Demarcate the shared boundary region (R12)
        col_start = self.W // 2 - 1
        col_end = self.W // 2
        plt.axvline(x=col_start - 0.5, color='red', linestyle='--', label='Shared Boundary Start')
        plt.axvline(x=col_end + 0.5, color='red', linestyle='-', label='Shared Boundary End')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        return fig
