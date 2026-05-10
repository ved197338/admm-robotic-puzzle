import numpy as np
import matplotlib.pyplot as plt
import config

class PuzzleEnvironment:
    def __init__(self, seed=None):
        """
        Initializes the Puzzle Environment containing regions and labels.
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
        local label matrices for Arm 1 and Arm 2.
        """
        x_coords, y_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        # Introduce some random phase shifts to generate different patterns
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        ground_truth = np.sin(x_coords * 0.5 + phase1) + np.cos(y_coords * 0.5 + phase2)
        
        # Add random noise to create local preferences
        noise1 = np.random.normal(0, 0.5, (self.H, self.W))
        noise2 = np.random.normal(0, 0.5, (self.H, self.W))
        
        pref1 = ground_truth + noise1
        pref2 = ground_truth + noise2
        
        # Arm 1 only acts in R1, Arm 2 only acts in R2
        pref1[~self.R1_mask] = 0.0
        pref2[~self.R2_mask] = 0.0
        
        # Make their preferences heavily disagree on the shared boundary
        pref1[self.R12_mask] += 2.0
        pref2[self.R12_mask] -= 2.0
        
        return ground_truth, pref1, pref2

    def plot_global_state(self, global_labels, title="Global State"):
        """
        Plots the global puzzle state showing the values and the shared region.
        """
        fig = plt.figure(figsize=(8, 4))
        plt.imshow(global_labels, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Label Value')
        plt.title(title)
        
        # Draw boundaries for R12
        col_start = self.W // 2 - 1
        col_end = self.W // 2
        plt.axvline(x=col_start - 0.5, color='red', linestyle='--', label='R12 Boundary Start')
        plt.axvline(x=col_end + 0.5, color='red', linestyle='-', label='R12 Boundary End')
        plt.legend()
        plt.tight_layout()
        
        return fig
