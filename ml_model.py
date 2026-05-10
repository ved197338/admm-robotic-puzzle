"""
ml_model.py

A lightweight Machine Learning accelerator (Ridge Regressor).
Predicts optimal initial states for ADMM to rapidly jumpstart convergence.
"""
import numpy as np
from sklearn.linear_model import Ridge

class MLCoordinator:
    """
    Uses historical data to predict the final consensus variable, drastically
    reducing the number of required ADMM iterations.
    """
    def __init__(self):
        """
        Initializes a fast, collinearity-robust Ridge regression model.
        """
        self.model = Ridge(alpha=1.0)
        self.is_trained = False
        
    def train(self, env, n_samples: int = 50):
        """
        Trains the model by observing various conflict scenarios between the arms.
        
        Args:
            env: The PuzzleEnvironment instance for generating samples.
            n_samples (int): Number of synthetic training examples to generate.
        """
        X, y = [], []
        
        for _ in range(n_samples):
            gt, p1, p2 = env.make_example_labels()
            
            # Feature extraction: stack the conflicting boundary signals
            b1 = p1[env.R12_mask]
            b2 = p2[env.R12_mask]
            features = np.concatenate([b1, b2])
            
            # Target generation: the true, harmonious boundary signal
            target = gt[env.R12_mask]
            
            X.append(features)
            y.append(target)
            
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict_initial_z(self, p1: np.ndarray, p2: np.ndarray, env) -> np.ndarray:
        """
        Predicts an intelligent starting point for the consensus variable 'z'.
        
        Args:
            p1 (np.ndarray): Arm 1's initial preference matrix.
            p2 (np.ndarray): Arm 2's initial preference matrix.
            env: The environment containing regional masks.
            
        Returns:
            np.ndarray: A matrix pre-populated with the predicted consensus boundary.
        """
        if not self.is_trained:
            return np.zeros_like(p1)
            
        # Extract exactly what the arms are fighting over
        b1 = p1[env.R12_mask]
        b2 = p2[env.R12_mask]
        features = np.concatenate([b1, b2]).reshape(1, -1)
        
        # Predict the most likely resolution
        pred_b = self.model.predict(features)[0]
        
        # Populate and return the shared boundary
        z = np.zeros_like(p1)
        z[env.R12_mask] = pred_b
        return z
