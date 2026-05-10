import numpy as np
from sklearn.linear_model import Ridge

class MLCoordinator:
    def __init__(self):
        """
        Initializes a lightweight regression model to predict the initial consensus variable z.
        """
        # Ridge regression is robust and extremely fast
        self.model = Ridge(alpha=1.0)
        self.is_trained = False
        
    def train(self, env, n_samples=50):
        """
        Trains the model to guess a good 'z' based on the discrepancy 
        between Arm 1 and Arm 2 boundary preferences.
        """
        X = []
        y = []
        for _ in range(n_samples):
            gt, p1, p2 = env.make_example_labels()
            # The features are the stacked boundary values from both arms
            b1 = p1[env.R12_mask]
            b2 = p2[env.R12_mask]
            features = np.concatenate([b1, b2])
            
            # The target is the ground truth boundary
            target = gt[env.R12_mask]
            
            X.append(features)
            y.append(target)
            
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict_initial_z(self, p1, p2, env):
        """
        Given Arm 1 and Arm 2 current preferences, predict a better initial boundary z.
        """
        if not self.is_trained:
            return np.zeros_like(p1)
            
        b1 = p1[env.R12_mask]
        b2 = p2[env.R12_mask]
        features = np.concatenate([b1, b2]).reshape(1, -1)
        
        pred_b = self.model.predict(features)[0]
        
        # Populate the shared boundary with the ML prediction
        z = np.zeros_like(p1)
        z[env.R12_mask] = pred_b
        return z
