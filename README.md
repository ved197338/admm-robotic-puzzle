# Consensus ADMM Robotic Puzzle Simulator

*Project by Vedanth Vaidya and Sameer Reddy*

Hey there! This is a simple decentralized optimization project I built. It simulates two independent robotic arms trying to put together a shared puzzle boundary using **Consensus ADMM** (Alternating Direction Method of Multipliers).

## What's happening here?
- **Robotic Arm 1** is in charge of the left half of the grid.
- **Robotic Arm 2** is in charge of the right half of the grid.
- Both arms overlap right in the center (Region 12).

Initially, both arms have totally different ideas of what that overlapping boundary should look like. Consensus ADMM forces them to negotiate and find a mathematical middle ground. This makes sure the final puzzle stitches together perfectly, without forcing either arm to completely abandon their original plans.

## Some cool things included:
- **Math Optimization**: The code uses a quick closed-form quadratic update (the 'x-update') to keep things fast.
- **Decentralized Negotiation**: It tracks errors using Primal and Dual variables, penalizing the arms when they refuse to agree.
- **Machine Learning Trick**: I included a lightweight `scikit-learn` Ridge Regressor. It acts as a predictor, estimating the optimal consensus variable ($z$) upfront so the ADMM loop finishes way faster.
- **Visuals**: The script automatically generates learning curve charts for the Primal/Dual Residuals and plots some nice colored heatmaps of the final puzzle state.

## What's in the files?
- `config.py`: Just some grid dimensions, ADMM parameters like $\rho$, and the boolean masks for the regions.
- `puzzle_env.py`: Creates a noisy environment, generates some conflicting ground truths, and handles the matplotlib drawing.
- `arm_models.py`: Holds the local 'x-update' math and cost functions for each robotic agent.
- `consensus_admm.py`: The main coordinator. It handles the $z$-updates, $u$-updates, and tracks the residual convergence.
- `ml_model.py`: The Ridge Regression model used to warm-start ADMM.
- `main.py`: The main script to run everything and compare standard ADMM against the ML version.
- `variable_usage.txt`: A quick cheat sheet explaining what all the variables and matrices do.
- `results/plots/`: The folder where all the generated charts and puzzle images get saved.

## How to run it

I wrote a quick automation script so you don't have to deal with installing packages globally. It just sets up a local Python virtual environment and installs the required stuff (`numpy`, `scikit-learn`, `matplotlib`).

```bash
# Clone the repository
git clone https://github.com/ved197338/admm-robotic-puzzle.git
cd admm-robotic-puzzle

# Run the setup script
chmod +x run.sh
./run.sh
```

Once it finishes, you'll see the convergence logs printed in the terminal, and you can check out the neat plots it saves in the `/results` folder.
