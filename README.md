# FitzHugh-Nagumo Optimal Control Project

## Overview
This project explores optimal control strategies for the FitzHugh–Nagumo (FHN) model, a simplified version of the Hodgkin–Huxley equations commonly used to describe neuronal activity. The control objective is to influence the system's dynamics using either open-loop or feedback strategies under deterministic and stochastic settings.

It is originally based on numerical experiments from the following paper:

> **A. Hocquet and A. Vogler**, Optimal Control of Mean Field Equations with Monotone
Coefficients and Applications in Neuroscience, in *Applied Mathematics & Optimization*, **84** (Suppl 2):S1925–S1968, 2021.

The code has been refactored and extended for maintainability, clarity, and future research use, particularly focusing on:
- Clean modular design following MLOps principles,
- A new feedback control approach approximated by neural networks (under development),
- Structured numerical experiments and reproducible results.

## Project Structure
```
├── main.py               # Entry point for gradient-based optimization
├── src/                  # Core logic (state, adjoint, cost, optimization)
├── results/              # Saved control trajectories and cost values
├── plots/                # Figures and plots from simulations
├── tests/                # Unit tests (to be developed)
├── notebooks/            # Experimental and visualization notebooks
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusion rules
├── README.md             # This file
├── HV-21-AMOP-fhn.pdf    # Original paper
```

## Installation
```bash
git clone https://github.com/yourusername/FitzHugh-Nagumo-Control.git
cd FitzHugh-Nagumo-Control
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage
To run the optimization loop:
```bash
python main.py
```
Results will be saved in `results/` and plots in `plots/`.

## Acknowledgements
The original codebase was developed by **Alexander Vogler** (GitHub: `alexander19a`).

This repository is maintained and extended by **Antoine Hocquet** to include advanced structuring, modern tooling, and new control strategies.

## License
MIT License (to be confirmed).

---

> ⚙️ This repository is part of a broader research and engineering effort to bridge optimal control theory, stochastic PDEs, and modern AI tools for neuroscience-inspired models.

