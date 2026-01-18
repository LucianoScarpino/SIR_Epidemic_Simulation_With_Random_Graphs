# Homework III — Network Dynamics and Learning (HW3)

This folder contains the code and material for **Homework III** (Network Dynamics and Learning, 2025/2026).
The implementation follows the assignment requirements: **SRI dynamics simulation**, **random graph generation** and **Nash equilibrium computation** on networks.

## Project structure

### Exercise 1: SIR Epidemic Dynamics and Random Graphs
* `Ex1.py` — **Problems 1-4**
    - **Problem 1**: SIR epidemic dynamics simulation on a k-regular graph with 500 nodes and k=4 and creation of preferential attachment (PA) graph generation with varying average degrees (2-10)
    - **Problem 2**: SIR simulation on PA without vaccination (with plots of new infected and total SIR over time)
    - **Problem 3**: SIR simulation on PA graphs with progressive vaccination schedule to observe disease spread reduction (with plots of new infected and total SIR over time)
    - **Problem 4**: Parameter estimation (average degree, infection rate $\beta$, recovery rate $\rho$) using H1N1 epidemic data by minimizing RMSE with a local grid search(with plots of real vs simulated infected over time, convergence of RMSE and estimates and total SIR over time). In addition there is an estimation with $n=100$ estimation timesteps with halving of the search grid step size for finer estimation. This does not improve results significantly but increases computation time.

    Plots are saved under `Homework_3/plots/Ex1/`.


* `Ex1_problem5.py` — **Problem 5**
    - Improved parameter estimation algorithm using multiple graphs and more simulations per parameter set to reduce stochasticity (with plots of real vs simulated infected over time, convergence of RMSE and estimates and total SIR over time) and halving of the search grid step size for finer estimation with maximum number of halvings. This significantly improves results at the cost of increased computation time.
    - Tests of the algorithm on PA model to show the improvement in estimation accuracy.
    - Tests of alternative graphs (non-linear PA, Erdős-Rényi, Configuration model with truncated power law, Watts-Strogatz) to show the impact of graph structure on estimation accuracy. These all outperform the basic PA model.
    - Saves plots of convergence, SIR over time and new infected compared with the ground truth

    Plots are saved under `Homework_3/plots/Ex1_problem5/{graph_name}`, where `{graph_name}` is the name of the graph model used.

    
### Exercise 2: Network Games and Game Dynamics
* `Ex2.py` — **Problems 1-3**
    - **Problem (a1)-(a4)**: Analyze Pure Nash Equilibria for coordination and anti-coordination games with varying numbers of coordinating players ($n_1 = 0, 1, 2, 3$) and corresponding non-coordinating players ($n_2 = 3 - n_1$).
    - **Problem (b1)**: Simulate and compare Best Response (BR) dynamics vs Noisy Best Response (NBR) dynamics on a coordination game ($n_1 = 3$), visualizing transition graphs and absorption distributions.
    - **Problem (b2)**: Simulate and compare BR vs NBR dynamics on an anti-coordination game ($n_1 = 0$), visualizing transition graphs and absorption distributions.

    Plots are saved under `Homework_3/plots/Ex2/`.
    


## How to run

From the `Homework_3/` directory:

```bash
python Ex1.py
python Ex1_problem5.py
python Ex2.py
```

All figures are saved automatically in the corresponding `plots/` subfolders.

## Dependencies

Main dependencies used:

* `numpy`
* `networkx`
* `matplotlib`
* `scipy` (used in Ex1 for Gaussian PDF comparison)

Install (example):

```bash
pip install numpy networkx matplotlib scipy
```

---

**Report:**

The final PDF report is located at each related homework's root [Scarpino_Report_HW3.pdf](./Scarpino_Report_HW3.pdf)

It contains:

* Explanation of methods
* All plotted results
* Theoretical answers to questions

**Collaboration**:
S346205 Luciano Scarpino, S334015 Andrea Vasco Grieco, S329057 Shadi Mahboubpardahi, S346378 Salvatore Nocita
