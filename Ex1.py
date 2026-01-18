import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from enum import IntEnum
from typing import Union, Tuple, Optional
from random import seed


# Enum for node status
class Status(IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    VACCINATED = 3


def create_k_regular_graph(n: int, k: int) -> nx.Graph:
    """Creates a k-regular undirected graph with n nodes.
    Assumes that k is even and k < n.
    Args:
        n: Number of nodes in the graph.
        k: Degree of each node (must be even and less than n).
    Returns:
        G: A NetworkX graph representing the k-regular graph.
    """

    # Sanity checks
    assert k % 2 == 0, "k must be even"
    assert k < n, "k must be less than n"

    # Create the graph
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    # Connect each node to its k/2 closest neighbors on each side (modulo n)
    # It suffices to add edges to the next k/2 nodes, as the graph is undirected
    for i in range(n):
        closest_k = [(i + j) % n for j in range(1, k // 2 + 1)]  # Next k/2 neighbors
        for j in closest_k:
            G.add_edge(i, j)  # Add edge between node i and its neighbor j
    return G


def create_pa_graph(total_nodes: int, avg_degree: int) -> nx.Graph:
    """Creates a preferential attachment graph with a given number of total nodes and average degree.
    The graph starts with a complete graph of size avg_degree + 1, and at each step a new node is added
    with avg_degree/2 edges to existing nodes chosen with probability proportional to their degree.
    If avg_degree is odd, the number of edges alternates between floor(avg_degree/2) and ceil(avg_degree/2).
    Args:
        total_nodes: Total number of nodes in the graph.
        avg_degree: Desired average degree of the graph.
    Returns:
        G: A NetworkX graph representing the preferential attachment graph.
    """
    # Initialize the graph with a complete graph of size avg_degree + 1
    G = nx.complete_graph(avg_degree + 1)
    additional_one = 0

    assert avg_degree + 1 < total_nodes, "Average degree plus 1 must be less than total nodes"

    # Add new nodes one by one
    for new_node in range(avg_degree + 1, total_nodes):
        # If avg_degree is odd, we need to alternate between floor and ceil, i.e. the integer division and integer division + 1
        # If it is even, this condition is ignored and it always sums 0 to avg_degree//2 below

        # Compute the degree distribution and probability distribution for preferential attachment
        degrees = np.array([G.degree(n) for n in G.nodes])
        prob_distribution = degrees / degrees.sum()

        # Alternate between 0 and 1 if odd (i.e. floor and ceil), else always 0
        additional_one = 1 - additional_one if avg_degree % 2 != 0 else 0

        # Select targets for the new edges based on the preferential attachment probability distribution
        targets = np.random.choice(
            list(G.nodes), size=avg_degree // 2 + additional_one, replace=False, p=prob_distribution
        )

        # Add the new edges, correcting the index of the new node
        for target in targets:
            G.add_edge(new_node, target)

    # Return the constructed graph
    return G


def plot_SIR_dynamics(
    susceptible_counts: np.ndarray,
    infected_counts: np.ndarray,
    recovered_counts: np.ndarray,
    average_new_infected: np.ndarray,
    vaccinated_counts: Optional[np.ndarray] = None,
    ground_truth_infected: Optional[np.ndarray] = None,
    xticks: Optional[list] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    graph_type: Optional[str] = None,
) -> None:
    """Plots the SIR dynamics over time.
    Args:
        susceptible_counts: Array of average susceptible counts over time.
        infected_counts: Array of average infected counts over time.
        recovered_counts: Array of average recovered counts over time.
        average_new_infected: Array of average new infected counts over time.
        vaccinated_counts: Optional array of average vaccinated counts over time.
        n_simulations: Number of simulations averaged over. Used for plot titles.
        plot_dir: Optional directory to save plots.
        graph_type: Optional string indicating the type of graph (used for plot titles and filenames).
    """

    # Create plot directory if specified
    if plot_dir:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)
        assert graph_type is not None, "graph_type must be specified if plot_dir is given"

    max_time = susceptible_counts.shape[0]  # Get number of time steps

    # Plot new infected
    plt.figure()
    plt.plot(range(max_time), average_new_infected, label="New Infected", color="orange", linewidth=2)
    if ground_truth_infected is not None:
        plt.plot(
            range(max_time),
            ground_truth_infected,
            label="Ground Truth Infected",
            color="blue",
            linestyle="--",
            linewidth=2,
        )
    plt.xlabel("Weeks", fontsize=14)
    if xticks is not None:
        plt.xticks(ticks=range(max_time), labels=xticks, fontsize=12)
    plt.ylabel("Number of new infected individuals", fontsize=14)
    plt.title(f'Average weekly new Infected ({graph_type.replace("_", " ")})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()

    # Save plot if directory is specified
    if plot_dir:
        plt.savefig(plot_dir / f"{graph_type}_new_infected.png")

    plt.clf()

    # Plot susceptible, infected, recovered and vaccinated if provided
    plt.figure()
    plt.plot(range(max_time), susceptible_counts, label="Susceptible", color="blue", linewidth=2)
    plt.plot(range(max_time), infected_counts, label="Infected", color="red", linewidth=2)
    plt.plot(range(max_time), recovered_counts, label="Recovered", color="green", linewidth=2)
    if vaccinated_counts is not None:
        plt.plot(range(max_time), vaccinated_counts, label="Vaccinated", color="purple", linewidth=2)
    plt.xlabel("Weeks", fontsize=14)
    if xticks is not None:
        plt.xticks(ticks=range(max_time), labels=xticks, fontsize=12)
    plt.ylabel("Number of individuals", fontsize=14)
    plt.title(f'Average SIR Evolution ({graph_type.replace("_", " ")})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid()
    plt.tight_layout()
    # Save plot if directory is specified
    if plot_dir:
        plt.savefig(plot_dir / f"{graph_type}_SIR_counts.png")

    plt.clf()


def plot_optimization_history(
    error_history: np.ndarray,
    k_history: np.ndarray,
    beta_history: np.ndarray,
    rho_history: np.ndarray,
    plot_dir: Optional[Union[str, Path]] = None,
    graph_type: Optional[str] = None,
) -> None:

    data = {
        "RMSE": error_history,
        "k": k_history,
        "beta": beta_history,
        "rho": rho_history,
    }
    for label, values in data.items():
        plt.figure()
        plt.plot(range(1, len(values) + 1), values, linewidth=2)
        plt.xlabel("Iteration", fontsize=14)
        param_name = rf"$\{label}$" if label not in ["k", "RMSE"] else f"${label}$" if label == "k" else "RMSE"
        plt.ylabel(param_name, fontsize=14)
        plt.title(rf"Value of {param_name} over Optimization Iterations", fontsize=16)
        plt.grid()
        plt.locator_params(axis="x", integer=True)

        plt.tight_layout()

        # Save plot if directory is specified
        if plot_dir:
            assert graph_type is not None, "graph_type must be specified if plot_dir is given"
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(exist_ok=True)
            plt.savefig(plot_dir / f"{graph_type}_{label.lower()}_optimization_history.png")

        plt.clf()


def simulate_SIR_dynamics(
    G: nx.Graph,
    beta: float,
    rho: float,
    n_initial_infected: int,
    max_time: int,
    vaccinations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulates the SIR dynamics on graph G.
    Args:
        G: The contact network as a NetworkX graph.
        beta: Infection probability per contact.
        rho: Recovery probability per time step.
        n_initial_infected: Number of initially infected individuals.
        max_time: Number of time steps to simulate.
        vaccinations: Optional array of length max_time indicating the cumulative percentage of vaccinated individuals at each time step.
    Returns:
        configuration_history: A (n, max_time + 1) array where each row corresponds to a node and each column to a time step,
                               with values indicating the status (0: susceptible, 1: infected, 2: recovered, 3: vaccinated).
    """

    if vaccinations is None:
        vaccinations = np.zeros(max_time)

    n = G.number_of_nodes()
    assert len(vaccinations) == max_time, "Average vaccinations must be the same as max_time"

    # Store the configuration at each time step, starting with all susceptible
    current_configuration = np.full(n, Status.SUSCEPTIBLE, dtype=int)

    # Set initial infected nodes, randomly chosen
    initial_infected = np.random.choice(n, n_initial_infected, replace=False)
    current_configuration[initial_infected] = Status.INFECTED

    # Allocate history array and set initial state. The history has shape (n, max_time + 1)
    # because we store the initial configuration at time 0
    configuration_history = np.empty((n, max_time + 1), dtype=int)
    configuration_history[:, 0] = current_configuration.copy()

    # Store sparse adjacency matrix for efficient neighbor lookups
    adj_matrix = nx.to_scipy_sparse_array(G)

    # Compute the increase in vaccination ratios for each time step
    vaccinations_ratios = np.diff(vaccinations, prepend=0) / 100

    for t in range(max_time):

        # Apply vaccinations to current configuration
        n_new_vaccinations = int(vaccinations_ratios[t] * n)  # Number of new vaccinations at time t

        # Select new vaccinations from those not already vaccinated
        new_vaccinations = np.random.choice(
            np.argwhere(current_configuration != Status.VACCINATED).flatten(), n_new_vaccinations, replace=False
        )

        # Update current configuration with new vaccinations
        current_configuration[new_vaccinations] = Status.VACCINATED

        # Compute new infections, by looking at infected neighbors
        infected_neighbors = adj_matrix @ (current_configuration == Status.INFECTED)
        new_infections_prob = 1 - (1 - beta) ** infected_neighbors

        # Ensure infections are applied only to susceptible and non-vaccinated individuals
        new_infections = (current_configuration == Status.SUSCEPTIBLE) & (np.random.rand(n) < new_infections_prob)

        # Compute recoveries, only among infected and non-vaccinated individuals
        recoveries = (current_configuration == Status.INFECTED) & (np.random.rand(n) < rho)

        # Update current configuration with new infections and recoveries
        current_configuration[new_infections] = Status.INFECTED
        current_configuration[recoveries] = Status.RECOVERED

        # Store the current configuration in history
        configuration_history[:, t + 1] = current_configuration.copy()

    # Return the full history of configurations
    return configuration_history


def test_SIR_simulation(
    G: nx.Graph,
    beta: float,
    rho: float,
    n_initial_infected: int,
    max_time: int,
    n_simulations: int,
    vaccinations: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """Tests the SIR simulation by running multiple simulations and averaging the results.
    Args:
        G: The contact network as a NetworkX graph.
        beta: Infection probability rate per contact.
        rho: Recovery probability rate per time step.
        n_initial_infected: Number of initially infected individuals.
        max_time: Number of time steps to simulate.
        n_simulations: Number of simulations to run for averaging.
        vaccinations: Optional array of length max_time indicating the cumulative percentage of vaccinated individuals at each time step.
    Returns:
        susceptible_counts: Array of average susceptible counts over time.
        infected_counts: Array of average infected counts over time.
        recovered_counts: Array of average recovered counts over time.
        average_new_infected: Array of average new infected counts over time.
        vaccinated_counts: Optional array of average vaccinated counts over time.
    """
    # Initialize arrays to accumulate counts
    susceptible_counts = np.zeros(max_time)
    infected_counts = np.zeros(max_time)
    recovered_counts = np.zeros(max_time)
    average_new_infected = np.zeros(max_time)
    vaccinated_counts = np.zeros(max_time) if vaccinations is not None else None  # Initialize if needed

    # Run multiple simulations and accumulate results. Dividing by n_simulations has the same effect as averaging at the end.
    for _ in range(n_simulations):
        history = simulate_SIR_dynamics(G, beta, rho, n_initial_infected, max_time, vaccinations=vaccinations)
        susceptible_counts += np.sum(history[:, 1:] == Status.SUSCEPTIBLE, axis=0) / n_simulations
        infected_counts += np.sum(history[:, 1:] == Status.INFECTED, axis=0) / n_simulations
        recovered_counts += np.sum(history[:, 1:] == Status.RECOVERED, axis=0) / n_simulations

        # Accumulate vaccinated counts if applicable
        if vaccinations is not None:
            vaccinated_counts += np.sum(history[:, 1:] == Status.VACCINATED, axis=0) / n_simulations

        # Compute new infected counts and their average
        new_infected = ((history[:, 1:] == Status.INFECTED) & (history[:, :-1] != Status.INFECTED)).sum(axis=0)
        average_new_infected += new_infected / n_simulations

    # Return the averaged results
    return susceptible_counts, infected_counts, recovered_counts, average_new_infected, vaccinated_counts


def estimate_params(
    n_nodes: int,
    vaccinations: np.ndarray,
    ground_truth_infected: np.ndarray,
    max_time: int,
    n_simulations: int,
    k: int,
    beta: float,
    rho: float,
    delta_k: int,
    delta_beta: float,
    delta_rho: float,
    halving_times: int,
    max_iters: int,
) -> Tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the parameters k, beta, and rho by minimizing the error between simulated and ground truth infected counts.
    Args:
        n_nodes: Number of nodes in the graph.
        vaccinations: Array of cumulative percentage of vaccinated individuals at each time step.
        ground_truth_infected: Array of ground truth infected counts over time.
        max_time: Number of time steps to simulate.
        n_simulations: Number of simulations to run for averaging.
        k: Initial guess for average degree k.
        beta: Initial guess for infection probability beta.
        rho: Initial guess for recovery probability rho.
        delta_k: Initial step size for k.
        delta_beta: Initial step size for beta.
        delta_rho: Initial step size for rho.
        halving_times: Number of times to halve the step sizes before stopping.
        max_iters: Maximum number of iterations to perform.
    Returns:
        k: Estimated average degree k.
        beta: Estimated infection probability beta.
        rho: Estimated recovery probability rho.
        error_history: Array of errors at each iteration.
        k_history: Array of k values at each iteration.
        beta_history: Array of beta values at each iteration.
        rho_history: Array of rho values at each iteration.
    """
    k_prev = beta_prev = rho_prev = None  # Initialize previous parameters for convergence check
    finished = False  # Flag to indicate when optimization is finished
    error_history = []  # Store list of errors for each iteration
    iter_count = 0  # Iteration counter, used for max_iters check
    count_halvings = 0  # Count how many times step sizes have been halved

    k_history = []  # Optional: store history of k values
    beta_history = []  # Optional: store history of beta values
    rho_history = []  # Optional: store history of rho values

    print("Initial error")
    initial_G = create_pa_graph(total_nodes=n_nodes, avg_degree=int(k))
    _, _, _, new_infected, _ = test_SIR_simulation(
        initial_G, beta, rho, ground_truth_infected[0], max_time, n_simulations, vaccinations=vaccinations
    )
    initial_error = np.mean((new_infected - ground_truth_infected[1:]) ** 2)
    initial_error = np.sqrt(initial_error)
    print("Initial error:", initial_error)

    # Initial logging
    print("\n--------- Parameter estimation ---------\n")
    print("Optimization parameters:")
    print(f"Initial k: {k} +- {delta_k}\nInitial beta: {beta} +- {delta_beta}\nInitial rho: {rho} +- {delta_rho}")
    print(f"Number of halvings: {halving_times}\nNumber of simulations per evaluation: {n_simulations}")
    print(f"Maximum iterations: {max_iters}")
    print("Starting optimization...\n")

    # Optimization loop
    while not finished:
        # Increment iteration counter
        iter_count += 1

        # Store previous best parameters
        k_prev = k
        beta_prev = beta
        rho_prev = rho

        # Initialize smallest error as infinity
        smallest_error = float("inf")

        # Grid search over the parameter space defined by current estimates and step sizes
        for test_k in [k - delta_k, k, k + delta_k]:
            for test_beta in [beta - delta_beta, beta, beta + delta_beta]:
                for test_rho in [rho - delta_rho, rho, rho + delta_rho]:

                    # Safety checks
                    if test_beta < 0 or test_beta > 1 or test_rho < 0 or test_rho > 1 or test_k < 1:
                        continue

                    # Create PA graph with test_k average degree
                    G = create_pa_graph(total_nodes=n_nodes, avg_degree=int(test_k))

                    # Simulate SIR dynamics and compute RMSE with ground truth
                    n_initial_infected = ground_truth_infected[0]
                    _, _, _, new_infected, _ = test_SIR_simulation(
                        G, test_beta, test_rho, n_initial_infected, max_time, n_simulations, vaccinations=vaccinations
                    )

                    # Compute the Root Mean Square Error (RMSE)
                    error = np.mean((new_infected - ground_truth_infected[1:]) ** 2)
                    error = np.sqrt(error)

                    # Update smallest error and best parameters if current error is smaller
                    if error < smallest_error:
                        smallest_error = error
                        k = test_k
                        beta = test_beta
                        rho = test_rho

        # Append the smallest error of this iteration to the history
        error_history.append(smallest_error)
        k_history.append(k)
        beta_history.append(beta)
        rho_history.append(rho)

        # If the parameters did not change, halve the step sizes
        if (k == k_prev) and (beta == beta_prev) and (rho == rho_prev):
            count_halvings += 1
            if count_halvings > halving_times:
                finished = True
            delta_k = max(1, delta_k // 2)
            delta_beta /= 2
            delta_rho /= 2

        # Check stopping criteria
        if count_halvings > halving_times or iter_count >= max_iters:
            finished = True

            # This is to notify if the optimization ended due to reaching max iterations, not convergence
            if iter_count >= max_iters:
                print("[WARNING] Maximum number of iterations reached.\n")

            # Final logging
            print("\n--------- Optimization Finished ---------")
            print(f"Final estimates: k={k}, beta={beta:.3f}, rho={rho:.3f}, error={smallest_error:.3f}\n")

        else:
            # Log current best values
            print(
                f"(Iteration {iter_count}) - Current estimates: k={k}, beta={beta:.3f}, rho={rho:.3f}, error={smallest_error:.3f}"
            )

    # Return the final estimates and histories
    return k, beta, rho, np.array(error_history), np.array(k_history), np.array(beta_history), np.array(rho_history)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    seed(42)

    # Common simulation parameters
    simulation_parameters = {
        "beta": 0.3,
        "rho": 0.7,
        "n_initial_infected": 10,
        "max_time": 15,
        "n_simulations": 100,
    }

    plot_dir = Path(__file__).parent / "plots" / "Ex1"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # --------------------------------------------------------------------------------
    #  Exercise 1.1.1: k-regular graph SIR simulation
    # --------------------------------------------------------------------------------

    # Create a 4-regular graph with 500 nodes, perform simulation and plot results
    G = create_k_regular_graph(n=500, k=4)
    simulation_results = test_SIR_simulation(G, **simulation_parameters)
    plot_SIR_dynamics(
        *simulation_results,
        plot_dir=plot_dir,
        graph_type="k_regular",
        xticks=[i + 1 for i in range(simulation_parameters["max_time"])],
    )

    # --------------------------------------------------------------------------------
    #  Exercise 1.1.2: Preferential attachment graph SIR simulation
    # --------------------------------------------------------------------------------

    print("\n --- Exercise 1.1.2: Creating PA graphs with varying average degrees ---")

    # For each k in 2 to 10, create a PA graph with 500 nodes and average degree k, and print its average degree
    for k in range(2, 10, 1):
        G = create_pa_graph(total_nodes=500, avg_degree=k)
        degrees = nx.adjacency_matrix(G).sum(axis=1)  # Get degree of each node
        avg_deg = sum(degrees) / len(degrees)  # Compute average degree
        print(f"Created PA graph with avg degree {avg_deg} and {G.number_of_edges()} edges.")

    # --------------------------------------------------------------------------------
    #  Exercise 1.2: Preferential attachment graph SIR simulation
    # --------------------------------------------------------------------------------

    # Create a PA graph with 500 nodes and average degree 6, perform simulation and plot results
    G_pa = create_pa_graph(total_nodes=500, avg_degree=6)
    simulation_results = test_SIR_simulation(G_pa, **simulation_parameters)
    plot_SIR_dynamics(
        *simulation_results,
        plot_dir=plot_dir,
        graph_type="PA_no_vaccination",
    )

    print("Maximum degree in PA graph:", max(dict(G_pa.degree()).values()))
    print(
        "Total people infected during simulation (no vaccination):",
        np.sum(simulation_results[3]) + simulation_parameters["n_initial_infected"],
    )

    # --------------------------------------------------------------------------------
    #  Exercise 1.3: Preferential attachment graph SIR simulation with vaccination
    # --------------------------------------------------------------------------------

    # Create a PA graph with 500 nodes and average degree 6
    G_pa = create_pa_graph(total_nodes=500, avg_degree=6)

    # Define a vaccination schedule (percentage of population vaccinated at each time step)
    vaccinations = np.array([0, 5, 15, 25, 35, 45, 55, 60, 60, 60, 60, 60, 60, 60, 60])
    simulation_parameters_with_vaccination = {
        **simulation_parameters,
        "vaccinations": vaccinations,
    }

    # Run simulation with vaccination
    simulation_results = test_SIR_simulation(G_pa, **simulation_parameters_with_vaccination)
    plot_SIR_dynamics(
        *simulation_results,
        plot_dir=plot_dir,
        graph_type="PA_with_vaccination",
    )
    print(
        "Total people infected during simulation (with vaccination):",
        np.sum(simulation_results[3]) + simulation_parameters_with_vaccination["n_initial_infected"],
    )

    # --------------------------------------------------------------------------------
    #  Exercise 1.4: Parameter estimation for H1N1 epidemic in a PA graph
    # --------------------------------------------------------------------------------

    # Given data from the H1N1 epidemic in a population of 934 individuals
    vaccinations = np.array([5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60])
    ground_truth_infected = np.array([1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0])
    population_size = 934

    # Parameters for estimation
    parameters_for_estimation = {
        "n_nodes": population_size,
        "vaccinations": vaccinations,
        "ground_truth_infected": ground_truth_infected,
        "max_time": 15,
        "n_simulations": 10,
        "k": 10,
        "beta": 0.3,
        "rho": 0.6,
        "delta_k": 1,
        "delta_beta": 0.1,
        "delta_rho": 0.1,
        "halving_times": 0,
        "max_iters": 100,
    }

    # Run parameter estimation
    avg_degree, beta, rho, error_history, k_history, beta_history, rho_history = estimate_params(
        **parameters_for_estimation
    )

    plot_optimization_history(
        error_history,
        k_history,
        beta_history,
        rho_history,
        plot_dir=plot_dir,
        graph_type="PA_estimated",
    )
    # Test initial error before estimation
    n_graphs = 30
    new_infections = np.zeros(parameters_for_estimation["max_time"])
    print(f"Evaluating initial parameters on {n_graphs} graphs and 1000 simulations per graph...")
    print(
        f"Initial parameters: k={parameters_for_estimation['k']}, beta={parameters_for_estimation['beta']:.3f}, rho={parameters_for_estimation['rho']:.3f}"
    )
    for _ in range(n_graphs):
        G_initial = create_pa_graph(total_nodes=population_size, avg_degree=int(parameters_for_estimation["k"]))
        simulation_results = test_SIR_simulation(
            G_initial,
            parameters_for_estimation["beta"],
            parameters_for_estimation["rho"],
            ground_truth_infected[0],
            15,
            1000,
            vaccinations=vaccinations,
        )
        new_infections += simulation_results[3] / n_graphs
    RMSE = np.sqrt(np.mean((new_infections - ground_truth_infected[1:]) ** 2))
    print(f"Initial RMSE before estimation: {RMSE:.3f}")

    # Compare the final simulation with estimated parameters to ground truth on 30 graphs
    susceptible = np.zeros(parameters_for_estimation["max_time"])
    infected = np.zeros(parameters_for_estimation["max_time"])
    recovered = np.zeros(parameters_for_estimation["max_time"])
    vaccinated = np.zeros(parameters_for_estimation["max_time"])
    new_infections = np.zeros(parameters_for_estimation["max_time"])
    print(f"Evaluating estimated parameters on {n_graphs} graphs and 1000 simulations per graph...")
    print(f"Estimated parameters: k={avg_degree}, beta={beta:.3f}, rho={rho:.3f}")
    for _ in range(n_graphs):
        G_estimated = create_pa_graph(total_nodes=population_size, avg_degree=int(avg_degree))
        simulation_results = test_SIR_simulation(
            G_estimated,
            beta,
            rho,
            ground_truth_infected[0],
            15,
            1000,
            vaccinations=vaccinations,
        )
        susceptible += simulation_results[0] / n_graphs
        infected += simulation_results[1] / n_graphs
        recovered += simulation_results[2] / n_graphs
        new_infections += simulation_results[3] / n_graphs
        vaccinated += simulation_results[4] / n_graphs

    RMSE = np.sqrt(np.mean((new_infections - ground_truth_infected[1:]) ** 2))
    print(f"Final RMSE with estimated parameters: {RMSE:.3f}")

    plot_SIR_dynamics(
        susceptible,
        infected,
        recovered,
        new_infections,
        vaccinated,
        ground_truth_infected=ground_truth_infected[1:],
        xticks=[(i + 42) % 52 + 1 for i in range(parameters_for_estimation["max_time"])],
        plot_dir=plot_dir,
        graph_type="PA_estimated",
    )

    # Attempt with uncapped halving times (greater than number of iterations needed for convergence)
    # This shows that no real improvement is achieved after the first few iterations
    print("Re-running estimation for 100 iterations (uncapped halving times)...")
    avg_degree, beta, rho, error_history, k_history, beta_history, rho_history = estimate_params(
        **{**parameters_for_estimation, "halving_times": 100, "delta_k": 4, "delta_beta": 0.2, "delta_rho": 0.2},
    )

    n_graphs = 30
    new_infections = np.zeros(parameters_for_estimation["max_time"])
    susceptible = np.zeros(parameters_for_estimation["max_time"])
    infected = np.zeros(parameters_for_estimation["max_time"])
    recovered = np.zeros(parameters_for_estimation["max_time"])
    vaccinated = np.zeros(parameters_for_estimation["max_time"])
    print(f"Evaluating estimated parameters on {n_graphs} graphs and 1000 simulations per graph (uncapped halvings)...")
    print(f"Estimated parameters: k={avg_degree}, beta={beta:.3f}, rho={rho:.3f}")

    for _ in range(n_graphs):
        G_estimated = create_pa_graph(total_nodes=population_size, avg_degree=int(avg_degree))
        simulation_results = test_SIR_simulation(
            G_estimated,
            beta,
            rho,
            ground_truth_infected[0],
            15,
            1000,
            vaccinations=vaccinations,
        )
        susceptible += simulation_results[0] / n_graphs
        infected += simulation_results[1] / n_graphs
        recovered += simulation_results[2] / n_graphs
        new_infections += simulation_results[3] / n_graphs
        vaccinated += simulation_results[4] / n_graphs

    RMSE = np.sqrt(np.mean((new_infections - ground_truth_infected[1:]) ** 2))
    print(f"Final RMSE with estimated parameters (uncapped halvings): {RMSE:.3f}")

    plot_SIR_dynamics(
        susceptible,
        infected,
        recovered,
        new_infections,
        vaccinated,
        ground_truth_infected=ground_truth_infected[1:],
        xticks=[(i + 42) % 52 + 1 for i in range(parameters_for_estimation["max_time"])],
        plot_dir=plot_dir,
        graph_type="PA_uncapped",
    )

    plot_optimization_history(
        error_history,
        k_history,
        beta_history,
        rho_history,
        plot_dir=plot_dir,
        graph_type="PA_uncapped",
    )
