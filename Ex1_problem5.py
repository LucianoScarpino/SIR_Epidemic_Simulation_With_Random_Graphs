from Ex1 import test_SIR_simulation, plot_SIR_dynamics
from typing import Union, Callable, Tuple, List
import networkx as nx
import numpy as np
from itertools import product
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class Parameter:
    """Class representing a parameter to be estimated.
    Attributes:
        name: Name of the parameter.
        value: Current value of the parameter.
        delta: Step size for generating parameter values.
        round_fn: Optional function to round the parameter values.
        min_value: Optional minimum bound for the parameter value.
        max_value: Optional maximum bound for the parameter value.
        min_delta: Optional minimum bound for the delta value.
    Functions:
        generate_list: Generates a list of three values: value - delta, value, value + delta.
        set_value: Sets the parameter value to a new value. Ensures it is within bounds and applies rounding if specified.
        halve_delta: Halves the delta value. Ensures it does not go below min_delta if specified.
        __str__: Returns a string representation of the parameter.
    """

    def __init__(
        self,
        name: str,
        value: Union[float, int],
        delta: Union[float, int],
        round_fn: Callable = None,
        min_value: Union[float, int] = None,
        max_value: Union[float, int] = None,
        min_delta: Union[float, int] = None,
    ):
        """Initializes the Parameter object with the given attributes.
        Args:
            name: Name of the parameter.
            value: Current value of the parameter.
            delta: Step size for generating parameter values.
            round_fn: Optional function to round the parameter values.
            min_value: Optional minimum bound for the parameter value.
            max_value: Optional maximum bound for the parameter value.
            min_delta: Optional minimum bound for the delta value.
        """
        self.name = name
        self.value = value
        self.delta = delta
        self.round_fn = round_fn
        self.min_value = min_value
        self.max_value = max_value
        self.min_delta = min_delta

    def generate_list(self) -> List[Union[float, int]]:
        """Generates a list of three values: value - delta, value, value + delta.
        It also applies rounding and enforces min/max bounds if specified.
        Returns:
            List of parameter values to consider.
        """
        # Generate the three values
        values = [self.value - self.delta, self.value, self.value + self.delta]

        # Apply rounding and enforce bounds
        if self.round_fn:
            values = [self.round_fn(v) for v in values]

        # Enforce min/max bounds, by filtering out values outside the bounds
        if self.min_value is not None:
            values = [v for v in values if v >= self.min_value]
        if self.max_value is not None:
            values = [v for v in values if v <= self.max_value]
        return values

    def set_value(self, new_value: Union[float, int]):
        """Sets the parameter value to a new value.
        Ensures it is within bounds and applies rounding if specified.
        Args:
            new_value: New value to set for the parameter.
        """
        # Set the new value
        self.value = new_value

        # Apply rounding and enforce bounds
        if self.round_fn:
            self.value = self.round_fn(self.value)
        if self.min_value is not None and self.value < self.min_value:
            self.value = self.min_value
        if self.max_value is not None and self.value > self.max_value:
            self.value = self.max_value

    def __str__(self):
        """Gets a string representation of the parameter, useful for logging."""
        return f"{self.name}: {self.value} +- {self.delta}"

    def halve_delta(self):
        """Halves the delta value. Ensures it does not go below min_delta if specified."""
        if self.min_delta is not None and self.delta / 2 < self.min_delta:
            self.delta = self.min_delta
        else:
            self.delta /= 2


class GraphCreator:
    """Class containing static methods to create various types of graphs.
    It is only used to group graph creation functions together."""

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def create_non_linear_pa_graph(total_nodes: int, avg_degree: int, alpha: float) -> nx.Graph:
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
            prob_distribution = degrees**alpha / (degrees**alpha).sum()

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

    @staticmethod
    def create_er_graph(total_nodes: int, avg_degree: int) -> nx.Graph:
        """Creates an Erdos-Renyi graph with a given number of total nodes and average degree.
        Args:
            total_nodes: Total number of nodes in the graph.
            avg_degree: Desired average degree of the graph.
        Returns:
            G: A NetworkX graph representing the Erdős-Rényi graph.
        """
        # Calculate the probability of edge creation
        p = avg_degree / (total_nodes - 1)

        # Create the Erdos-Renyi graph
        G = nx.erdos_renyi_graph(total_nodes, p)

        return G

    @staticmethod
    def create_ws_graph(total_nodes: int, avg_degree: int, rewire_prob: float) -> nx.Graph:
        """Creates a Watts-Strogatz small-world graph with a given number of total nodes, average degree, and rewiring probability.
        Args:
            total_nodes: Total number of nodes in the graph.
            avg_degree: Desired average degree of the graph (must be even).
            rewire_prob: Probability of rewiring each edge.
        Returns:
            G: A NetworkX graph representing the Watts-Strogatz small-world graph.
        """
        G = GraphCreator.create_k_regular_graph(total_nodes, avg_degree)
        nodes = list(G.nodes)
        for node in nodes:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if np.random.rand() < rewire_prob:
                    # Rewire this edge
                    G.remove_edge(node, neighbor)
                    # Choose a new target node that is not the current node and not already a neighbor
                    potential_targets = set(nodes) - {node} - set(G.neighbors(node))
                    if potential_targets:
                        new_target = np.random.choice(list(potential_targets))
                        G.add_edge(node, new_target)
        return G

    @staticmethod
    def _sample_truncated_powerlaw_weights(n, gamma, k_min=1.0, k_max=60.0, rng=None):
        """
        Samples n 'degree weights' w_i ~ truncated powe-law on [k_min, k_max].
        Uses discrete pmf on {ceil(k_min), ..., floor(k_max)} for stability
        (instead of CDF with rounds on non integer values)
        """
        if rng is None:
            rng = np.random.default_rng()

        ks = np.arange(int(np.ceil(k_min)), int(np.floor(k_max)) + 1)
        pmf = ks ** (-gamma)
        pmf = pmf / pmf.sum()
        w = rng.choice(ks, size=n, p=pmf).astype(float)
        return w

    @staticmethod
    def create_cm_powerlaw_graph_expected_degree(
        total_nodes, avg_degree, gamma, k_min=1.0, k_max=60.0, seed=None
    ) -> nx.Graph:
        """
        Builds a variant of configuration model (Chung-Lu): generates a simple graph with
        expected degrees distributed as truncated power-law and forced mean ~ avg_degree.
        """
        if seed is None:
            seed = np.random.randint(0, 1e6)
        rng = np.random.default_rng(seed)

        w = GraphCreator._sample_truncated_powerlaw_weights(total_nodes, gamma, k_min=k_min, k_max=k_max, rng=rng)

        # scala per imporre media ~ avg_degree
        w = w * (avg_degree / w.mean())

        # evita valori fuori range
        w = np.clip(w, k_min, k_max)

        # expected_degree_graph genera un grafo semplice (senza self-loops se selfloops=False)
        G = nx.expected_degree_graph(w, selfloops=False, seed=seed)
        G = nx.convert_node_labels_to_integers(G)

        return G


def _eval_param_set(
    graph_fn: Callable,
    param_set: dict,
    ground_truth_infected: np.ndarray,
    max_time: int,
    n_simulations: int,
    vaccinations: np.ndarray,
    n_graphs: int,
) -> Tuple[dict, float]:
    """Evaluate a set of parameters by running SIR simulations and computing the RMSE error.
    This is used to perform parallel evaluations of parameter sets.
    Args:
        graph_fn: Function to create the graph given parameters.
        param_set: Dictionary of parameter values to evaluate.
        ground_truth_infected: Array of ground truth new infected counts over time.
        max_time: Maximum time steps for the simulation.
        n_simulations: Number of simulations to run for averaging.
        vaccinations: Array of vaccination counts over time.
    Returns:
        Tuple containing the parameter set and the computed RMSE error.
    """
    # Run multiple simulations and average the results
    avg_new_infected = np.zeros(max_time)

    # Create multiple graphs and average the simulation results
    for _ in range(n_graphs):

        # Create the graph with the current parameter set
        G = graph_fn(**{k: v for k, v in param_set.items() if k not in ["beta", "rho"]})

        # Run the SIR simulation
        _, _, _, new_infected, _ = test_SIR_simulation(
            G,
            param_set["beta"],
            param_set["rho"],
            ground_truth_infected[0],
            max_time,
            n_simulations,
            vaccinations=vaccinations,
        )
        avg_new_infected += new_infected / n_graphs  # Average over graphs

    # Compute RMSE error
    error = np.mean((avg_new_infected - ground_truth_infected[1:]) ** 2)
    error = np.sqrt(error)

    # Return the parameter set and the computed error
    return param_set, error


def estimate_params(
    graph_fn: Callable,
    vaccinations: np.ndarray,
    ground_truth_infected: np.ndarray,
    max_time: int,
    n_simulations: int,
    parameters: List[Parameter],
    halving_times: int,
    max_iters: int,
    max_params_tests: int,
    max_repetitions: int = 1,
    n_graphs: int = 1,
) -> Tuple[dict, List[float], dict]:
    """Estimate the parameters of the SIR model using a grid search optimization approach.
    This is the improved version of the algorithm, which uses adaptive halving, multiple graphs, parallel evaluation, and has better stopping criteria.
    Args:
        graph_fn: Function to create the graph given parameters.
        vaccinations: Array of vaccination counts over time.
        ground_truth_infected: Array of ground truth new infected counts over time.
        max_time: Maximum time steps for the simulation.
        n_simulations: Number of simulations to run for averaging.
        parameters: List of Parameter objects to estimate.
        halving_times: Number of times to halve the parameter deltas.
        max_iters: Maximum number of iterations for the optimization.
        max_params_tests: Maximum number of parameter sets to evaluate per iteration.
        max_repetitions: Maximum number of repetitions without improvement before halving deltas.
        n_graphs: Number of different graphs to average over for each parameter set evaluation.
    Returns:
        Tuple containing the best parameter set, error history, and parameter history.
    """
    # Initialize history tracking
    error_history = []
    params_history = {param.name: [] for param in parameters}
    previous_best_params_list = []

    # Initialize optimization variables
    best_params = None
    finished = False
    iter_count = 0
    repetition_count = 0

    # Get parameter names for easy access
    params_names = [param.name for param in parameters]

    # Log optimization settings
    print("--------------------------------------------")
    print("----------- Parameter estimation -----------")
    print("--------------------------------------------\n")

    print("Optimization parameters:")
    print(f"Number of halvings: {halving_times}\nNumber of simulations per evaluation: {n_simulations}")
    print(f"Maximum iterations: {max_iters}")
    print(f"Maximum parameter sets tested per iteration: {max_params_tests}")
    print(f"Maximum repetitions without improvement before halving deltas: {max_repetitions}")
    print(f"Number of graphs per evaluation: {n_graphs}\n")
    print("Initial parameters:")
    for param in parameters:
        print(param)
    print()
    print("Starting optimization...\n")

    # Record starting time
    starting_time = time()

    # Optimization loop
    while not finished:

        # Increment iteration counter
        iter_count += 1

        # Set smallest error to infinity at the start of each iteration
        smallest_error = float("inf")

        # Generate parameter grid with name and values, using current parameter values and deltas (implicitly
        # done by Parameter.generate_list. This also ensures that the parameter values are within their bounds.)
        params_grid_values = list(product(*[p.generate_list() for p in parameters]))
        params_grid = [dict(zip(params_names, param_set)) for param_set in params_grid_values]

        # Randomly sample a subset of parameter sets to evaluate,
        # with a size of min(max_params_tests, total number of sets)
        idx = np.random.choice(
            len(params_grid_values), size=min(max_params_tests, len(params_grid_values)), replace=False
        )
        params_grid = [params_grid[i] for i in idx]

        # Ensure the best parameters from the previous iteration are included in the grid, if we are not in the first iteration
        if best_params not in params_grid and best_params is not None:
            params_grid.append(best_params)

        # Compute the number of workers for parallel execution
        max_workers = min(os.cpu_count(), len(params_grid))

        # Evaluate parameter sets in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as pool_executor:
            futures = [
                pool_executor.submit(
                    _eval_param_set,  # function to execute
                    # arguments to the function
                    graph_fn,
                    param_set,
                    ground_truth_infected,
                    max_time,
                    n_simulations,
                    vaccinations,
                    n_graphs,
                )
                for param_set in params_grid
            ]

            # Collect results as they complete and store the best parameters
            for fut in as_completed(futures):
                param_set, rmse = fut.result()

                # Update best parameters if current RMSE is smaller
                if rmse < smallest_error:
                    smallest_error = rmse
                    best_params = param_set

        # Update history tracking
        error_history.append(smallest_error)

        # Record parameter values
        for param in parameters:
            params_history[param.name].append(best_params[param.name])
            param.set_value(best_params[param.name])  # Update parameter object with best value found

        # Check stopping criteria
        if iter_count >= max_iters or best_params in previous_best_params_list:
            repetition_count += 1  # No improvement in this iteration

            # Check if we need to halve deltas
            if repetition_count >= max_repetitions:
                repetition_count = 0  # Reset repetition count
                for param in parameters:
                    # Halve the delta of each parameter, using safe conditions defined in Parameter.halve_delta
                    # e.g. respecting min_delta
                    param.halve_delta()
                halving_times -= 1  # Decrement halving times

                # Check if we have exhausted all halvings
                if halving_times < 0:
                    finished = True
                else:
                    print(f"\nHalving parameter deltas. Remaining halvings: {halving_times}\n")

            # Check if maximum iterations reached and stop if so
            if iter_count >= max_iters:
                finished = True

        # Log iteration results
        print(f"Iteration {iter_count}: Best params: {best_params} with RMSE: {smallest_error:.3f}")

        # If finished, print summary
        if finished:
            print("\nOptimization finished.\n")
            print(f"Best parameters found: {best_params} with RMSE: {smallest_error:.3f}")
            print(f"Total optimization time: {time() - starting_time:.2f} seconds\n")

            # Warn if maximum iterations reached, meaning that no convergence was achieved
            if iter_count >= max_iters:
                print("[WARNING] - Maximum iterations reached.\n")

        # Store the best parameters of this iteration to check for repetitions
        previous_best_params_list.append(best_params)

    # Return the best parameters, error history, and parameter history
    return best_params, error_history, params_history


def plot_optimization_history(
    error_history: List[float],
    params_history: dict,
    plot_dir: str = None,
    graph_name: str = None,
):
    """Plot the optimization history including error and parameter values over iterations.
    Args:
        error_history: List of RMSE errors over iterations.
        params_history: Dictionary of parameter names to their values over iterations.
        plot_dir: Directory to save the plots. If None, plots are not saved.
        graph_name: Name of the graph type for labeling the plots.
    """

    # Create plot directory if specified
    if plot_dir is not None:
        plot_dir = Path(plot_dir) / graph_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        assert graph_name is not None, "Graph name must be provided if plot_dir is specified."

    # Generate iterations list, used for x-axis
    iterations = list(range(1, len(error_history) + 1))

    # Create and save error history plot
    plt.figure()
    plt.plot(iterations, error_history)
    plt.title(f"Optimization History for {graph_name}" if graph_name is not None else "Optimization History")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE Error")
    plt.grid()
    plt.locator_params(axis="x", integer=True)
    if plot_dir is not None:
        plt.savefig(plot_dir / f"{graph_name}_error_history.png")
    plt.clf()

    # Create and save parameter history plots
    for param_name, values in params_history.items():
        plt.plot(iterations, values, linewidth=2)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Parameter Value", fontsize=14)
        plt.title(
            (
                f"Parameter '{param_name}' History for {graph_name}"
                if graph_name is not None
                else f"Parameter '{param_name}' History"
            ),
            fontsize=16,
        )
        plt.grid()
        plt.locator_params(axis="x", integer=True)

        if plot_dir is not None:
            plt.savefig(plot_dir / f"{graph_name}_{param_name}_params_history.png")
        plt.clf()


def evaluate_graph(parameters_for_estimation: dict, graph_name: str, n_graphs_eval: int, n_evaluations: int) -> float:
    """Evaluate a graph type by estimating parameters and testing the SIR simulation with the estimated parameters
    Args:
        parameters_for_estimation: Dictionary of parameters for the estimation process.
        graph_name: Name of the graph type for labeling plots.
        n_graphs_eval: Number of different graphs to average over for each parameter set evaluation.
        n_evaluations: Number of simulations to run for averaging during testing.
    Returns:
        RMSE error of the SIR simulation with the estimated parameters.
    """
    print(f"\n================ Evaluating graph: {graph_name} ================\n")
    # Estimate parameters using the provided estimation function
    best_params, error_history, params_history = estimate_params(**parameters_for_estimation)

    # Plot error and parameter history
    plot_optimization_history(
        error_history,
        params_history,
        plot_dir=Path(__file__).parent / "plots" / "Ex1_problem5",
        graph_name=graph_name,
    )

    # Test the SIR simulation with the estimated parameters
    rho = best_params.pop("rho")
    beta = best_params.pop("beta")
    graph_fn = parameters_for_estimation["graph_fn"]
    ground_truth_infected = parameters_for_estimation["ground_truth_infected"]

    # ----- Run multiple simulations to get average dynamics -----
    # Initialize accumulators for dynamics
    average_new_infected = np.zeros(parameters_for_estimation["max_time"])
    susceptible = np.zeros(parameters_for_estimation["max_time"])
    infected = np.zeros(parameters_for_estimation["max_time"])
    recovered = np.zeros(parameters_for_estimation["max_time"])
    vaccinated = np.zeros(parameters_for_estimation["max_time"])

    # Create multiple graphs and average the simulation results
    for _ in range(n_graphs_eval):
        G = graph_fn(**best_params)
        simulation_results = test_SIR_simulation(
            G,
            beta,
            rho,
            ground_truth_infected[0],
            parameters_for_estimation["max_time"],
            n_evaluations,
            vaccinations=parameters_for_estimation["vaccinations"],
        )
        susceptible += simulation_results[0] / n_graphs_eval
        infected += simulation_results[1] / n_graphs_eval
        recovered += simulation_results[2] / n_graphs_eval
        average_new_infected += simulation_results[3] / n_graphs_eval
        vaccinated += simulation_results[4] / n_graphs_eval

    # Compute final RMSE and print it
    RMSE = np.sqrt(np.mean((average_new_infected - ground_truth_infected[1:]) ** 2))
    print(f"Final RMSE with estimated parameters on {graph_name}: {RMSE:.3f}")

    # Plot the SIR dynamics with the estimated parameters
    plot_SIR_dynamics(
        susceptible,
        infected,
        recovered,
        average_new_infected,
        vaccinated,
        ground_truth_infected[1:],
        xticks=[(i + 42) % 52 + 1 for i in range(parameters_for_estimation["max_time"])],
        plot_dir=Path(__file__).parent / "plots" / "Ex1_problem5" / graph_name,
        graph_type=graph_name,
    )

    # Return the RMSE error
    return RMSE


if __name__ == "__main__":

    # Set random seed for reproducibility
    np.random.seed(42)
    population_size = 934

    # Dictionary to store RMSE errors for each graph type
    errors = {}

    # Ground truth data for vaccinations and new infected counts over time
    vaccinations = np.array([5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60])
    ground_truth_infected = np.array([1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0])

    # Number of different graphs to average over during evaluation
    n_graphs_eval = 30

    # Common parameters for all graph evaluations and parameter estimations
    common_params = {
        "vaccinations": vaccinations,
        "ground_truth_infected": ground_truth_infected,
        "max_time": 15,
        "n_simulations": 40,
        "halving_times": 2,
        "max_iters": 100,
        "max_params_tests": 12,
        "max_repetitions": 4,
        "n_graphs": 5,
    }

    # --------------------------------------------
    # ------ PA Graph Parameter Estimation ------
    # --------------------------------------------

    # Create the graph function for PA graph, fixing the total number of nodes
    graph_fn = partial(GraphCreator.create_pa_graph, total_nodes=population_size)

    # Define parameters for estimation
    parameters_for_estimation = {
        **common_params,
        "graph_fn": graph_fn,
        "parameters": [
            Parameter(
                name="avg_degree", value=10, delta=1, round_fn=lambda x: max(int(x), 1), min_value=1, min_delta=1
            ),
            Parameter(name="beta", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="rho", value=0.6, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
        ],
    }

    # Evaluate the PA graph and store the RMSE error
    RMSE = evaluate_graph(parameters_for_estimation, "PA_Graph", n_graphs_eval=n_graphs_eval, n_evaluations=1000)
    errors["PA_Graph"] = RMSE

    # --------------------------------------------
    # -- Non-Linear PA Graph Parameter Estimation --
    # --------------------------------------------

    # Create the graph function for Non-Linear PA graph, fixing the total number of nodes
    graph_fn = partial(GraphCreator.create_non_linear_pa_graph, total_nodes=population_size)

    # Define parameters for estimation
    parameters_for_estimation = {
        **common_params,
        "graph_fn": graph_fn,
        "parameters": [
            Parameter(
                name="avg_degree", value=10, delta=1, round_fn=lambda x: max(int(x), 1), min_value=1, min_delta=1
            ),
            Parameter(name="beta", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="rho", value=0.6, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="alpha", value=0.5, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0),
        ],
    }

    # Evaluate the Non-Linear PA graph and store the RMSE error
    RMSE = evaluate_graph(
        parameters_for_estimation, "Non_Linear_PA_Graph", n_graphs_eval=n_graphs_eval, n_evaluations=1000
    )
    errors["Non_Linear_PA_Graph"] = RMSE

    # ----------------------------------------------------
    # ------ ER Graph Parameter Estimation ------
    # ----------------------------------------------------

    # Create the graph function for ER graph, fixing the total number of nodes
    graph_fn = partial(GraphCreator.create_er_graph, total_nodes=population_size)

    # Define parameters for estimation
    parameters_for_estimation = {
        **common_params,
        "graph_fn": graph_fn,
        "parameters": [
            Parameter(
                name="avg_degree", value=10, delta=1, round_fn=lambda x: max(int(x), 1), min_value=1, min_delta=1
            ),
            Parameter(name="beta", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="rho", value=0.6, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
        ],
    }

    # Evaluate the ER graph and store the RMSE error
    RMSE = evaluate_graph(parameters_for_estimation, "ER_Graph", n_graphs_eval=n_graphs_eval, n_evaluations=1000)
    errors["ER_Graph"] = RMSE

    # --------------------------------------------
    # -- CM Power-Law Graph Parameter Estimation --
    # --------------------------------------------

    # Create the graph function for CM Power-Law graph, fixing the total number of nodes
    graph_fn = partial(GraphCreator.create_cm_powerlaw_graph_expected_degree, total_nodes=population_size)

    # Define parameters for estimation
    parameters_for_estimation = {
        **common_params,
        "graph_fn": graph_fn,
        "parameters": [
            Parameter(
                name="avg_degree", value=10, delta=1, round_fn=lambda x: max(int(x), 1), min_value=1, min_delta=1
            ),
            Parameter(name="gamma", value=2.6, delta=0.3, round_fn=lambda x: round(x, 3), min_value=1.1, max_value=5),
            Parameter(name="beta", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="rho", value=0.6, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
        ],
    }

    # Evaluate the CM Power-Law graph and store the RMSE error
    RMSE = evaluate_graph(
        parameters_for_estimation, "CM_Power_Law_Graph", n_graphs_eval=n_graphs_eval, n_evaluations=1000
    )
    errors["CM_Power_Law_Graph"] = RMSE

    # ----------------------------------------------------
    # ------ WS Graph Parameter Estimation ------
    # ----------------------------------------------------

    # Define a rounding function to ensure avg_degree is even
    round_to_even = lambda x: int(round(x) // 2 * 2)

    # Create the graph function for WS graph, fixing the total number of nodes
    graph_fn = partial(GraphCreator.create_ws_graph, total_nodes=population_size)

    # Define parameters for estimation
    parameters_for_estimation = {
        **common_params,
        "graph_fn": graph_fn,
        "parameters": [
            Parameter(name="avg_degree", value=10, delta=2, round_fn=round_to_even, min_value=2, min_delta=2),
            Parameter(
                name="rewire_prob", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1
            ),
            Parameter(name="beta", value=0.3, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
            Parameter(name="rho", value=0.6, delta=0.1, round_fn=lambda x: round(x, 3), min_value=0, max_value=1),
        ],
    }

    # Evaluate the WS graph and store the RMSE error
    RMSE = evaluate_graph(parameters_for_estimation, "WS_Graph", n_graphs_eval=n_graphs_eval, n_evaluations=1000)
    errors["WS_Graph"] = RMSE

    # Final summary of errors
    print("\nSummary of RMSE errors:")
    for graph_type, rmse in errors.items():
        print(f"{graph_type}: {rmse:.3f}")
