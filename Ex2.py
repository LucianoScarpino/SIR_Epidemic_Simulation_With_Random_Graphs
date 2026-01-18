import numpy as np
import scipy as sp
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import os

# --- GLOBAL CONFIGURATION ---
ACTIONS = [-1, +1]

class NetworkGame:
    def __init__(self, n_players, n_coord_players):
        """
        n_players: Total number of players.
        n_coord_players (n1): Players from 0 to n1-1 are 'Coordination' (V1).
                              Players from n1 to n-1 are 'Anti-Coordination' (V2).
        Assumes a complete graph topology.
        """
        self.n = n_players
        self.n1 = n_coord_players
        self.players = range(n_players)
        self.n_config = len(ACTIONS) ** n_players
        
        # Pre-compute map ID <-> Configuration
        self.id2config = {}
        self.config2id = {}
        for i, config in enumerate(itertools.product(ACTIONS, repeat=n_players)):
            self.id2config[i] = config
            self.config2id[config] = i

    def utility(self, player_idx, config):
        """
        Calculates the utility for a player given the full configuration.
        V1 (Coord): sum |xi + xj| (2 if equal, 0 if different).
        V2 (Anti):  sum |xi - xj| (2 if different, 0 if equal).
        """
        xi = config[player_idx]
        u = 0
        is_coord = player_idx < self.n1
        
        for j in self.players:
            if j == player_idx: continue
            xj = config[j]
            if is_coord:
                u += abs(xi + xj) # Reward agreement
            else:
                u += abs(xi - xj) # Reward disagreement
        return 0.5 * u

    def get_best_response(self, player_idx, config):
        """Returns a list of optimal actions (contains 1 or 2 actions)."""
        payoffs = {}
        curr_config = list(config)
        
        for action in ACTIONS:
            curr_config[player_idx] = action
            payoffs[action] = self.utility(player_idx, tuple(curr_config))
            
        max_payoff = max(payoffs.values())
        return [a for a, p in payoffs.items() if p == max_payoff]

    def find_nash_equilibria(self):
        """Finds all Pure Nash Equilibria by brute-force checking every configuration."""
        nes = []
        for x_id in range(self.n_config):
            config = self.id2config[x_id]
            is_ne = True
            for i in self.players:
                # If the current action is not in the best response set, it's not a NE
                if config[i] not in self.get_best_response(i, config):
                    is_ne = False
                    break
            if is_ne:
                nes.append(config)
        return nes
    


class DynamicsSimulator:
    def __init__(self, game):
        self.game = game

    def build_lambda_br(self):
        """Builds the transition rate matrix for Asynchronous Best Response."""
        n_config = self.game.n_config
        Lambda = sp.sparse.lil_matrix((n_config, n_config), dtype=float)

        for x_id in range(n_config):
            x = self.game.id2config[x_id]
            for i in self.game.players:
                br_set = self.game.get_best_response(i, x)
                # Rate 1 for activation. If indifferent, split probability 1/len(br)
                weight = 1.0 / len(br_set)
                
                for action in br_set:
                    if action == x[i]: continue # Self-loops are ignored in the off-diagonal matrix
                    
                    y = list(x)
                    y[i] = action
                    y_id = self.game.config2id[tuple(y)]
                    Lambda[x_id, y_id] += weight
                    
        return Lambda.tocsr()

    def build_lambda_nbr(self, eta):
        """Builds the transition rate matrix for Noisy Best Response."""
        n_config = self.game.n_config
        Lambda = sp.sparse.lil_matrix((n_config, n_config), dtype=float)

        for x_id in range(n_config):
            x = self.game.id2config[x_id]
            for i in self.game.players:
                # Calculate utility for all possible actions of player i
                utils = []
                for a in ACTIONS:
                    y_temp = list(x)
                    y_temp[i] = a
                    utils.append(self.game.utility(i, tuple(y_temp)))
                
                # Compute Logit probabilities
                utils = np.array(utils)
                z = eta * utils
                exp_z = np.exp(z)
                probs = exp_z / np.sum(exp_z)
                
                # Add transitions
                for k, action in enumerate(ACTIONS):
                    if action == x[i]: continue
                    
                    y = list(x)
                    y[i] = action
                    y_id = self.game.config2id[tuple(y)]
                    Lambda[x_id, y_id] += 1.0 * probs[k] # Activation rate = 1
                    
        return Lambda.tocsr()

    def simulate(self, Lambda, x0_config, n_steps=1000, seed=None):
        """Executes a CTMC simulation using the Gillespie method."""
        rng = np.random.default_rng(seed)
        x0_id = self.game.config2id[x0_config]
        
        # Calculate total exit rates
        w = np.array(Lambda.sum(axis=1)).flatten()
        
        # Handle absorbing states: if w=0, we add a virtual self-loop rate 
        Lambda_sim = Lambda.tolil(copy = True)
        for i in range(len(w)):
            if w[i] == 0:
                w[i] = 1.0
                Lambda_sim[i, i] = 1.0 
        
        Lambda_sim = Lambda_sim.tocsr()

        # Discrete jump probability matrix
        D_inv = sp.sparse.diags(1.0 / w)
        P = D_inv @ Lambda_sim
        
        states = np.zeros(n_steps, dtype=int)
        times = np.zeros(n_steps, dtype=float) # Cumulative times
        
        curr_state = x0_id
        curr_time = 0.0
        states[0] = curr_state
        
        # Initial waiting time
        t_next = -np.log(rng.random()) / w[curr_state]
        curr_time += t_next
        times[0] = curr_time

        for k in range(1, n_steps):
            row = P[curr_state, :].toarray().flatten()
            next_state = rng.choice(len(row), p=row)
            
            states[k] = next_state
            curr_state = next_state
            
            # Holding time in the new state
            t_wait = -np.log(rng.random()) / w[curr_state]
            curr_time += t_wait
            times[k] = curr_time
            
        return states, times

    def get_absorption_distribution(self, Lambda, x0, n_runs=2000, n_steps_per_run=200):
        """
        Runs many short simulations and counts the final state of each run.
        Returns the frequency of absorption into each sink.
        """
        final_states = []
        for r in range(n_runs):
            # Seed must change per run
            states, _ = self.simulate(Lambda, x0, n_steps=n_steps_per_run, seed=None)
            final_states.append(states[-1])
        
        counts = np.bincount(final_states, minlength=self.game.n_config)
        return counts / n_runs

    def get_ergodic_distribution(self, Lambda, x0, n_steps=50000, seed=42):
        """
        Runs ONE long simulation and computes the time-weighted average occupation.
        """
        states, times = self.simulate(Lambda, x0, n_steps=n_steps, seed=seed)
        
        intervals = np.zeros(n_steps)
        intervals[0] = times[0]
        intervals[1:] = np.diff(times)
        
        burn = int(n_steps * 0.1)
        valid_intervals = intervals[burn:]
        valid_states = states[burn:]
        
        dist = np.zeros(self.game.n_config)
        np.add.at(dist, valid_states, valid_intervals)
        
        total_time = np.sum(dist)
        if total_time == 0: return dist
        return dist / total_time
    

# --- PLOTTING FUNCTIONS ---
def layered_pos_3players(node_labels, xgap=3.4, ygap=1.7):
    """
    Calculates positions for a 3-player hypercube.
    node_labels: list of strings like "(1, -1, 1)"
    """
    levels = {0: [], 1: [], 2: [], 3: []}
    for lbl in node_labels:
        # In una tupla di 3 elementi, il livello è dato dal numero di '1'.
        # Se la stringa è "(1, -1, 1)", il numero di '-1' ci dice quanti NON sono 1.
        # Livello = 3 - (numero di '-1')
        num_neg = lbl.count('-1')
        lvl = 3 - num_neg
        
        # Safety check (se qualcosa va storto col parsing)
        if lvl < 0: lvl = 0
        if lvl > 3: lvl = 3
        
        levels[lvl].append(lbl)
        
    for lvl in levels:
        levels[lvl] = sorted(levels[lvl])

    pos = {}
    for lvl in range(4):
        nodes = levels[lvl]
        m = len(nodes)
        y0 = (m - 1) / 2.0
        for r, c in enumerate(nodes):
            pos[c] = (lvl * xgap, (y0 - r) * ygap)
    return pos

def plot_transition_graph(Lambda, game, title, ax, use_layered=False, 
                          seed=0, xgap=3.4, ygap=1.7, threshold=1e-3,
                          spring_k=2.2, spring_iter=400):
    L = Lambda.tocoo()
    G = nx.DiGraph()

    # 1. Add Filtered Nodes and Edges
    for i, j, rate in zip(L.row, L.col, L.data):
        if rate > threshold:
            u_cfg = game.id2config[i]
            v_cfg = game.id2config[j]
            
            # FORMATTING CHANGE: Use standard tuple string (1, -1, 1)
            u_lbl = str(u_cfg)
            v_lbl = str(v_cfg)
            
            G.add_edge(u_lbl, v_lbl, weight=rate, label=f"{rate:.2g}")

    # Ensure all nodes are present
    for x_id in range(game.n_config):
         cfg = game.id2config[x_id]
         lbl = str(cfg)
         if lbl not in G: G.add_node(lbl)

    # 2. Layout
    if use_layered:
        pos = layered_pos_3players(list(G.nodes), xgap=xgap, ygap=ygap)
    else:
        pos = nx.spring_layout(G, seed=seed, k=spring_k, iterations=spring_iter)
    
    # 3. Draw Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1200, 
                           node_color="tab:blue", alpha=0.4)
    
    # 4. Draw Node Labels (Ridotto font per far stare la tupla)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="black")

    # 5. Draw Edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=14,
        width=1.2,
        connectionstyle='arc3,rad=0.1',
        min_source_margin=18,
        min_target_margin=18
    )

    # 6. Draw Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, 
                                 label_pos=0.35, font_size=7)

    ax.set_title(title, fontsize=10)
    ax.axis('off')

def plot_histograms(probs, game, title, ax):

    labels = [str(game.id2config[i]).replace(" ", "") for i in range(game.n_config)]
    
    x_pos = range(len(labels))
    
    bars = ax.bar(x_pos, probs, color='skyblue', edgecolor='black', width=0.7)
    
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    
    for rect in bars:
        height = rect.get_height()
        if height > 0.01:
            ax.text(rect.get_x() + rect.get_width()/2.0, height + 0.01, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=7)


# --- MAIN ---
if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "plots", "Ex2")
    os.makedirs(save_dir, exist_ok=True)

    print("--- 1. ANALYSIS OF NASH EQUILIBRIA ---")
    scenarios = [0, 1, 2, 3] 
    for n1 in scenarios:
        g = NetworkGame(n_players=3, n_coord_players=n1)
        nes = g.find_nash_equilibria()
        # Print using 1/-1 format
        print(f"n_coord_players = {n1}: Found {len(nes)} NEs -> {nes}")
    print("-" * 40)

    # --- PART 2: SIMULATIONS & PLOTTING ---
    
    fig_graphs, ax_graphs = plt.subplots(2, 2, figsize=(16, 12))
    fig_graphs.subplots_adjust(wspace=0.1, hspace=0.3)
    
    fig_hist, ax_hist = plt.subplots(2, 2, figsize=(14, 12)) # increased height for rotated labels
    fig_hist.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.15)
    
    x0 = (+1, -1, +1)
    eta = 5.0
    
    # --- 2a. COORDINATION GAME (n1=3) ---
    print("\n--- 2. COORDINATION GAME (n1=3) ---")
    g_coord = NetworkGame(n_players=3, n_coord_players=3)
    sim_coord = DynamicsSimulator(g_coord)
    
    # BR
    print("Simulating BR...")
    Lambda_br_c = sim_coord.build_lambda_br()
    
    
    plot_transition_graph(Lambda_br_c, g_coord, "Coordination BR Dynamics", 
                          ax=ax_graphs[0, 0], threshold=0.0, use_layered=True)
    
    dist_br_c = sim_coord.get_absorption_distribution(Lambda_br_c, x0, n_steps_per_run=50, n_runs=5000)
    plot_histograms(dist_br_c, g_coord, f"Coordination BR\n", ax_hist[0, 0])
    
    # NBR
    print("Simulating NBR...")
    Lambda_nbr_c = sim_coord.build_lambda_nbr(eta=eta)
    
    plot_transition_graph(Lambda_nbr_c, g_coord, f"Coordination NBR (eta={eta})\n", 
                          ax=ax_graphs[0, 1], threshold=0.0, use_layered=True, xgap=3.4, ygap=1.7)
    
    dist_nbr_c = sim_coord.get_ergodic_distribution(Lambda_nbr_c, x0, n_steps=50000)
    plot_histograms(dist_nbr_c, g_coord, f"Coordination NBR (eta={eta})\n", ax_hist[0, 1])

    # --- 2b. ANTI-COORDINATION GAME (n1=0) ---
    print("\n--- 3. ANTI-COORDINATION GAME (n1=0) ---")
    g_antic = NetworkGame(n_players=3, n_coord_players=0)
    sim_antic = DynamicsSimulator(g_antic)
    
    # BR
    print("Simulating BR...")
    Lambda_br_a = sim_antic.build_lambda_br()
    
    
    plot_transition_graph(Lambda_br_a, g_antic, "Anti-Coordination BR Dynamics", 
                          ax=ax_graphs[1, 0], threshold=0.0, seed=0,
                          use_layered=True, xgap=3.4, ygap=1.7)
    
    dist_br_a = sim_antic.get_ergodic_distribution(Lambda_br_a, x0, n_steps=50000)
    plot_histograms(dist_br_a, g_antic, f"Anti-Coordination BR\n", ax_hist[1, 0])
    
    # NBR
    print("Simulating NBR...")
    Lambda_nbr_a = sim_antic.build_lambda_nbr(eta=eta)
    
    plot_transition_graph(Lambda_nbr_a, g_antic, f"Anti-Coordination NBR (eta={eta})\n", 
                          ax=ax_graphs[1, 1], threshold=0.0, seed=1, use_layered=True)
    
    dist_nbr_a = sim_antic.get_ergodic_distribution(Lambda_nbr_a, x0, n_steps=50000)
    plot_histograms(dist_nbr_a, g_antic, f"Anti-Coordination NBR (eta={eta})\n", ax_hist[1, 1])

    # --- SAVE ---
    graph_save_path = os.path.join(save_dir, "transition_graphs.png")
    fig_graphs.savefig(graph_save_path, dpi=300, bbox_inches='tight')
    print(f"\nTransition Graphs saved to: {graph_save_path}")
    
    hist_save_path = os.path.join(save_dir, "dynamics_comparison.png")
    fig_hist.savefig(hist_save_path, dpi=300, bbox_inches='tight')
    print(f"Histograms saved to: {hist_save_path}")
    
    plt.show()