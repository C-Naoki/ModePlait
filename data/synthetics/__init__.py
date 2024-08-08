import random

import igraph as ig
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def load_data(**kwargs: dict) -> np.ndarray:
    global seed
    d = kwargs["d"] if kwargs.get("d") else 5
    p = kwargs["p"] if kwargs.get("p") else 0.5
    n = kwargs["n"] if kwargs.get("n") else 500
    seq_type = kwargs["seq_type"] if kwargs.get("seq_type") else [0, 1, 2, 1, 0]
    noise_type = kwargs["noise_type"] if kwargs.get("noise_type") else "laplace"
    graph_type = kwargs["graph_type"] if kwargs.get("graph_type") else "ER"
    seed = kwargs["seed"] if kwargs.get("seed") else 42

    # initialize seed
    np.random.seed(seed)
    random.seed(seed)

    t_span = (0, 0.05 * d)
    seeds = np.random.randint(0, 1000, len(set(seq_type)))

    B_ls = simulate_dags(d=d, p=p, seq_type=seq_type, seeds=seeds, graph_type=graph_type)
    W_ls = simulate_parameter(B_ls=B_ls, seq_type=seq_type, seeds=seeds)
    data = simulate_linear_sem(W_ls=W_ls, n=n, t_span=t_span, noise_type=noise_type)

    if "date" not in data.columns:
        data["date"] = pd.date_range(start="2021-01-01", periods=len(data), freq="D")
        data.set_index("date", inplace=True)

    return data, W_ls


# for OrbitMap
def load_arr_data(uuid: int, **kwargs: dict) -> np.ndarray:
    kwargs["uuid"] = uuid
    data, _ = load_data(**kwargs)
    return data.to_numpy()


def simulate_linear_sem(
    W_ls: list[np.ndarray], n: int, t_span: tuple, noise_type: str = "laplace"
) -> np.ndarray:
    """Simulate samples from linear SEM

    Parameters
    ----------
    W_ls: list[np.ndarray]
        each element of W_ls (i.e., W_ls[i]) is weighted adj matrix of DAGs.
        Shape: (n_seg,)
    n: int
        num of samples in each segment. total num of samples in X is n * n_seg.

    Returns
    -------
    X: np.ndarray
        synthetic dataset matrix. Shape: (n * n_seg, d)
    """

    n_seg = len(W_ls)  # num of segments
    d = W_ls[0].shape[0]  # dimension of data (i.e., the number of nodes)
    X = np.zeros([n * n_seg, d])
    for i in range(n_seg):
        W = W_ls[i]
        seg_range = range(i * n, (i + 1) * n)
        if not is_dag(W):
            raise ValueError("W must be a DAG")
        # empirical risk
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        E = _simulate_inherent_signals(n=n * n_seg, d=d, t_span=t_span, noise_type=noise_type)
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[seg_range, j] = _simulate_single_equation(
                X[:, parents][seg_range], W[parents, j], E[seg_range, j]
            )

    return pd.DataFrame(X, columns=[f"x{i+1}" for i in range(d)])


def simulate_dags(
    d: int, p: float, seq_type: list[int], seeds: list[int], graph_type: str = "ER"
) -> list[np.ndarray]:
    """
    Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    d: int
        num of nodes
    p: float
        expected number of edges (i.e., edge density)
    s0: int
        expected number of edges for scale-free graph
    seg_type: list[int]
        type of temporal sequences (e.g., [0, 1, 2, 1, 0] for 3 segments)
        Shape: (n_seg,)
    noise_type: str
        type of noise (e.g., "uniform" or "chaotic system")

    Returns
    -------
    B_ls: list[np.ndarray]
        binary adj matrix of DAGs. Shape: (n_seg,)
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    # set seed
    np.random.seed(seed)
    random.seed(seed)

    B_ls = []
    for i in seq_type:
        # set seed (for generating different graphs for each segment)
        np.random.seed(seeds[i])
        random.seed(seeds[i])

        if graph_type == "ER":
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, p=p)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == "SF":
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=1, directed=True)
            B = _graph_to_adjmat(G)
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        B_ls.append(B_perm)

    # reset seed
    np.random.seed(seed)
    random.seed(seed)

    return B_ls


def simulate_parameter(B_ls, seq_type, seeds, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """

    # set seed
    np.random.seed(seed)
    random.seed(seed)

    # W = np.zeros(B.shape)
    W_ls = []
    for i in seq_type:
        # set seed (for generating different graphs for each segment)
        np.random.seed(seeds[i])
        random.seed(seeds[i])

        W = np.zeros(B_ls[i].shape)
        S = np.random.randint(len(w_ranges), size=B_ls[i].shape)  # which range
        for j, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B_ls[i].shape)
            W += B_ls[i] * (S == j) * U
        W_ls.append(W)

    # reset seed
    np.random.seed(seed)
    random.seed(seed)

    return W_ls


def _simulate_inherent_signals(n: int, d: int, t_span: tuple, noise_type: str) -> np.ndarray:
    E = np.zeros([n, d])
    if noise_type == "chaotic":
        chaotic_ls = [
            gen_lorenz,
            gen_rossler,
            gen_van_der_pol,
            gen_lotka_volterra,
            gen_cubic_oscillator,
        ]
        for i in range(d):
            E[:, i] = chaotic_ls[i](size=n, t_span=t_span)[0]
    else:
        h = np.zeros([n, d])
        # `beta` is the coefficient in E's AR(1) process
        beta = np.random.uniform(0.8, 0.998, size=d)
        # `v` is the variance of the noise in E's AR(1) process
        v = np.random.uniform(0.01, 0.1, size=d)
        for t in range(n):
            for i in range(d):
                if t == 0:
                    h[t, :] = np.log(np.random.uniform(0.1, 0.5, size=d))
                else:
                    h[t, i] = beta[i] * h[t - 1, i] + np.random.normal(0, v[i])
                # non-Gaussian distribution
                if noise_type == "laplace":
                    E[t, i] = np.random.laplace(0, np.exp(h[t, i]))
                elif noise_type == "uniform":
                    E[t, i] = np.random.uniform(-np.exp(h[t, i]), np.exp(h[t, i]))
                elif noise_type == "gaussian":
                    E[t, i] = np.random.normal(0, np.exp(h[t, i]))
                else:
                    raise ValueError(
                        f"`{noise_type}` is not supported. Please try others (i.e., `laplace`)"
                    )

    return E


def _simulate_single_equation(X: np.ndarray, w: float, z: np.ndarray):
    """X: [n, num of parents], w: [num of parents], x: [n]"""
    x = X @ w + z
    return x


def gen_lorenz(
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    rho: float = 28.0,
    y0: list[float] = [1.0, 1.0, 1.0],
    size: int = 1000,
    t_span: tuple = (0, 50),
):
    # define the lorenz equations
    def lorenz(t, state, sigma, beta, rho):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    t_eval = np.linspace(*t_span, size)

    return solve_ivp(lorenz, t_span, y0, args=(sigma, beta, rho), t_eval=t_eval).y


def gen_rossler(
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    y0: list[float] = [0.0, 1.0, 0.0],
    size: int = 1000,
    t_span: tuple = (0, 50),
):
    # define the rossler equations
    def rossler(t, y, a, b, c):
        dydt = [-y[1] - y[2], y[0] + a * y[1], b + y[2] * (y[0] - c)]
        return dydt

    t_eval = np.linspace(*t_span, size)

    return solve_ivp(rossler, t_span, y0, args=(a, b, c), t_eval=t_eval).y


def gen_van_der_pol(
    mu: float = 1.0,
    y0: list[float] = [2.0, 0.0],
    size=1000,
    t_span: tuple = (0, 50),
):
    # define the van der pol equations
    def van_der_pol(t, y, mu):
        dydt = [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]
        return dydt

    t_eval = np.linspace(*t_span, size)

    return solve_ivp(van_der_pol, t_span, y0, args=(mu,), t_eval=t_eval).y


def gen_lotka_volterra(
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    gamma: float = 1.0,
    y0: list[float] = [10.0, 5.0],
    size: int = 1000,
    t_span: tuple = (0, 50),
):
    # define the lotka-volterra equations
    def lotka_volterra(t, state, alpha, beta, delta, gamma):
        prey, predator = state
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator
        return [dprey_dt, dpredator_dt]

    t_eval = np.linspace(*t_span, size)

    return solve_ivp(lotka_volterra, t_span, y0, args=(alpha, beta, delta, gamma), t_eval=t_eval).y


def gen_cubic_oscillator(
    a: float = 0.1,
    b: float = 2.0,
    c: float = 2.0,
    d: float = 0.1,
    y0: list[float] = [2.0, 0.0],
    size: int = 1000,
    t_span: tuple = (0, 50),
):
    # define the cubic oscillator equations
    def cubic_oscillator(t, state, a, b, c, d):
        x, y = state
        dxdt = -a * x**3 + b * y**3
        dydt = -c * x**3 - d * y**3
        return [dxdt, dydt]

    t_eval = np.linspace(*t_span, size)

    return solve_ivp(cubic_oscillator, t_span, y0, args=(a, b, c, d), t_eval=t_eval).y


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


if __name__ == "__main__":
    data = load_data(0)
