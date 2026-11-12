import numpy as np
from tqdm.auto import tqdm
import igraph as ig
from scipy.special import expit as sigmoid
import random
from castle.metrics import MetricsDAG

def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def measurement(real, pred):
    FP = ((real == 0) & (pred != 0)).sum()
    TP = ((real != 0) & (pred != 0)).sum()
    FN = ((real != 0) & (pred == 0)).sum()
    TN = ((real == 0) & (pred == 0)).sum()
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # compute MCC
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0:
        mcc = 0.0
    else:
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # compute fdr
    if TP + FP == 0:
        fdr = 0.0
    else:
        fdr = FP / (TP + FP)
        
    if TP + FN == 0:
        fnr = 0.0
    else:
        fnr = FN / (TP + FN)
    
    lf = np.linalg.norm(real - pred, ord='fro')
    l1 = np.linalg.norm(real - pred, ord=1)
    l2 = np.linalg.norm(real - pred, ord=2)
    ln = np.linalg.norm(real - pred, ord='nuc')
    linf = np.linalg.norm(real - pred, ord=np.inf)
    shd = MetricsDAG((pred!=0).astype(int),(real!=0).astype(int)).metrics['shd']
    result_dict = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'fdr': float(fdr),
        'fnr': float(fnr),
        'lf': float(lf),
        'l1': float(l1),
        'l2': float(l2),
        'ln': float(ln),
        'linf': float(linf),
        'shd': float(shd)}
    return result_dict

def to_dag(W, max_edges=None, verbose=False):
    nozero_indices = np.nonzero(W)
    W_dag = np.zeros_like(W)
    abs_values = []
    if verbose:
        print(f"Number of non-zero entries in W: {len(nozero_indices[0])}")
    for i, j in zip(*nozero_indices):
        abs_values.append((np.abs(W[i, j]), i, j))
    abs_values.sort(reverse=True)
    if max_edges is None:
        max_edges = len(abs_values)
    if verbose:
        pbar = tqdm(total=max_edges, desc="Building DAG")
    count = 0
    for _, i, j in abs_values:
        if verbose:
            pbar.update(1)
        W_dag[i, j] = W[i, j]
        G = ig.Graph.Weighted_Adjacency(W_dag.tolist())
        if not G.is_dag():
            W_dag[i, j] = 0
        else:
            count += 1
            if count >= max_edges:
                break
    if verbose:
        pbar.close()
    return W_dag


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W
