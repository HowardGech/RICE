import numpy as np
import typing
import igraph as ig
from scipy.stats import nbinom


def simulate_Pois(W: np.ndarray, 
                        theta: np.ndarray,
                        n: int, 
                        intercept: bool = True,
                        g: callable = lambda x: x,
                        d: int = 0,
                        shift: float = 1.0,
                        coef : float = 1.0,
                        covar: float = 1.0,
                        intervene_prob: float = 0.9,
                        hard_intervention: bool = True
                        ) -> np.ndarray:
    def _simulate_single_equation(X, w, cov, mask, shift):
        n = X.shape[0]
        p = w.shape[0]
        cov = cov.reshape(-1, 1)
        X = X.reshape(n, -1)
        w = w.reshape(-1, 1)
        Xw = (X @ w).reshape(-1, 1)
        shift = shift.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        l = (Xw + cov) * (1 - mask) + shift
        if np.max(l) > 7.5:
            l -= (np.max(l) - 7.5)
        elif np.mean(l) < .5:
            l += (.5 - np.mean(l))
        x = np.random.poisson(np.exp(l))
        return x
        

    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
        
    
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])
    
    U = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        U = np.vstack((theta, U))

    
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
        d += 1
    ones = np.ones((n, 1))
    
    V = np.random.uniform(-1.5*shift, -shift, size=(1, p))
    
    covar = Z @ U
    
    shift = ones @ V
    
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    nonintervene_idx = np.setdiff1d(np.arange(n), intervene_idx)
    intervene_mask = np.zeros((n, p), dtype=bool)

    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    # from remaining nodes, randomly select each_intervention_size nodes to intervene on the current node
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)
    
    
    
    shift_masked = shift * intervene_mask
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        
        if hard_intervention:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar[:, j], intervene_mask[:, j], shift_masked[:, j]).reshape(-1)
        else:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar[:, j], np.zeros(n), shift_masked[:, j]).reshape(-1)

   
    return X, intervene_mask, Z, U, V

def simulate_NB(W: np.ndarray, 
                        theta_m: np.ndarray,
                        theta_r: np.ndarray,
                        n: int, 
                        intercept: bool = True,
                        g: callable = lambda x: x,
                        d: int = 0,
                        shift: float = 1.0,
                        coef : float = 1.0,
                        covar: float = 1.0,
                        intervene_prob: float = 0.9,
                        hard_intervention: bool = True
                        ) -> np.ndarray:
    def _simulate_single_equation(X, w, cov_m, cov_r, mask, shift_m, shift_r):
        n = X.shape[0]
        p = w.shape[0]
        cov_m = cov_m.reshape(-1, 1)
        cov_r = cov_r.reshape(-1, 1)
        X = X.reshape(n, -1)
        w = w.reshape(-1, 1)
        Xw = (X @ w).reshape(-1, 1)
        shift_m = shift_m.reshape(-1, 1)
        shift_r = shift_r.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        m = (Xw + cov_m) * (1 - mask) + shift_m
        r = cov_r * (1 - mask) + shift_r
        if np.max(m) > 6.5:
            m -= (np.max(m)- 6.5)
        if np.mean(m) < .5:
            m += (.5 - np.mean(m))
        m = np.exp(m)
        r = np.exp(r)
        m = m.flatten()
        r = r.flatten()
        x = nbinom.rvs(r, r/(m+r))
        return x
    
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])
    
    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))

    
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
        d += 1
    ones = np.ones((n, 1))
    
    Vm = np.random.uniform(-1.5*shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift/2, shift/2, size=(1, p))
    
    covar_m = Z @ Um
    covar_r = Z @ Ur
    
    shift_m = ones @ Vm
    shift_r = ones @ Vr
    
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    nonintervene_idx = np.setdiff1d(np.arange(n), intervene_idx)
    intervene_mask = np.zeros((n, p), dtype=bool)

    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    # from remaining nodes, randomly select each_intervention_size nodes to intervene on the current node
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)
    
    
    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        
        if hard_intervention:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j], intervene_mask[:, j], shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)
        else:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j], np.zeros(n), shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)
            
    return X, intervene_mask, Z, Um, Ur, Vm, Vr

def simulate_PLN(W: np.ndarray, 
                        theta: np.ndarray,
                        n: int,
                        intercept: bool = True,
                        g: callable = lambda x: x,
                        d: int = 0,
                        shift: float = 1.0,
                        coef : float = 1.0,
                        covar: float = 1.0,
                        noise_scale: typing.Optional[typing.Union[float,typing.List[float]]] = None,
                        intervene_prob: float = 0.9,
                        hard_intervention: bool = True
                        ) -> np.ndarray:
    def _simulate_single_equation(X, w, cov, noise_scale, mask, shift):
        n = X.shape[0]
        p = w.shape[0]
        cov = cov.reshape(-1, 1)
        X = X.reshape(n, -1)
        w = w.reshape(-1, 1)
        Xw = (X @ w).reshape(-1, 1)
        shift = shift.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        l = (Xw + cov) * (1 - mask) + shift
        if np.max(l) > 7.5:
            l -= (np.max(l) - 7.5)
        if np.mean(l) < .5:
            l += (.5 - np.mean(l))
        y = np.random.normal(size=l.shape, scale=noise_scale) if noise_scale!=0 else 0
        l += y
        x = np.random.poisson(np.exp(l))
        return x
    
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
        
    
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])
    
    if noise_scale is None:
        noise_scale_vec = np.zeros(p)
    elif isinstance(noise_scale, (int, float)):
        noise_scale_vec = np.ones(p) * noise_scale
    elif isinstance(noise_scale, list):
        assert len(noise_scale) == p, "noise_scale should have the same length as the number of nodes"
        noise_scale_vec = np.array(noise_scale)
    else:
        raise ValueError("noise_scale should be a float, int, or a list of floats")
    
    U = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        U = np.vstack((theta, U))
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
        d += 1
    ones = np.ones((n, 1))
    V = np.random.uniform(-1.5*shift, -shift, size=(1, p))
    covar = Z @ U
    shift = ones @ V
    
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    nonintervene_idx = np.setdiff1d(np.arange(n), intervene_idx)
    intervene_mask = np.zeros((n, p), dtype=bool)

    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    # from remaining nodes, randomly select each_intervention_size nodes to intervene on the current node
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)
    
    shift_masked = shift * intervene_mask
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        
        if hard_intervention:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar[:, j], noise_scale_vec[j], intervene_mask[:, j], shift_masked[:, j]).reshape(-1)
        else:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar[:, j], noise_scale_vec[j], np.zeros(n), shift_masked[:, j]).reshape(-1)
            
    return X, intervene_mask, Z, U, V


def simulate_NB_highMOI(W: np.ndarray, 
                        theta_m: np.ndarray,
                        theta_r: np.ndarray,
                        n: int, 
                        MOI: int = 1,
                        intercept: bool = True,
                        g: callable = lambda x: x,
                        d: int = 0,
                        shift: float = 1.0,
                        coef : float = 1.0,
                        covar: float = 1.0,
                        intervene_prob: float = 0.9,
                        hard_intervention: bool = True
                        ) -> np.ndarray:
    def _simulate_single_equation(X, w, cov_m, cov_r, mask, shift_m, shift_r):
        n = X.shape[0]
        p = w.shape[0]
        cov_m = cov_m.reshape(-1, 1)
        cov_r = cov_r.reshape(-1, 1)
        X = X.reshape(n, -1)
        w = w.reshape(-1, 1)
        Xw = (X @ w).reshape(-1, 1)
        shift_m = shift_m.reshape(-1, 1)
        shift_r = shift_r.reshape(-1, 1)
        mask = mask.reshape(-1, 1)
        m = (Xw + cov_m) * (1 - mask) + shift_m
        r = cov_r * (1 - mask) + shift_r
        if np.max(m) > 6.5:
            m -= (np.max(m)- 6.5)
        if np.mean(m) < .5:
            m += (.5 - np.mean(m))
        m = np.exp(m)
        r = np.exp(r)
        m = m.flatten()
        r = r.flatten()
        x = nbinom.rvs(r, r/(m+r))
        return x
    
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])
    
    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))

    
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
        d += 1
    ones = np.ones((n, 1))
    
    Vm = np.random.uniform(-1.5*shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift/2, shift/2, size=(1, p))
    
    covar_m = Z @ Um
    covar_r = Z @ Ur
    
    shift_m = ones @ Vm
    shift_r = ones @ Vr
    
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    nonintervene_idx = np.setdiff1d(np.arange(n), intervene_idx)
    intervene_mask = np.zeros((n, p), dtype=bool)

    for i in intervene_idx:
        choice = np.random.choice(p, size=MOI, replace=False)
        intervene_mask[i, choice] = True
    
    
    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        
        if hard_intervention:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j], intervene_mask[:, j], shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)
        else:
            X[:,j] = _simulate_single_equation(g(X[:, parents]), W[parents, j], covar_m[:, j], covar_r[:, j], np.zeros(n), shift_m_masked[:, j], shift_r_masked[:, j]).reshape(-1)
            
    return X, intervene_mask, Z, Um, Ur, Vm, Vr