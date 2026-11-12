import numpy as np
import typing
import igraph as ig
from scipy.stats import nbinom


def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def _simulate_single_equation(X, w, cov_m, cov_r, mask, shift_m, shift_r,
                              apply_clip=False, cap=7.5, floor=-0.5):
    """One structural equation, on the count scale.

    The linear predictor is capped at `cap` and its mean floored at `floor`, reproducing the
    rescale of the published generator, which is the default (`apply_clip=False`).

    `apply_clip=True` additionally clips elementwise at `cap` after the floor adjustment. The
    floor can undo the cap when a single cell is extreme; exp(m) then exceeds the
    representable integer range and nbinom.rvs returns negative counts, which is the failure
    the guard prevents. It is NOT a no-op otherwise: at p=12, n=400, d=3 with g=log(1+x) the
    clip changed the maximum count from 7518 to 376 on the same seed, so clipped and
    unclipped runs must not be pooled. Record which was used alongside the output.
    """
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
    if np.max(m) > cap:
        m -= (np.max(m) - cap)
    if np.mean(m) < floor:
        m += (floor - np.mean(m))
    if apply_clip:
        m = np.minimum(m, cap)
    m = np.exp(m)
    r = np.exp(r)
    m = m.flatten()
    r = r.flatten()
    x = nbinom.rvs(r, r / (m + r))
    return x


def simulate_Pois_tgt(W: np.ndarray, 
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

def simulate_NB_tgt(W: np.ndarray, 
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
                        apply_clip: bool = False,
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
        if apply_clip:
            m = np.minimum(m, 6.5)        # see _simulate_single_equation
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

def simulate_PLN_tgt(W: np.ndarray, 
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



def simulate_NB_structured(W: np.ndarray,
                           theta_m: np.ndarray,
                           theta_r: np.ndarray,
                           n: int,
                           intercept: bool = True,
                           g: callable = lambda x: x,
                           d: int = 0,
                           shift: float = 1.0,
                           coef: float = 1.0,
                           covar: float = 1.0,
                           intervene_prob: float = 0.9,
                           hard_intervention: bool = True,
                           n_modules: int = 5,
                           module_noise_scale: float = 0.0,
                           module_assignment: np.ndarray = None,
                           ) -> tuple:
    """
    Simulate Perturb-seq data with module-structured unmeasured confounding
    (Setting 4 in the manuscript).
    
    Unlike simulate_NB_tgt, which samples confounder coefficients
    independently for each gene, this function groups genes into modules
    and assigns correlated coefficients within each module. The strength
    of within-module correlation is controlled by `module_noise_scale`.
    
    Parameters
    ----------
    W, theta_m, theta_r, n, intercept, g, d, shift, coef, covar,
    intervene_prob, hard_intervention :
        Same as in simulate_NB_tgt.
    
    n_modules : int, default 5
        Number of gene modules for structured confounding. Each unmeasured
        confounder affects genes within a module coherently.
    
    module_noise_scale : float, default 0.0
        Per-gene deviation from the module-level coefficient, in units of
        `coef`. Within-module correlation is approximately
        rho ≈ 1 / (1 + module_noise_scale^2):
            - 0.0 = perfectly correlated within modules
            - 0.5 ≈ rho 0.8
            - 1.0 = rho 0.5
            - 3.0 ≈ rho 0.1 (effectively i.i.d.)
    
    module_assignment : array of length p, or None
        Gene-to-module assignment. If None, a random permutation is used
        so that module membership is decoupled from network topology.
    
    Returns
    -------
    X : (n, p) array of simulated counts
    intervene_mask : (n, p) bool array
    Z : observed covariate matrix
    Um, Ur : confounder coefficient matrices (with module structure)
    Vm, Vr : intervention effect vectors
    module_assignment : (p,) array recording which module each gene belongs to
    """
    

    
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == p
    X = np.zeros([n, p])
    
    # --- Module-structured confounder coefficients ---
    if module_assignment is None:
        # Random assignment to decouple module membership from DAG topology
        base = np.repeat(np.arange(n_modules),
                         int(np.ceil(p / n_modules)))[:p]
        module_assignment = np.random.permutation(base)
    else:
        module_assignment = np.asarray(module_assignment)
        assert len(module_assignment) == p
        assert module_assignment.max() < n_modules
    
    if d > 0:
        # Module-level coefficient per (confounder, module)
        mu_m = np.random.uniform(-coef, coef, size=(d, n_modules))
        mu_r = np.random.uniform(-coef, coef, size=(d, n_modules))
        
        # Per-gene deviation around the module coefficient
        s = module_noise_scale
        if s > 0:
            noise_m = np.random.uniform(-coef * s, coef * s, size=(d, p))
            noise_r = np.random.uniform(-coef * s, coef * s, size=(d, p))
        else:
            noise_m = np.zeros((d, p))
            noise_r = np.zeros((d, p))
        
        Um = mu_m[:, module_assignment] + noise_m
        Ur = mu_r[:, module_assignment] + noise_r
    else:
        Um = np.zeros((0, p))
        Ur = np.zeros((0, p))
    
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))
    
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))
    ones = np.ones((n, 1))
    
    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))
    
    covar_m = Z @ Um
    covar_r = Z @ Ur
    
    shift_m = ones @ Vm
    shift_r = ones @ Vr
    
    # --- Intervention assignment (identical to simulate_NB_tgt) ---
    intervene_idx = np.random.choice(n, size=int(n * intervene_prob),
                                     replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes,
                                  size=each_intervention_size,
                                  replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)
    
    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask
    
    # --- Simulate gene-by-gene in topological order ---
    # Reuses _simulate_single_equation from simulate_NB_tgt.
    # If you prefer not to rely on that, copy the inner function here.
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _simulate_single_equation(
            g(X[:, parents]),
            W[parents, j],
            covar_m[:, j],
            covar_r[:, j],
            mask_j,
            shift_m_masked[:, j],
            shift_r_masked[:, j],
        ).reshape(-1)
    
    return X, intervene_mask, Z, Um, Ur, Vm, Vr, module_assignment


def simulate_NB_perturbation_interaction(W, theta_m, theta_r, n,
                                          intercept=True, g=lambda x: x,
                                          d=0, shift=1.0, coef=1.0,
                                          covar=1.0, intervene_prob=0.9,
                                          hard_intervention=True,
                                          interaction_strength=2.0):
    """
    Setting 1: confounder modulates perturbation effect.
    
    The shift induced by perturbation of gene j is v_j + gamma_j * z_um,
    where z_um is an unmeasured confounder. The first-stage linear
    regression cannot represent this product, so the full control function
    residual carries an irreducible interaction-induced signal.
    
    Parameters
    ----------
    interaction_strength : float
        Magnitude of the interaction coefficients gamma_j, sampled uniformly
        from [-interaction_strength, interaction_strength]. Set to 0 to
        recover an additive-confounding baseline.
    """
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, p])

    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))

    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))

    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))

    # Designate the LAST observed covariate as a "leaked" latent factor:
    # we pass Z to the analyst with that column REMOVED, so it acts as
    # unmeasured confounding. The interaction uses this column.
    if d >= 1:
        z_um = Z[:, -1]  # shape (n,)
    else:
        z_um = np.zeros(n)

    # Interaction coefficients: per-gene modulation strength
    gamma_m = np.random.uniform(-interaction_strength, interaction_strength, size=p)
    gamma_r = np.random.uniform(-interaction_strength * 0.3,
                                 interaction_strength * 0.3, size=p)

    covar_m = Z @ Um
    covar_r = Z @ Ur

    ones = np.ones((n, 1))
    shift_m = ones @ Vm  # (n, p)
    shift_r = ones @ Vr

    # Interaction-modulated shift: add gamma_j * z_um to v_j when perturbed
    interaction_m = np.outer(z_um, gamma_m)  # (n, p)
    interaction_r = np.outer(z_um, gamma_r)

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    # Apply interaction only where the cell is perturbed for gene j
    shift_m_total = (shift_m + interaction_m) * intervene_mask
    shift_r_total = (shift_r + interaction_r) * intervene_mask

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j],
            covar_m[:, j], covar_r[:, j],
            mask_j,
            shift_m_total[:, j], shift_r_total[:, j],
        ).reshape(-1)

    # IMPORTANT: drop the last column of Z before returning so it's hidden
    # from the analyst, acting as a true unmeasured confounder.
    Z_observed = Z[:, :-1] if d >= 1 else Z
    return X, intervene_mask, Z_observed, Um, Ur, Vm, Vr


def simulate_NB_mixture(W, theta_m, theta_r, n,
                        intercept=True, g=lambda x: x,
                        d=0, shift=1.0, coef=1.0, covar=1.0,
                        intervene_prob=0.9, hard_intervention=True,
                        n_states=3, state_effect_scale=1.0,
                        state_perturbation_scale=0.5):
    """
    Setting 3: cells belong to discrete latent states.
    
    Each cell is assigned to one of n_states latent states. State affects
    gene baseline expression and perturbation responsiveness. The state
    label is NOT returned among the observed covariates, so it acts as
    unmeasured confounding that the linear first-stage cannot recover.
    
    Parameters
    ----------
    n_states : int
        Number of discrete latent cell states.
    state_effect_scale : float
        Magnitude of state-specific baseline shifts per gene.
    state_perturbation_scale : float
        Magnitude of state-specific perturbation effect modulation.
        Set to 0 to make state affect only baseline (a strictly additive
        confounder, which the linear stage should partially absorb).
    """
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, p])

    # Standard observed covariates
    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))

    # Latent state assignment (hidden from analyst)
    state = np.random.choice(n_states, size=n)

    # State-specific baselines per gene
    state_baseline_m = np.random.uniform(-state_effect_scale,
                                          state_effect_scale,
                                          size=(n_states, p))
    state_baseline_r = np.random.uniform(-state_effect_scale * 0.3,
                                          state_effect_scale * 0.3,
                                          size=(n_states, p))

    # State-specific perturbation modulation per gene
    state_pert_m = np.random.uniform(-state_perturbation_scale,
                                      state_perturbation_scale,
                                      size=(n_states, p))
    state_pert_r = np.random.uniform(-state_perturbation_scale * 0.3,
                                      state_perturbation_scale * 0.3,
                                      size=(n_states, p))

    covar_m = Z @ Um + state_baseline_m[state]  # (n, p)
    covar_r = Z @ Ur + state_baseline_r[state]

    ones = np.ones((n, 1))
    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))

    # State-modulated perturbation effect
    shift_m_base = ones @ Vm + state_pert_m[state]
    shift_r_base = ones @ Vr + state_pert_r[state]

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    shift_m_masked = shift_m_base * intervene_mask
    shift_r_masked = shift_r_base * intervene_mask

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        mask_j = intervene_mask[:, j] if hard_intervention else np.zeros(n)
        X[:, j] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j],
            covar_m[:, j], covar_r[:, j],
            mask_j,
            shift_m_masked[:, j], shift_r_masked[:, j],
        ).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr, state




def simulate_NB_hidden_mediator(W, theta_m, theta_r, n,
                                  intercept=True, g=lambda x: x,
                                  d=0, shift=1.0, coef=1.0, covar=1.0,
                                  intervene_prob=0.9,
                                  hard_intervention=True,
                                  mediator_density=0.05,
                                  mediator_in_effect=3.0,
                                  mediator_out_effect=3.0):
    """
    Simulate Perturb-seq data with hidden mediators that bypass the
    observed regulatory cascade.

    For a fraction of (source, target) gene pairs (i, k), the perturbation
    D_i affects x_k through both the observed regulatory path (i -> ... -> k
    in W) AND through an unobserved mediator h that is not part of the
    visible network. The hidden mediator h also affects other observed
    genes downstream of k, creating a perturbation-mediated correlation
    that the full CF's predicted expression x_k_hat captures but x_i_hat
    does not.

    This is the setting under which the full control function reliably
    produces false edges, because the second-stage regression cannot
    screen off the hidden-mediator signal via observed parent expression.

    Parameters
    ----------
    mediator_density : float in [0, 1]
        Fraction of (source perturbation, target gene) pairs that have
        a hidden-mediator bypass.
    mediator_in_effect : float
        Strength of D_source -> h coupling.
    mediator_out_effect : float
        Strength of h -> x_target coupling.
    """
    p = W.shape[0]
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    X = np.zeros([n, p])

    Um = np.random.uniform(-coef, coef, size=(d, p))
    Ur = np.random.uniform(-coef, coef, size=(d, p))
    if intercept:
        Um = np.vstack((theta_m, Um))
        Ur = np.vstack((theta_r, Ur))
    Z = np.random.normal(0, covar, size=(n, d))
    if intercept:
        Z = np.hstack((np.ones((n, 1)), Z))

    ones = np.ones((n, 1))
    Vm = np.random.uniform(-1.5 * shift, -shift, size=(1, p))
    Vr = np.random.uniform(-shift / 2, shift / 2, size=(1, p))

    covar_m = Z @ Um
    covar_r = Z @ Ur
    shift_m = ones @ Vm
    shift_r = ones @ Vr

    intervene_idx = np.random.choice(n, size=int(n * intervene_prob), replace=False)
    intervene_idx.sort()
    intervene_mask = np.zeros((n, p), dtype=bool)
    remaining_nodes = intervene_idx
    each_intervention_size = len(intervene_idx) // p
    for i in range(p):
        choice = np.random.choice(remaining_nodes, size=each_intervention_size, replace=False)
        intervene_mask[choice, i] = True
        remaining_nodes = np.setdiff1d(remaining_nodes, choice)

    D = intervene_mask.astype(float)  # (n, p)

    # --- Hidden mediator structure ---
    # For each pair (source, target), decide whether a hidden mediator
    # exists. We use a sparse mask H_struct[source, target] = 1 if the
    # pair has a hidden bypass.
    H_struct = (np.random.rand(p, p) < mediator_density).astype(float)
    np.fill_diagonal(H_struct, 0)  # no self-mediation

    # For each (source, target) pair with a mediator, sample the
    # mediator's effect on the target (mediator_out) and on other
    # downstream genes. Crucially, the mediator's downstream effect must
    # touch OBSERVED GENES OTHER THAN the source's observed regulatory
    # descendants, so that the false edge has somewhere to land.
    # Simplest version: for each (source, target) with a mediator,
    # sample h ~ N(coupling * D_source, 1), and add h's effect to a
    # randomly chosen "leak gene" downstream.

    # Implementation: we'll directly inject the hidden-mediator effect
    # into a small number of "leak genes" per active source. For each
    # source gene i with a mediator-bypass row, we choose a leak gene k
    # (other than i and its visible descendants) and add an effect
    # proportional to D_i to x_k.

    # Build leak structure: for each source, list of leak target genes
    # that receive hidden-mediator effects.
    leak_targets = {}
    in_coeffs = {}
    out_coeffs = {}
    for source in range(p):
        targets = np.where(H_struct[source] > 0)[0]
        if len(targets) == 0:
            continue
        # Avoid the visible descendants of `source` to ensure the false
        # edge actually corresponds to an unobserved path.
        descendants = G.subcomponent(source, mode=ig.OUT)
        targets = [t for t in targets if t not in descendants]
        if len(targets) == 0:
            continue
        leak_targets[source] = targets
        in_coeffs[source] = np.random.uniform(-mediator_in_effect,
                                               mediator_in_effect)
        out_coeffs[source] = np.random.uniform(-mediator_out_effect,
                                                mediator_out_effect,
                                                size=len(targets))

    # Now generate the hidden mediator values per cell. We allow one
    # hidden mediator per source (parameter-light); h_source(cell) is a
    # noisy linear function of D[source, cell].
    n_sources_with_mediator = len(leak_targets)
    if n_sources_with_mediator > 0:
        source_keys = list(leak_targets.keys())
        H_values = np.zeros((n, len(source_keys)))
        for idx, source in enumerate(source_keys):
            H_values[:, idx] = in_coeffs[source] * D[:, source] + \
                                np.random.normal(0, 0.3, size=n)

        # Compute the hidden-mediator contribution to each observed gene
        hidden_contribution = np.zeros((n, p))
        for idx, source in enumerate(source_keys):
            for t_idx, target in enumerate(leak_targets[source]):
                hidden_contribution[:, target] += \
                    out_coeffs[source][t_idx] * H_values[:, idx]
    else:
        hidden_contribution = np.zeros((n, p))
        
    hidden_contribution = np.clip(hidden_contribution, -5, 5)
    shift_m_masked = shift_m * intervene_mask
    shift_r_masked = shift_r * intervene_mask

    for j_idx in ordered_vertices:
        parents = G.neighbors(j_idx, mode=ig.IN)
        cov_m_j = covar_m[:, j_idx] + hidden_contribution[:, j_idx]
        cov_r_j = covar_r[:, j_idx]
        mask_j = intervene_mask[:, j_idx] if hard_intervention else np.zeros(n)
        X[:, j_idx] = _simulate_single_equation(
            g(X[:, parents]), W[parents, j_idx],
            cov_m_j, cov_r_j,
            mask_j,
            shift_m_masked[:, j_idx], shift_r_masked[:, j_idx],
        ).reshape(-1)

    return X, intervene_mask, Z, Um, Ur, Vm, Vr, leak_targets