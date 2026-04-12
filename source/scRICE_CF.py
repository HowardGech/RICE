import numpy as np
import cupy as cp
from cupyx.scipy.special import digamma, gammaln
import cupyx.scipy.linalg as cpxlin
import cupy.linalg as cplin
import typing
import igraph as ig


class scRICE_CF:
    """
    A Python object that implements single-cell Gene Regulatory Analysis with Negative Binomial model with Perturb-seq data using cupy.
    """
    
    def __init__(self, dtype: type = cp.float64) -> None:
        r"""
        Initialize the scGRAPH_CF object.
        
        Parameters
        ----------
        dtype : type, optional
            The data type to use for computations. Default is `cp.float64`.
        """
        self.dtype = dtype
        if self.dtype == cp.float32:
            clip_value = 30
        elif self.dtype == cp.float64:
            clip_value = 300
        elif self.dtype == cp.float16:
            clip_value = 15
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}. Supported dtypes are cp.float16, cp.float32, and cp.float64.")
        self.clip_value = clip_value
        
    def trace_binsum(self, A, k):
        eigvals = cp.linalg.eigvals(A)
        denominators = cp.arange(1, k + 1)
        powers = cp.arange(1, k + 1)
        
        term_matrix = (cp.power(eigvals[:, None], powers) / denominators).real
        term_matrix = cp.nan_to_num(term_matrix, nan=1e10, posinf=1e10, neginf=-1e10)
        term_matrix = cp.clip(term_matrix, -1e10, 1e10)
        return cp.sum(term_matrix).real
    
    def block_mat_sum(self, A, k):
        n = A.shape[0]
        I = cp.eye(n, dtype=A.dtype)
        Z = cp.zeros((n, n), dtype=A.dtype)
        top = cp.hstack([A, I])
        bottom = cp.hstack([Z, I])
        M = cp.vstack([top, bottom])
        M_pow = cp.linalg.matrix_power(M, k)
        M_pow = cp.nan_to_num(M_pow, nan=1e10, posinf=1e10, neginf=-1e10)
        M_pow = cp.clip(M_pow, -1e10, 1e10)
        sum_part = M_pow[:n, n:]
        
        return  M_pow[:n, n:] + M_pow[:n, :n]

    def _score_grad(self, W: cp.ndarray, 
                    Ur_g: cp.ndarray,
                    Um_g: cp.ndarray,
                    Vr: cp.ndarray,
                    Vm: cp.ndarray,
                    Ur_p: cp.ndarray,
                    Um_p: cp.ndarray,
                    W_c: cp.ndarray
                    ) -> typing.Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:

        
        r_clipped = self.Zr_g @ Ur_g + Vr * self.Gamma + cp.einsum('npd,pd->np', self.Zr_p, Ur_p)
        m_clipped = self.Zm_g @ Um_g + Vm * self.Gamma + (self.gX @ W) * self.mask + cp.einsum('npd,pd->np', self.Zm_p, Um_p) + (self.gX_c @ W_c) * self.mask
        r_clipped = cp.clip(r_clipped, -self.clip_value, self.clip_value)
        m_clipped = cp.clip(m_clipped, -self.clip_value, self.clip_value)
        
        exp_Ur = cp.exp(r_clipped)
        exp_Um = cp.exp(m_clipped)
        weighted_temp_m = self.weights * exp_Ur / (exp_Ur + exp_Um) * (self.X - exp_Um)
        weighted_temp_r = self.weights * (exp_Ur * (cp.log(exp_Ur / (exp_Ur + exp_Um)) + digamma(self.X + exp_Ur) - digamma(exp_Ur))) - weighted_temp_m
        G_loss_Ur_g = - self.Zr_g.T @ weighted_temp_r / self.n 
        G_loss_Um_g = - self.Zm_g.T @ weighted_temp_m / self.n 
        G_loss_Vm = - (weighted_temp_m * self.Gamma).mean(axis=0, keepdims=True) 
        G_loss_Vr = - (weighted_temp_r * self.Gamma).mean(axis=0, keepdims=True) 
        G_loss_W = - self.gX.T @ (weighted_temp_m * self.mask) / self.n 
        G_loss_Ur_p = - cp.einsum('npd,np->pd', self.Zr_p, weighted_temp_r) / self.n 
        G_loss_Um_p = - cp.einsum('npd,np->pd', self.Zm_p, weighted_temp_m) / self.n
        G_loss_W_c = - self.gX_c.T @ (weighted_temp_m * self.mask) / self.n
        return G_loss_W, G_loss_Ur_g, G_loss_Um_g, G_loss_Vr, G_loss_Vm, G_loss_Ur_p, G_loss_Um_p, G_loss_W_c
    
    def _score_value(self, W: cp.ndarray,
                     Ur_g: cp.ndarray,
                     Um_g: cp.ndarray,
                     Vr: cp.ndarray,
                     Vm: cp.ndarray,
                     Ur_p: cp.ndarray,
                    Um_p: cp.ndarray,
                    W_c: cp.ndarray
                     ) -> float:

        
        r_clipped = self.Zr_g @ Ur_g + Vr * self.Gamma + cp.einsum('npd,pd->np', self.Zr_p, Ur_p)
        m_clipped = self.Zm_g @ Um_g + Vm * self.Gamma + (self.gX @ W) * self.mask + cp.einsum('npd,pd->np', self.Zm_p, Um_p) + (self.gX_c @ W_c) * self.mask
        r_clipped = cp.clip(r_clipped, -self.clip_value, self.clip_value)
        m_clipped = cp.clip(m_clipped, -self.clip_value, self.clip_value)
        
        exp_Ur = cp.exp(r_clipped)
        exp_Um = cp.exp(m_clipped)
        neg_loglik = - exp_Ur * cp.log(exp_Ur / (exp_Ur + exp_Um)) - self.X * cp.log(exp_Um / (exp_Ur + exp_Um)) - gammaln(exp_Ur + self.X) + gammaln(exp_Ur)
        score = cp.sum(neg_loglik * self.weights) / self.n 
        return score

    def _h_grad(self, W: cp.ndarray, s: float = 1.0, reg: str = 'logdet') -> cp.ndarray:

        if reg == 'expm':
            E = cpxlin.expm(W * W)
            G_h = 2 * W * E.T
        elif reg == 'logdet':
            M = s * self.Id - W * W
            G_h = 2 * W * cplin.inv(M).T
        elif reg == 'binsum':
            M = self.block_mat_sum(W * W, self.binsum_k-1)
            G_h = 2 * W * M.T
        elif reg == 'nodag':
            G_h = cp.zeros_like(W)
        return G_h
    

    
    def _h_value(self, W: cp.ndarray, s: float = 1.0, reg: str = 'logdet') -> float:

        if reg == 'logdet':
            M = s * self.Id - W * W
            h = - cplin.slogdet(M)[1] + self.p * np.log(s)
        elif reg == 'expm':
            E = cpxlin.expm(W * W)
            h = cp.trace(E) - self.p
        elif reg == 'binsum':
            h = self.trace_binsum(W * W, self.binsum_k)
        elif reg == 'nodag':
            h = 0
        return h
    
    
        

    def _func(self,  W: cp.ndarray, 
              Ur_g: cp.ndarray,
                Um_g: cp.ndarray,
                Vr: cp.ndarray,
                Vm: cp.ndarray,
                Ur_p: cp.ndarray,
                Um_p: cp.ndarray,
                W_c: cp.ndarray,
                s: float = 1.0, reg: str = 'logdet') -> typing.Tuple[float, cp.ndarray]:

        
        score = self._score_value(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c)
        h = self._h_value(W, s, reg)
        obj = score + self.pen1 * cp.abs(W).sum() + self.alpha * h + 0.5 * self.rho * h ** 2  + self.pen2 * (W**2).sum() + self.pen1_c * cp.abs(W_c).sum() + self.pen2_c * (W_c**2).sum()
        return obj, score, h
    
    def _adam_update(self, grad: cp.ndarray, iter: int,  param: str) -> cp.ndarray:

        
        beta_1 = self.optim_param[0]
        beta_2 = self.optim_param[1]
        self.opt_m[param] = self.opt_m[param] * beta_1 + (1 - beta_1) * grad
        self.opt_v[param] = self.opt_v[param] * beta_2 + (1 - beta_2) * (grad ** 2)
        m_hat = self.opt_m[param] / (1 - beta_1 ** iter)
        v_hat = self.opt_v[param] / (1 - beta_2 ** iter)
        grad = m_hat / (cp.sqrt(v_hat) + 1e-8)
        return grad
    

    
    def optimize_param(self, 
                 W: cp.ndarray,
                 Ur_g: cp.ndarray,
                 Um_g: cp.ndarray,
                 Vr: cp.ndarray,
                 Vm: cp.ndarray,
                 Ur_p: cp.ndarray,
                Um_p: cp.ndarray,
                W_c: cp.ndarray,
                 epochs: int, 
                 s: float, 
                 lr: float, 
                 tol: float = 1e-6,
                 reg: str = 'logdet',
                 ) -> typing.Tuple[cp.ndarray, bool]:        

        
        obj_prev = 1e16

            

        for epoch in range(1, epochs+1):
            ## Compute the (sub)gradient of the objective
            if reg == 'logdet':
                M_inv = s * self.Id - W * W
                M = cplin.inv(s * self.Id - W * W) + 1e-16
                while cp.any(M < 0):
                    if epoch == 1 or s <= 0.9:
                        return W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, False
                    else:
                        self.vprint(f'Negative eigenvalues detected in M. Reducing lr from {lr:.2e} to {lr*0.5:.2e} and restarting optimization.')
                        W += lr * grad_W
                        lr *= .5
                        if lr <= 1e-16:
                            return W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, False
                        W -= lr * grad_W
                        M_inv = s * self.Id - W * W
                        M = cplin.inv(s * self.Id - W * W)
                augL_h = - cplin.slogdet(M_inv)[1] + self.p  * np.log(s)
                G_augL_h = 2 * W * M.T 
            elif reg == 'expm':
                E = cpxlin.expm(W * W)
                augL_h = cp.trace(E) - self.p
                G_augL_h = 2 * W * E.T
            elif reg == 'binsum':
                M = self.block_mat_sum(W * W, self.binsum_k-1)
                augL_h = self.trace_binsum(W * W, self.binsum_k)
                G_augL_h = 2 * W * M.T
            elif reg == 'nodag':
                augL_h = 0
                G_augL_h = cp.zeros_like(W)
                
            G_score_W, G_score_Ur_g, G_score_Um_g, G_score_Vr, G_score_Vm, G_score_Ur_p, G_score_Um_p, G_score_W_c = self._score_grad(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c)
                

            Gobj_W = self.pen1 * cp.sign(W) + self.rho * augL_h * G_augL_h + self.alpha * G_augL_h + self.pen2 * 2 * W
            Gobj_W += G_score_W
            
            G_score_W_c += self.pen1_c * cp.sign(W_c) + self.pen2_c * 2 * W_c
            ## Adam step

            grad_W = self._adam_update(Gobj_W, epoch, 'W')
            grad_Ur_g = self._adam_update(G_score_Ur_g, epoch, 'Ur_g')
            grad_Um_g = self._adam_update(G_score_Um_g, epoch, 'Um_g')
            grad_Vr = self._adam_update(G_score_Vr, epoch, 'Vr')
            grad_Vm = self._adam_update(G_score_Vm, epoch, 'Vm')
            grad_Ur_p = self._adam_update(G_score_Ur_p, epoch, 'Ur_p')
            grad_Um_p = self._adam_update(G_score_Um_p, epoch, 'Um_p')
            grad_W_c = self._adam_update(G_score_W_c, epoch, 'W_c')

            
            
            W -= lr * grad_W
            Ur_g -= self.lr_coef_multiplier * lr * grad_Ur_g
            Um_g -= self.lr_coef_multiplier * lr * grad_Um_g
            Vr -= self.lr_coef_multiplier * lr * grad_Vr
            Vm -= self.lr_coef_multiplier * lr * grad_Vm
            Ur_p -= self.lr_coef_multiplier * lr * grad_Ur_p
            Um_p -= self.lr_coef_multiplier * lr * grad_Um_p
            W_c -= lr * grad_W_c
            
            cp.fill_diagonal(W, 0)
            cp.fill_diagonal(W_c, 0)
                
            ## Check obj convergence
            if epoch % self.checkpoint == 0 or epoch == epochs:
                obj_new, score, h = self._func(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, s, reg)
                delta = cp.abs((obj_prev - obj_new) / cp.abs(obj_prev)+1e-8)
                self.delta = delta
                if self.progress_bar:
                    self.pbar.set_postfix({'epoch': f'{epoch}/{epochs}','obj': f'{obj_new:.2e}','Δ': f'{delta:.2e}'})
                if delta <= tol:
                    break
                obj_prev = obj_new
        return W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, True
    

    
    def run(self, 
            max_iter: int = 100,
            epochs: int = 1e4,
            lr: float = 0.0003,
            rho_max: float = 1e16,
            h_tol: float = 1e-4,
            loss_tol: float = 1e-6,
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6],
            reg: str = 'logdet') -> None:
        
        
        self.rho, self.alpha, h = self.rho_init, 0.0, np.inf
        for i in range(max_iter):
            if self.progress_bar:
                self.pbar.update(1)
                self.pbar.set_description(f'reg={reg}, ρ={self.rho:.2e}, α={self.alpha:.2e}, h={h:.2e}')
            # self.vprint(f'Iteration {i+1}/{max_iter} -- ρ: {self.rho:.2e} -- α: {self.alpha:.2e} -- h: {h:.2e}')
            
            lr_adam, success = lr, False
            while success is False and self.rho <= rho_max:
                self.pbar.set_description(f'reg={reg}, ρ={self.rho:.2e}, α={self.alpha:.2e}, h={h:.2e}')
                W_temp, Ur_g_temp, Um_g_temp, Vr_temp, Vm_temp, Ur_p_temp, Um_p_temp, W_c_temp, success =  self.optimize_param(self.W.copy(), self.Ur_g.copy(), self.Um_g.copy(), self.Vr.copy(), self.Vm.copy(), self.Ur_p.copy(), self.Um_p.copy(), self.W_c.copy(), epochs=epochs, s=s[i], tol=loss_tol, lr=lr_adam, reg=reg)
                if success is False:
                    self.vprint(f'Optimization failed with lr={lr_adam:.2e} and s={s[i]:.2f}. Increasing s.')
                    # lr_adam *= 0.5
                    s[i] += 0.1
                h_new  = self._h_value(W_temp, s[i], reg=reg)
                if success and cp.abs(h_new) > 0.5*cp.abs(h):
                    self.rho *= self.rho_multiplier

            self.W, h, self.W_c = W_temp, h_new, W_c_temp
            self.Ur_g, self.Um_g, self.Vr, self.Vm, self.Ur_p, self.Um_p = Ur_g_temp, Um_g_temp, Vr_temp, Vm_temp, Ur_p_temp, Um_p_temp
            self.alpha += self.rho * h
            if cp.abs(h) <= h_tol or self.rho > rho_max:
                if self.rho > rho_max:
                    self.vprint(f'Max rho reached. Stopping optimization.')
                else:
                    self.vprint(f'h tolerance reached. h = {h}. Stopping optimization.')
                break
        if self.progress_bar:
            self.pbar.set_description(f'reg={reg}, ρ={self.rho:.2e}, α={self.alpha:.2e}, h={h:.2e}')
            self.pbar.update(max_iter-i-1)
        
        
    def prep(self, X: np.ndarray,
                intervention_effect: np.ndarray,
                confounding_res: typing.Optional[np.ndarray] = None,
                  predictor: typing.Optional[np.ndarray] = None,
                  intervention_type: str = 'soft',
                  Zm_g: typing.Optional[np.ndarray] = None,
                  Zr_g: typing.Optional[np.ndarray] = None,
                  Zm_p: typing.Optional[np.ndarray] = None,
                  Zr_p: typing.Optional[np.ndarray] = None,
                  add_intercept: bool = True,
                  Gamma: typing.Optional[np.ndarray] = None,
                  weights: typing.Optional[np.ndarray] = None,
                  ) -> None:
        
        self.X = cp.asarray(X, dtype=self.dtype)
        
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        
        
        if predictor is None:
            print("No predictor provided, using X as predictor.")
            self.gX = self.X.copy()
        else:
            assert isinstance(predictor, np.ndarray) or isinstance(predictor, cp.ndarray), "predictor should be a numpy or cupy array"
            assert predictor.shape == self.X.shape, "predictor should have the same size as X"
            self.gX = cp.asarray(predictor, dtype=self.dtype)
            
        if confounding_res is None:
            print("No confounding_residual provided, using zeros.")
            self.gX_c = cp.zeros_like(self.X, dtype=self.dtype)
        else:
            assert isinstance(confounding_res, np.ndarray) or isinstance(confounding_res, cp.ndarray), "confounding residual should be a numpy or cupy array"
            assert confounding_res.shape == self.X.shape, "confounding residual should have the same size as X"
            self.gX_c = cp.asarray(confounding_res, dtype=self.dtype)
        

        if intervention_type not in ['soft', 'hard']:
            raise ValueError("intervention_type should be either 'soft' or 'hard'")

        assert intervention_effect.shape == self.X.shape, "intervention_effect should have the same size as response"
        assert (intervention_effect!=0).sum(axis=0).min() > 0, "need at least one intervention for each column"
        assert (intervention_effect!=0).sum(axis=1).min() == 0, "need at least one control samples"
            
        if Zm_g is None:
            self.Zm_g = cp.zeros((self.X.shape[0], 0), dtype=self.dtype)
        else:
            assert isinstance(Zm_g, np.ndarray) or isinstance(Zm_g, cp.ndarray), "Zm_g should be a numpy or cupy array"
            assert Zm_g.shape[0] == self.X.shape[0], "Zm_g should have the same rows as X"
            self.Zm_g = cp.asarray(Zm_g, dtype=self.dtype)
        if add_intercept:
            self.Zm_g = cp.hstack((cp.ones((self.Zm_g.shape[0], 1), dtype=self.dtype), self.Zm_g))
        self.dm_g = self.Zm_g.shape[1]
        
        
        if Zr_g is None:
            self.Zr_g = cp.zeros((self.X.shape[0], 0), dtype=self.dtype)
        else:
            assert isinstance(Zr_g, np.ndarray) or isinstance(Zr_g, cp.ndarray), "Zr_g should be a numpy or cupy array"
            assert Zr_g.shape[0] == self.X.shape[0], "Zr_g should have the same rows as X"
            self.Zr_g = cp.asarray(Zr_g, dtype=self.dtype)
        if add_intercept:
            self.Zr_g = cp.hstack((cp.ones((self.Zr_g.shape[0], 1), dtype=self.dtype), self.Zr_g))
        self.dr_g = self.Zr_g.shape[1]
            
        if Zm_p is None:
            self.Zm_p = cp.zeros((self.X.shape[0], self.p, 0), dtype=self.dtype)
        else:
            assert isinstance(Zm_p, np.ndarray) or isinstance(Zm_p, cp.ndarray), "Zm_p should be a numpy or cupy array"
            assert Zm_p.shape[0] == self.X.shape[0], "Zm_p should have the same rows as X"
            assert Zm_p.shape[1] == self.X.shape[1], "Zm_p should have the same number of columns as X in the second dimension"
            self.Zm_p = cp.asarray(Zm_p, dtype=self.dtype)
        self.dm_p = self.Zm_p.shape[2]
        
        if Zr_p is None:
            self.Zr_p = cp.zeros((self.X.shape[0], self.p, 0), dtype=self.dtype)
        else:
            assert isinstance(Zr_p, np.ndarray) or isinstance(Zr_p, cp.ndarray), "Zr_p should be a numpy or cupy array"
            assert Zr_p.shape[0] == self.X.shape[0], "Zr_p should have the same rows as X"
            assert Zr_p.shape[1] == self.X.shape[1], "Zr_p should have the same number of columns as X in the second dimension"
            self.Zr_p = cp.asarray(Zr_p, dtype=self.dtype)
        self.dr_p = self.Zr_p.shape[2]
        
        self.Gamma = cp.asarray(intervention_effect, dtype=self.dtype)
        

        if intervention_type == 'hard':
            self.mask = (self.Gamma == 0)
        else:
            self.mask = cp.ones_like(self.Gamma, dtype=self.dtype)
            
            
        if weights is None:
            weights = cp.ones(self.X.shape[0], dtype=self.dtype)
        else:
            assert isinstance(weights, np.ndarray) or isinstance(weights, cp.ndarray), "weights should be a numpy or cupy array"
            assert weights.shape[0] == self.X.shape[0], "weights should have the same number of rows as X"
            assert weights.ndim == 1, "weights should be a 1-dimensional array"
        self.weights = cp.asarray(weights, dtype=self.dtype).reshape(-1,1)

        
   
        self.Ur_g = cp.zeros((self.dr_g, self.p), dtype=self.dtype)
        self.Um_g = cp.zeros((self.dm_g, self.p), dtype=self.dtype)
        self.Vr = cp.zeros((1, self.p), dtype=self.dtype)
        self.Vm = cp.zeros((1, self.p), dtype=self.dtype)
        self.Ur_p = cp.zeros((self.p, self.dr_p), dtype=self.dtype)
        self.Um_p = cp.zeros((self.p, self.dm_p), dtype=self.dtype)
        self.W = cp.zeros((self.p, self.p), dtype=self.dtype)
        self.W_c = cp.zeros((self.p, self.p), dtype=self.dtype)
        self.Id = cp.eye(self.p, dtype=self.dtype)
        
        self.opt_m = {'Ur_g': 0, 'Um_g': 0, 'Vr': 0, 'Vm': 0, 'Ur_p': 0, 'Um_p': 0, 'W': 0, 'W_c': 0}
        self.opt_v = {'Ur_g': 0, 'Um_g': 0, 'Vr': 0, 'Vm': 0, 'Ur_p': 0, 'Um_p': 0, 'W': 0, 'W_c': 0}
    
        
    
    def fit(self, 
            pen1: float = 0.03, 
            pen2 : float = 0.03,
            pen1_c: float = 0.0,
            pen2_c: float = 0.0,
            regularizer: str = 'logdet',
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            epochs: int = 1e4,
            max_iter: int = 100,
            lr: float = 0.0003, 
            lr_coef_multiplier: float = 1.0,
            rho_init: float = 1.0,
            checkpoint: int = 2000, 
            optim_param: typing.Tuple[float, float] = (0.9, 0.99),
            rho_max: float = 1e16,
            rho_multiplier: float = 10.0,
            h_tol: float = 1e-3,
            loss_tol: float = 1e-6,
            binsum_k: int = 7,
            verbose: bool = False,
            progress_bar: bool = True
        ) -> None:
        
        
        ## INITALIZING VARIABLES 


        valid_regularizers = ['expm', 'logdet', 'binsum', 'nodag']
        assert regularizer in valid_regularizers, f"regularizer should be in {valid_regularizers}"
            

        self.rho_init, self.lr_coef_multiplier, self.binsum_k = rho_init, lr_coef_multiplier, binsum_k
        if type(s) == list:
            if len(s) < max_iter: 
                s = s + (max_iter - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = max_iter * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        ## START ALGORITHM
        epochs = int(epochs)
        max_iter = int(max_iter)
        self.optim_param, self.checkpoint, self.pen1, self.pen2, self.pen1_c, self.pen2_c, self.rho_multiplier = optim_param, int(checkpoint), pen1, pen2, pen1_c, pen2_c, rho_multiplier
        
        if verbose:
            self.vprint = print
        else:
            self.vprint = lambda *args, **kwargs: None
        if progress_bar:
            from tqdm.auto import tqdm
            self.pbar = tqdm(total=(max_iter))
        self.progress_bar = progress_bar
        
        self.run(max_iter=max_iter, epochs=epochs, lr=lr, rho_max=rho_max, h_tol=h_tol, loss_tol=loss_tol, s=s, reg=regularizer)
        if self.progress_bar:
            self.pbar.close()
                
        
    def result(self, w_threshold: float = 0.0) -> typing.Dict[str, cp.ndarray]:
        
        
        W = self.W.copy()
        Ur_g = cp.asnumpy(self.Ur_g)
        Um_g = cp.asnumpy(self.Um_g)
        Vr = cp.asnumpy(self.Vr)
        Vm = cp.asnumpy(self.Vm)
        Ur_p = cp.asnumpy(self.Ur_p)
        Um_p = cp.asnumpy(self.Um_p)
        W[cp.abs(W) < w_threshold] = 0.0
        W = cp.asnumpy(W)
        W_c = self.W_c.copy()
        
        return {
            'W': W,
            'Ur_g': Ur_g,
            'Um_g': Um_g,
            'Vr': Vr,
            'Vm': Vm,
            'Ur_p': Ur_p,
            'Um_p': Um_p,
            'W_c': W_c
        }
        
    def fit_path(self, 
            pen1_list: typing.List[float],
            pen2 : float = 0.03,
            pen1_c: float = 0.0,
            pen2_c: float = 0.0,
            regularizer: str = 'logdet',
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            epochs: int = 1e4,
            max_iter: int = 100,
            lr: float = 0.0003, 
            lr_coef_multiplier: float = 1.0,
            rho_init: float = 1.0,
            checkpoint: int = 2000, 
            optim_param: typing.Tuple[float, float] = (0.9, 0.99),
            rho_max: float = 1e16,
            rho_multiplier: float = 10.0,
            h_tol: float = 1e-3,
            loss_tol: float = 1e-6,
            binsum_k: int = 7,
            verbose: bool = False,
            progress_bar: bool = True
        ) -> typing.List[typing.Dict[str, cp.ndarray]]:
        results = dict()
        for pen1 in pen1_list:
            self.opt_m = {'Ur_g': 0, 'Um_g': 0, 'Vr': 0, 'Vm': 0, 'Ur_p': 0, 'Um_p': 0, 'W': 0, 'W_c': 0}
            self.opt_v = {'Ur_g': 0, 'Um_g': 0, 'Vr': 0, 'Vm': 0, 'Ur_p': 0, 'Um_p': 0, 'W': 0, 'W_c': 0}
            self.fit(pen1=pen1, pen2=pen2, pen1_c=pen1_c, pen2_c=pen2_c, regularizer=regularizer, s=s, epochs=epochs, max_iter=max_iter, lr=lr, lr_coef_multiplier=lr_coef_multiplier, rho_init=rho_init, checkpoint=checkpoint, optim_param=optim_param, rho_max=rho_max, rho_multiplier=rho_multiplier, h_tol=h_tol, loss_tol=loss_tol, binsum_k=binsum_k, verbose=verbose, progress_bar=progress_bar)
            results[pen1] = self.result()
            results[pen1]['h'] = self._h_value(self.W, s=s[0], reg=regularizer)
            results[pen1]['score'] = self._score_value(self.W, self.Ur_g, self.Um_g, self.Vr, self.Vm, self.Ur_p, self.Um_p, self.W_c)
        return results
            
        

