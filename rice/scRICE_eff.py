"""
scRICE with learnable per-gene knockdown efficacy.

Extends the published scRICE_CF model by replacing the binary soft/hard
intervention switch with a continuous per-gene parameter:

    Dtilde_ij = 1 - (1 - xi_j) * D_ij,      xi_j = sigmoid(eta_j) in (0, 1)

    xi_j = 0  ->  hard intervention (parents fully overwritten)
    xi_j = 1  ->  soft intervention (parents fully retained)

Naming: the manuscript writes the mask as D^e_j = 1 - D_j e_j with e_j the
perturbation EFFICACY (e_j = 1 hard, e_j = 0 soft). This module's free parameter
is the complement, xi_j = 1 - e_j (residual parent influence), because that is
what multiplies the parent sum directly. `result()` returns BOTH `xi` and `e`.

`xi_scope` selects what xi multiplies. 'both' (the strict generalization of
Eq. 3) scales the parent sum and the CF residual sum together; 'parent' scales
only the parent sum and leaves the CF term unmasked. The choice matters: with
'both', xi is estimated with a substantial upward bias because the model can
trade the xi-scaled CF channel against the parent channel.

xi is RESIDUAL PARENT INFLUENCE, i.e. xi_j = 1 - efficacy_j: a gene knocked
down with 100% efficacy has xi_j = 0, one at 50% efficacy has xi_j = 0.5.
As in the published model, Dtilde multiplies BOTH the parent sum and the
control-function residual sum, so eta_j -> -inf and eta_j -> +inf reproduce
the original hard and soft objectives exactly.

The unconstrained parameter is named `Eta` (not gamma) because `Gamma` is
already the (n, p) perturbation-indicator matrix in the published class.

This module is backend-agnostic: pass backend='numpy' to run on CPU (used to
finite-difference the gradients) or backend='cupy' to run on GPU. The maths is
identical; only the array module differs.
"""

import numpy as np
import typing


def _resolve_backend(backend: str):
    if backend == 'auto':
        try:
            import cupy  # noqa: F401
            backend = 'cupy'
        except Exception:
            backend = 'numpy'
    if backend == 'cupy':
        import cupy as xp
        from cupyx.scipy.special import digamma, gammaln
        from cupyx.scipy.linalg import expm
        return xp, digamma, gammaln, expm, xp.asnumpy
    elif backend == 'numpy':
        import numpy as xp
        from scipy.special import digamma, gammaln
        from scipy.linalg import expm
        return xp, digamma, gammaln, expm, (lambda a: np.asarray(a))
    raise ValueError("backend must be 'auto', 'numpy' or 'cupy'")


class scRICE_CF_eff:
    """
    Negative-binomial causal network model with reduced control function and
    learnable per-gene knockdown efficacy.

    Usage mirrors the published class: prep(...) -> fit(...) -> result(...).
    Set intervention_type='efficacy' in prep() to learn xi; 'soft' and 'hard'
    reproduce the fixed-mask behaviour of the original model.
    """

    def __init__(self, dtype=None, backend: str = 'auto') -> None:
        self.xp, self._digamma, self._gammaln, self._expm, self._tonumpy = \
            _resolve_backend(backend)
        xp = self.xp
        self.dtype = xp.float64 if dtype is None else dtype
        if self.dtype == xp.float32:
            self.clip_value = 30
        elif self.dtype == xp.float64:
            self.clip_value = 300
        elif self.dtype == xp.float16:
            self.clip_value = 15
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    # ---------------------------------------------------------------- masks

    def _mask_from_eta(self, Eta):
        """Continuous Dtilde. Falls back to the fixed mask when not learning xi."""
        if not self.learn_xi:
            return self.mask
        xp = self.xp
        xi = 1.0 / (1.0 + xp.exp(-xp.clip(Eta, -self.eta_clip, self.eta_clip)))
        return 1.0 - (1.0 - xi) * self.D

    def _xi(self, Eta):
        xp = self.xp
        return 1.0 / (1.0 + xp.exp(-xp.clip(Eta, -self.eta_clip, self.eta_clip)))

    # ------------------------------------------------------- score & grads

    def _lin(self, W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta):
        """Linear predictors for dispersion (r) and mean (m), plus reusables.

        `A` is the part of the mean that xi actually scales, and is therefore
        what the Eta gradient contracts against. With xi_scope='both' that is
        the parent sum plus the CF residual sum; with 'parent' the CF term is
        carried at a fixed mask of ones and excluded from A.
        """
        xp = self.xp
        Mc = self._mask_from_eta(Eta)
        A_par = self.gX @ W
        A_cf = self.gX_c @ W_c
        if self.xi_scope == 'both':
            A = A_par + A_cf
            Mc_cf = Mc
        else:
            A = A_par
            Mc_cf = self.cf_mask
        self._Mc_cf = Mc_cf
        r_lin = self.Zr_g @ Ur_g + Vr * self.Gamma \
            + xp.einsum('npd,pd->np', self.Zr_p, Ur_p)
        m_lin = self.Zm_g @ Um_g + Vm * self.Gamma + A_par * Mc + A_cf * Mc_cf \
            + xp.einsum('npd,pd->np', self.Zm_p, Um_p)
        r_lin = xp.clip(r_lin, -self.clip_value, self.clip_value)
        m_lin = xp.clip(m_lin, -self.clip_value, self.clip_value)
        return Mc, A, xp.exp(r_lin), xp.exp(m_lin)

    def _score_value(self, W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta) -> float:
        xp = self.xp
        _, _, rr, mm = self._lin(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta)
        nll = (-rr * xp.log(rr / (rr + mm))
               - self.X * xp.log(mm / (rr + mm))
               - self._gammaln(rr + self.X) + self._gammaln(rr))
        return xp.sum(nll * self.weights) / self.n

    def _score_grad(self, W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta):
        xp = self.xp
        Mc, A, rr, mm = self._lin(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta)
        wtm = self.weights * rr / (rr + mm) * (self.X - mm)
        wtr = self.weights * (rr * (xp.log(rr / (rr + mm))
                                    + self._digamma(self.X + rr)
                                    - self._digamma(rr))) - wtm
        G_Ur_g = -self.Zr_g.T @ wtr / self.n
        G_Um_g = -self.Zm_g.T @ wtm / self.n
        G_Vm = -(wtm * self.Gamma).mean(axis=0, keepdims=True)
        G_Vr = -(wtr * self.Gamma).mean(axis=0, keepdims=True)
        G_W = -self.gX.T @ (wtm * Mc) / self.n
        G_W_c = -self.gX_c.T @ (wtm * self._Mc_cf) / self.n
        G_Ur_p = -xp.einsum('npd,np->pd', self.Zr_p, wtr) / self.n
        G_Um_p = -xp.einsum('npd,np->pd', self.Zm_p, wtm) / self.n
        if self.learn_xi:
            xi = self._xi(Eta)
            # d m_ij / d eta_j = A_ij * D_ij * xi_j * (1 - xi_j)
            G_Eta = -(xi * (1.0 - xi)) \
                * (wtm * A * self.D).sum(axis=0, keepdims=True) / self.n
        else:
            G_Eta = xp.zeros_like(Eta)
        return (G_W, G_Ur_g, G_Um_g, G_Vr, G_Vm, G_Ur_p, G_Um_p, G_W_c, G_Eta)

    # ------------------------------------------------------ acyclicity term

    def _h_value(self, W, s: float = 1.0, reg: str = 'expm') -> float:
        xp = self.xp
        if reg == 'expm':
            return xp.trace(self._expm(W * W)) - self.p
        elif reg == 'logdet':
            M = s * self.Id - W * W
            return -xp.linalg.slogdet(M)[1] + self.p * np.log(s)
        elif reg == 'nodag':
            return 0.0
        raise ValueError(f'unknown reg {reg}')

    def _h_grad(self, W, s: float = 1.0, reg: str = 'expm'):
        xp = self.xp
        if reg == 'expm':
            E = self._expm(W * W)
            return 2 * W * E.T
        elif reg == 'logdet':
            M = s * self.Id - W * W
            return 2 * W * xp.linalg.inv(M).T
        elif reg == 'nodag':
            return xp.zeros_like(W)
        raise ValueError(f'unknown reg {reg}')

    def _func(self, W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta,
              s: float = 1.0, reg: str = 'expm'):
        xp = self.xp
        score = self._score_value(W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta)
        h = self._h_value(W, s, reg)
        obj = (score + self.pen1 * xp.abs(W).sum() + self.alpha * h
               + 0.5 * self.rho * h ** 2 + self.pen2 * (W ** 2).sum()
               + self.pen1_c * xp.abs(W_c).sum() + self.pen2_c * (W_c ** 2).sum())
        return obj, score, h

    # -------------------------------------------------------------- optimiser

    def _adam_update(self, grad, it: int, param: str):
        xp = self.xp
        b1, b2 = self.optim_param
        self.opt_m[param] = self.opt_m[param] * b1 + (1 - b1) * grad
        self.opt_v[param] = self.opt_v[param] * b2 + (1 - b2) * (grad ** 2)
        m_hat = self.opt_m[param] / (1 - b1 ** it)
        v_hat = self.opt_v[param] / (1 - b2 ** it)
        return m_hat / (xp.sqrt(v_hat) + 1e-8)

    def optimize_param(self, W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta,
                       epochs: int, s: float, lr: float, tol: float = 1e-6,
                       reg: str = 'expm'):
        xp = self.xp
        obj_prev = 1e16
        for epoch in range(1, epochs + 1):
            augL_h = self._h_value(W, s, reg)
            G_augL_h = self._h_grad(W, s, reg)
            (G_W, G_Ur_g, G_Um_g, G_Vr, G_Vm,
             G_Ur_p, G_Um_p, G_W_c, G_Eta) = self._score_grad(
                W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta)

            Gobj_W = (G_W + self.pen1 * xp.sign(W)
                      + self.rho * augL_h * G_augL_h + self.alpha * G_augL_h
                      + self.pen2 * 2 * W)
            Gobj_W_c = G_W_c + self.pen1_c * xp.sign(W_c) + self.pen2_c * 2 * W_c

            W = W - lr * self._adam_update(Gobj_W, epoch, 'W')
            W_c = W_c - lr * self._adam_update(Gobj_W_c, epoch, 'W_c')
            m = self.lr_coef_multiplier * lr
            Ur_g = Ur_g - m * self._adam_update(G_Ur_g, epoch, 'Ur_g')
            Um_g = Um_g - m * self._adam_update(G_Um_g, epoch, 'Um_g')
            Vr = Vr - m * self._adam_update(G_Vr, epoch, 'Vr')
            Vm = Vm - m * self._adam_update(G_Vm, epoch, 'Vm')
            Ur_p = Ur_p - m * self._adam_update(G_Ur_p, epoch, 'Ur_p')
            Um_p = Um_p - m * self._adam_update(G_Um_p, epoch, 'Um_p')
            if self.learn_xi:
                Eta = Eta - self.lr_eta_multiplier * lr * \
                    self._adam_update(G_Eta, epoch, 'Eta')
                Eta = xp.clip(Eta, -self.eta_clip, self.eta_clip)

            xp.fill_diagonal(W, 0)
            xp.fill_diagonal(W_c, 0)

            if epoch % self.checkpoint == 0 or epoch == epochs:
                obj_new, score, h = self._func(W, Ur_g, Um_g, Vr, Vm,
                                               Ur_p, Um_p, W_c, Eta, s, reg)
                delta = abs((obj_prev - obj_new) / (abs(obj_prev) + 1e-8))
                self.delta = delta
                if self.progress_bar:
                    self.pbar.set_postfix({'epoch': f'{epoch}/{epochs}',
                                           'obj': f'{float(obj_new):.2e}',
                                           'd': f'{float(delta):.2e}'})
                if delta <= tol:
                    break
                obj_prev = obj_new
        return W, Ur_g, Um_g, Vr, Vm, Ur_p, Um_p, W_c, Eta

    def run(self, max_iter=100, epochs=1e4, lr=1e-4, rho_max=1e16,
            h_tol=1e-2, loss_tol=1e-6, s=1.0, reg='expm') -> None:
        self.rho, self.alpha, h = self.rho_init, 0.0, np.inf
        s_list = s if isinstance(s, list) else [s] * int(max_iter)
        if len(s_list) < max_iter:
            s_list = s_list + [s_list[-1]] * (int(max_iter) - len(s_list))
        for i in range(int(max_iter)):
            if self.progress_bar:
                self.pbar.update(1)
                self.pbar.set_description(
                    f'reg={reg}, rho={self.rho:.2e}, a={self.alpha:.2e}, h={float(h):.2e}')
            (self.W, self.Ur_g, self.Um_g, self.Vr, self.Vm,
             self.Ur_p, self.Um_p, self.W_c, self.Eta) = self.optimize_param(
                self.W.copy(), self.Ur_g.copy(), self.Um_g.copy(),
                self.Vr.copy(), self.Vm.copy(), self.Ur_p.copy(),
                self.Um_p.copy(), self.W_c.copy(), self.Eta.copy(),
                epochs=int(epochs), s=s_list[i], tol=loss_tol, lr=lr, reg=reg)
            h_new = self._h_value(self.W, s_list[i], reg)
            if abs(h_new) > 0.5 * abs(h):
                self.rho *= self.rho_multiplier
            h = h_new
            self.alpha += self.rho * h
            if abs(h) <= h_tol or self.rho > rho_max:
                break

    # ------------------------------------------------------------------ prep

    def prep(self, X, intervention_effect, confounding_res=None, predictor=None,
             intervention_type: str = 'efficacy', Zm_g=None, Zr_g=None,
             Zm_p=None, Zr_p=None, add_intercept: bool = True,
             weights=None, eta_init: float = 0.0, eta_clip: float = 8.0,
             xi_scope: str = 'both') -> None:
        xp = self.xp
        asa = lambda a: xp.asarray(a, dtype=self.dtype)

        self.X = asa(X)
        self.n, self.p = self.X.shape
        self.gX = self.X.copy() if predictor is None else asa(predictor)
        self.gX_c = (xp.zeros_like(self.X) if confounding_res is None
                     else asa(confounding_res))
        assert self.gX.shape == self.X.shape
        assert self.gX_c.shape == self.X.shape
        assert intervention_effect.shape == self.X.shape
        assert (intervention_effect != 0).sum(axis=0).min() > 0, \
            'need at least one intervention for each gene'
        assert (intervention_effect != 0).sum(axis=1).min() == 0, \
            'need at least one control cell'

        if intervention_type not in ('soft', 'hard', 'efficacy'):
            raise ValueError("intervention_type must be 'soft', 'hard' or 'efficacy'")

        def _mkZg(Z):
            Z = xp.zeros((self.n, 0), dtype=self.dtype) if Z is None else asa(Z)
            if add_intercept:
                Z = xp.hstack((xp.ones((self.n, 1), dtype=self.dtype), Z))
            return Z

        self.Zm_g, self.Zr_g = _mkZg(Zm_g), _mkZg(Zr_g)
        self.Zm_p = (xp.zeros((self.n, self.p, 0), dtype=self.dtype)
                     if Zm_p is None else asa(Zm_p))
        self.Zr_p = (xp.zeros((self.n, self.p, 0), dtype=self.dtype)
                     if Zr_p is None else asa(Zr_p))

        self.Gamma = asa(intervention_effect)
        self.D = (self.Gamma != 0).astype(self.dtype)
        self.eta_clip = eta_clip
        if xi_scope not in ('both', 'parent'):
            raise ValueError("xi_scope must be 'both' or 'parent'")
        self.xi_scope = xi_scope
        self.cf_mask = xp.ones_like(self.Gamma, dtype=self.dtype)
        self.learn_xi = (intervention_type == 'efficacy')
        if intervention_type == 'hard':
            self.mask = (self.Gamma == 0).astype(self.dtype)
        else:
            self.mask = xp.ones_like(self.Gamma, dtype=self.dtype)
        if not self.learn_xi:
            # published model applies the same fixed mask to the CF term
            self.xi_scope = 'both'
            self.cf_mask = self.mask

        self.weights = (xp.ones((self.n, 1), dtype=self.dtype) if weights is None
                        else asa(weights).reshape(-1, 1))

        self.Ur_g = xp.zeros((self.Zr_g.shape[1], self.p), dtype=self.dtype)
        self.Um_g = xp.zeros((self.Zm_g.shape[1], self.p), dtype=self.dtype)
        self.Vr = xp.zeros((1, self.p), dtype=self.dtype)
        self.Vm = xp.zeros((1, self.p), dtype=self.dtype)
        self.Ur_p = xp.zeros((self.p, self.Zr_p.shape[2]), dtype=self.dtype)
        self.Um_p = xp.zeros((self.p, self.Zm_p.shape[2]), dtype=self.dtype)
        self.W = xp.zeros((self.p, self.p), dtype=self.dtype)
        self.W_c = xp.zeros((self.p, self.p), dtype=self.dtype)
        self.Eta = xp.full((1, self.p), float(eta_init), dtype=self.dtype)
        self.Id = xp.eye(self.p, dtype=self.dtype)

        keys = ['Ur_g', 'Um_g', 'Vr', 'Vm', 'Ur_p', 'Um_p', 'W', 'W_c', 'Eta']
        self.opt_m = {k: 0 for k in keys}
        self.opt_v = {k: 0 for k in keys}

    # ------------------------------------------------------------------- fit

    def fit(self, pen1=1e-4, pen2=0.0, pen1_c=0.0, pen2_c=1e-2,
            regularizer='expm', s=1.0, epochs=1e4, max_iter=100, lr=1e-4,
            lr_coef_multiplier=1.0, lr_eta_multiplier=1.0, rho_init=1.0,
            checkpoint=2000, optim_param=(0.9, 0.99), rho_max=1e16,
            rho_multiplier=10.0, h_tol=1e-2, loss_tol=1e-6,
            verbose=False, progress_bar=False) -> None:
        assert regularizer in ('expm', 'logdet', 'nodag')
        self.pen1, self.pen2 = pen1, pen2
        self.pen1_c, self.pen2_c = pen1_c, pen2_c
        self.rho_init, self.rho_multiplier = rho_init, rho_multiplier
        self.lr_coef_multiplier = lr_coef_multiplier
        self.lr_eta_multiplier = lr_eta_multiplier
        self.optim_param, self.checkpoint = optim_param, int(checkpoint)
        self.progress_bar = progress_bar
        if progress_bar:
            from tqdm.auto import tqdm
            self.pbar = tqdm(total=int(max_iter))
        self.run(max_iter=max_iter, epochs=epochs, lr=lr, rho_max=rho_max,
                 h_tol=h_tol, loss_tol=loss_tol, s=s, reg=regularizer)
        if progress_bar:
            self.pbar.close()

    def result(self, w_threshold: float = 0.0) -> dict:
        xp = self.xp
        W = self.W.copy()
        W[xp.abs(W) < w_threshold] = 0.0
        out = {'W': self._tonumpy(W),
               'W_c': self._tonumpy(self.W_c),
               'Ur_g': self._tonumpy(self.Ur_g),
               'Um_g': self._tonumpy(self.Um_g),
               'Vr': self._tonumpy(self.Vr),
               'Vm': self._tonumpy(self.Vm),
               'Ur_p': self._tonumpy(self.Ur_p),
               'Um_p': self._tonumpy(self.Um_p),
               'Eta': self._tonumpy(self.Eta).ravel()}
        out['xi'] = 1.0 / (1.0 + np.exp(-np.clip(out['Eta'], -self.eta_clip,
                                                 self.eta_clip)))
        # Manuscript parameterization: the mask is D^e = 1 - D_j * e_j with e_j the
        # perturbation EFFICACY (e=1 hard, e=0 soft). Internally the free parameter is
        # xi = residual parent influence = 1 - e, so report both and avoid sign errors
        # downstream. These are the same quantity; 'e' is what the paper calls e_j.
        out['e'] = 1.0 - out['xi']
        return out
