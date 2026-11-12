"""Generate the latent-confounding sweep (d = 0, 2, ..., 10) used for the
confounding figure. One replicate per invocation; the replicate index seeds the graph.
"""
import numpy as np
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from utils import *
from sim_base import *
import sys
from tqdm.auto import tqdm


import time

job_id = int(os.environ['LSB_JOBINDEX'])
args = sys.argv[1:]
model = str(args[0])
um = int(args[1])
intervention = str(args[2])
hard_intervention = False if intervention == 'soft' else True
# Output tag. The original driver hardcoded '_hm' for EVERY generator, so switching the
# commented/uncommented generator line silently overwrote the previous run's files.
tag = str(args[3]) if len(args) > 3 else 'nb'
# The PLN/Pois branches below refer to `cova`, which the original file never assigns --
# they would raise NameError. It is the same latent-factor count as `um`.
cova = um

set_random_seed(job_id)


n, p, s0, graph_type = 10000,100,400, 'ER'
B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges = ((-1, -.5), (.5, 1.0)))


if model == 'NB' or model == 'GNB':
    theta_m =  -1*np.ones(p)
    theta_r =  1*np.ones(p)
    X, mask, Z, Um, Ur, Vm, Vr = simulate_NB_tgt(W=W_true, theta_m=theta_m, d=um, intercept=True, theta_r=theta_r,  n=n, g=lambda x: np.log(1+x), shift=2,hard_intervention=hard_intervention,coef=1, covar=1)
    # X, mask, Z, Um, Ur, Vm, Vr, _ = simulate_NB_structured(W=W_true, theta_m=theta_m, d=um, intercept=True, theta_r=theta_r,  n=n, g=lambda x: np.log(1+x), shift=2,hard_intervention=hard_intervention,coef=1, covar=1)
    # X, mask, Z, Um, Ur, Vm, Vr, leak_targets = simulate_NB_hidden_mediator(W=W_true, theta_m=theta_m, d=um, intercept=True, theta_r=theta_r,  n=n, g=lambda x: np.log(1+x), shift=2,hard_intervention=hard_intervention,coef=1, covar=1)

    np.savez(f'result/simu_{intervention}/data/{model}/n{n//1000}k_p{p}_s{s0}_{graph_type}_um{um}_{tag}_repl{job_id}.npz', X=X, W=W_true, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr)

elif model == 'PLN':
    theta =  -1*np.ones(p)
    X, mask, Z, U, V = simulate_PLN_tgt(W=W_true, theta=theta, n=n, intercept=True, g=lambda x: np.log(1+x), shift=2, noise_scale=0.5, d=cova, hard_intervention=hard_intervention,coef=1, covar=1)
    np.savez(f'result/simu_{intervention}/data/{model}/n{n//1000}k_p{p}_s{s0}_{graph_type}_cova{cova}_repl{job_id}.npz', X=X, W=W_true, mask=mask, Z=Z, U=U, V=V)
elif model == 'Pois':
    theta =  -1*np.ones(p)
    X, mask, Z, U, V = simulate_Pois_tgt(W=W_true, theta=theta, n=n, intercept=True, g=lambda x: np.log(1+x), shift=2, d=cova, hard_intervention=hard_intervention,coef=1, covar=1)
    np.savez(f'result/simu_{intervention}/data/{model}/n{n//1000}k_p{p}_s{s0}_{graph_type}_cova{cova}_repl{job_id}.npz', X=X, W=W_true, mask=mask, Z=Z, U=U, V=V)
else:
    # raise error
    raise ValueError('Unknown model type')


