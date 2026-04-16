import numpy as np
import os
import sys
sys.path.append('../source/')
from utils import *
from utils_gen import *
from tqdm.auto import tqdm
import time


args = sys.argv[1:]
n, p, s0, graph_type, intervention, job_id = int(args[0]), int(args[1]), int(args[2]), args[3], args[4], int(args[5])
hard_intervention = False if intervention == 'soft' else True

set_random_seed(job_id)


B_true = simulate_dag(p, s0, graph_type)
W_true = simulate_parameter(B_true, w_ranges = ((-1, -.5), (.5, 1)))


# The following code generates negative binomial data. Modify 'simulate_NB' function in utils_gen.py to change the data generation process.
theta_m =  -1*np.ones(p)
theta_r =  1*np.ones(p)
X, mask, Z, Um, Ur, Vm, Vr = simulate_NB(W=W_true, theta_m=theta_m, d=0, intercept=True, theta_r=theta_r,  n=n, g=lambda x: np.log(1+x), shift=2,hard_intervention=hard_intervention,coef=1, covar=1)
np.savez(f'data/n{n//1000}k_p{p}_s{s0}_{graph_type}_{intervention}_job{job_id}.npz', X=X, W=W_true, mask=mask, Z=Z, Um=Um, Ur=Ur, Vm=Vm, Vr=Vr)

