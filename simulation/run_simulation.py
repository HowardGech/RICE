import numpy as np
import os
import sys
sys.path.append("../source/")
from utils import *
from scRICE_CF import *
import cupy as cp
from utils_gen import *
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
from sklearn.linear_model import LinearRegression as LR
import ruptures as rpt


results = pd.DataFrame(columns=['measure', 'value'])

args = sys.argv[1:]
n, p, s0, graph_type, intervention, job_id = int(args[0]), int(args[1]), int(args[2]), args[3], args[4], int(args[5])
set_random_seed(job_id)
npz_file = np.load(f'data/n{n//1000}k_p{p}_s{s0}_{graph_type}_job{job_id}.npz', allow_pickle=True)

X = npz_file['X']
W_true = npz_file['W']
mask = npz_file['mask']
B_true = (W_true != 0).astype(int)

predictor = np.log(1 + X)
res = np.zeros_like(X)
control_idx = np.where(mask.sum(axis=1) == 0)[0]

# Reduced control function
for i in range(p):
    lr = LR().fit(mask[:,i].reshape(-1,1), predictor[:, i])
    res[:, i] = predictor[:, i] - lr.predict(mask[:,i].reshape(-1,1))
    
    
import threading
import time
from pynvml import *

def _extract_pid_mem(proc):
    """
    Returns (pid, used_bytes) or (None, None) if it can't extract.
    Handles older pynvml variants where proc may be:
      - an object with .pid and .usedGpuMemory
      - a tuple/list (pid, usedGpuMemory)
      - a dict-like {'pid':..., 'usedGpuMemory':...}
    """
    # Object with attributes
    pid = getattr(proc, "pid", None)
    used = getattr(proc, "usedGpuMemory", None)

    if pid is not None and used is not None:
        return pid, used

    # Tuple/list
    if isinstance(proc, (tuple, list)) and len(proc) >= 2:
        return proc[0], proc[1]

    # Dict-like
    if isinstance(proc, dict):
        if "pid" in proc and ("usedGpuMemory" in proc or "used_gpu_memory" in proc):
            return proc["pid"], proc.get("usedGpuMemory", proc.get("used_gpu_memory"))

    return None, None


def start_nvml_pid_monitor(interval_s=0.1, print_live=True):
    """
    Monitors GPU memory used by *this Python process only* (PID-based).
    Correct for MIG and multi-tenant GPUs.
    Returns (stop_fn, stats_dict).
    """
    pid_self = os.getpid()
    stop_event = threading.Event()
    stats = {"peak_mb": 0.0}

    def worker():
        nvmlInit()
        try:
            peak = 0.0

            while not stop_event.is_set():
                used_mb = 0.0

                for i in range(nvmlDeviceGetCount()):
                    h = nvmlDeviceGetHandleByIndex(i)

                    # Some setups report under compute, some under graphics, some under both.
                    proc_lists = []
                    for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                        f = globals().get(fn)
                        if f is None:
                            continue
                        try:
                            proc_lists.append(f(h))
                        except NVMLError:
                            pass

                    for procs in proc_lists:
                        for p in procs:
                            ppid, used_bytes = _extract_pid_mem(p)
                            if ppid != pid_self:
                                continue
                            if used_bytes is None or used_bytes <= 0:
                                continue
                            used_mb += used_bytes / (1024 ** 2)

                peak = max(peak, used_mb)
                stats["peak_mb"] = peak

                if print_live:
                    print(
                        f"\rPID {pid_self} GPU mem: {used_mb:8.1f} MB | peak: {peak:8.1f} MB",
                        end=""
                    )

                time.sleep(interval_s)

            if print_live:
                print()

        finally:
            nvmlShutdown()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def stop():
        stop_event.set()
        t.join()

    return stop, stats

stop_monitor, stats = start_nvml_pid_monitor(interval_s=1, print_live=False)

try:
    start_time = time.time()
    model = scRICE_CF(dtype=cp.float64)

    model.prep(X=X, predictor=predictor, intervention_effect=mask, intervention_type=intervention, add_intercept=True, confounding_res=res)

    model.fit(regularizer='expm', binsum_k=X.shape[1], pen1=1e-4, pen2=0, pen1_c=0, pen2_c=1e-2, lr=0.0001, max_iter=100, checkpoint=2000, h_tol=1e-2, loss_tol=1e-6)
    end_time = time.time()
    run_time = end_time - start_time
finally:
    stop_monitor()

W_raw = model.result(w_threshold=0.0)['W']

edge_weights = sorted(np.abs(W_raw).flatten(), reverse=True)
x = np.arange(len(edge_weights))
y = edge_weights

signal = np.array(edge_weights).reshape(-1, 1)

# Detect change points. You may need to adjust the model and parameters based on your specific data and requirements.
algo = rpt.BottomUp(model='l1').fit(signal)
result = algo.predict(n_bkps=1)

print(f"Change point detected at index: {result}")
thre = edge_weights[result[0]-1]

W_est = W_raw.copy()
W_est[np.abs(W_raw) < thre] = 0
W_est = to_dag(W_est)
acc = measurement(W_true, W_est)

acc['auc'] = roc_auc_score((W_true!=0).astype(int).flatten(), np.abs(W_raw).flatten())
acc['time'] = run_time
acc['peak_gpu_mem_mb'] = stats['peak_mb']

for m in acc.keys():
    results = pd.concat([results, pd.DataFrame({'measure': m, 'value': acc[m]},index=[0])], ignore_index=True)

results.to_csv(f'data/n{n//1000}k_p{p}_s{s0}_{graph_type}_job{job_id}_result.csv', index=False)