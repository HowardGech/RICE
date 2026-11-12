#!/usr/bin/env python
"""
Fix the count-rescale overflow in the negative-binomial simulators.

    python patch_count_rescale.py sim_base.py sim_crossguide.py

THE BUG
-------
Every NB generator rescales the log-mean `m` in two steps:

    if np.max(m) > CAP:   m -= (np.max(m) - CAP)      # 1. cap the largest log-mean
    if np.mean(m) < FLOOR: m += (FLOOR - np.mean(m))  # 2. lift the average log-mean

Step 2 can undo step 1. When one cell has an extreme log-mean, step 1 subtracts a large
constant, which drags the column's mean far below FLOOR; step 2 then adds that whole
distance back, leaving the maximum FAR above CAP. Worked example, 10,000 cells at
m = -3 with a single cell at m = 40, CAP = 6.5, FLOOR = 0.5:

    start                  ->  max 40.00, mean  -3.00
    step 1 subtracts 33.50 ->  max  6.50, mean -36.50
    step 2 adds     37.00  ->  max 43.50, mean   0.50
    exp(43.50) = 7.76e18   ->  NB mean larger than int64 max (9.22e18)

`nbinom.rvs` then returns garbage: on some builds it saturates near 9.223e18, on others
it wraps NEGATIVE. A negative count makes the next stage's `np.log(1 + X)` produce NaN
("RuntimeWarning: invalid value encountered in log"), and the first-stage regression dies
with "Input y contains NaN". It also cascades: a gene with astronomic counts enters its
children through g(X) = log(1+X) ~ 40, so their log-means explode too.

At p = 100 this is not rare. Measured on simulate_NB_tgt with no exclusion violation,
27-36 of 100 genes had maximum counts above 1e6 and up to 3 genes above 1e12.

THE FIX
-------
Clip the log-mean elementwise at CAP after both shift steps:

    m = np.minimum(m, CAP)

This is a no-op whenever the pathology does not trigger -- if the two shifts leave
max(m) <= CAP, nothing changes -- so well-behaved genes keep the published
data-generating process exactly. Only cells whose counts would otherwise be
astronomically large (and, on the cluster, negative) are affected. The mean floor is
still applied, but the cap now wins where the two conflict, because the cap is the
overflow guard.

CAVEAT, read before rerunning anything
--------------------------------------
Because the pathology triggers on most p = 100 replicates, patched runs will NOT
reproduce unpatched numbers gene-for-gene. Any figure mixing patched and unpatched
replicates is invalid. Regenerate a whole arm, not part of one.

Cap-only sites (`if np.max(l) > 7.5` in the Poisson and PLN paths) are left alone: with no
mean floor after them there is nothing to undo the cap, so they cannot overflow this way.
"""
import re
import shutil
import sys

CAP_RE = re.compile(r"^(\s*)if np\.max\(m\) > ([\d.]+):\s*$")
SUB_RE = re.compile(r"^\s*m -= \(np\.max\(m\)\s*-\s*([\d.]+)\)\s*$")
FLOOR_RE = re.compile(r"^(\s*)if np\.mean\(m\) < (-?[\d.]+):\s*$")
ADD_RE = re.compile(r"^\s*m \+= \((-?[\d.]+) - np\.mean\(m\)\)\s*$")


def patch_text(lines):
    """Insert an elementwise clip after each cap+floor rescale pair."""
    out, i, sites = [], 0, []
    while i < len(lines):
        cap_m = CAP_RE.match(lines[i])
        # a rescale site is exactly: cap test, cap shift, floor test, floor shift
        if (cap_m and i + 3 < len(lines) and SUB_RE.match(lines[i + 1])
                and FLOOR_RE.match(lines[i + 2]) and ADD_RE.match(lines[i + 3])):
            indent, cap = cap_m.group(1), cap_m.group(2)
            out.extend(lines[i:i + 4])
            # look past the inserted comment block, not just the next line or two,
            # or a second run double-patches
            already = any("np.minimum(m," in lines[j]
                          for j in range(i + 4, min(i + 12, len(lines))))
            if not already:
                out.append(f"{indent}# the mean floor above can undo the cap when one "
                           f"cell is extreme, leaving")
                out.append(f"{indent}# max(m) far above {cap} and exp(m) beyond int64, "
                           f"which makes nbinom.rvs")
                out.append(f"{indent}# return negative counts. Clip elementwise: a no-op "
                           f"unless that happens.")
                out.append(f"{indent}m = np.minimum(m, {cap})")
                sites.append((i + 1, cap, "patched"))
            else:
                sites.append((i + 1, cap, "already patched"))
            i += 4
            continue
        out.append(lines[i])
        i += 1
    return out, sites


def main(paths):
    if not paths:
        raise SystemExit(__doc__.strip().split("\n")[2])
    for path in paths:
        lines = open(path).read().split("\n")
        new, sites = patch_text(lines)
        if not sites:
            print(f"{path}: no cap+floor rescale sites found")
            continue
        changed = [s for s in sites if s[2] == "patched"]
        for line_no, cap, status in sites:
            print(f"{path}:{line_no}  cap={cap}  {status}")
        if changed:
            shutil.copy(path, path + ".bak")
            open(path, "w").write("\n".join(new))
            print(f"{path}: {len(changed)} site(s) patched, original saved to {path}.bak")
        else:
            print(f"{path}: nothing to do")


if __name__ == "__main__":
    main(sys.argv[1:])
