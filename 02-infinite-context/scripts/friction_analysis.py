#!/usr/bin/env python3
"""
Friction-decay analysis of contextarena.ai MRCRv2 8-needle leaderboard export.
Self-contained reproduction script — handoff artifact (Fabian Franz / Claude, 2026-06-10).

Claims this script reproduces:
  C1. Pure exponential pass(L) = exp(-lambda*L) fits the top models with
      near-constant per-8k-block survival (GPT-5.5: k=0.98, R^2=0.996).
  C2. Weibull score = exp(-(lambda*L)^k) fitted per config: shape k <= 1 for
      every monotone curve in the dataset (no length-accelerated hazard).
  C3. The only k > 1.15 fits are Claude Opus 4.7 configs whose curves are
      NON-MONOTONE (collapse to ~1-2% at 128k-256k, recovery to ~9% at 512k)
      -- a regime event, not a hazard process; excluded from C2's family.
  C4. k < 1 (the majority) is the classical signature of heterogeneous
      constant-rate subpopulations (Proschan 1963: mixtures of exponentials
      have decreasing failure rate) -- i.e., per-row constant friction with
      row-dependent rates, which is what a multi-surface failure model predicts.

Method notes:
  - Fits are direct nonlinear least squares in SCORE space (grid + refinement).
    Do NOT use the log-log linearization ln(-ln s) ~ k ln x: saturated scores
    (100%) become unbounded leverage points and bias k upward (this produced a
    spurious k=1.98 for GPT-5.5 before correction).
  - Metric is contextarena avg score (not pass@0.90); qualitative agreement
    with the paper's pass@0.90 ladder was verified separately on DeepSeek V4
    Flash (per-8k-block survival 0.92-0.97 across 8k-256k raw; map protocol
    0.993; combined 0.997; out-of-sample check: predicted 128k map pass
    0.993^16 = 89.4% vs observed 89.7%).

Usage: python3 friction_analysis.py <leaderboard.csv>
"""
import csv, math, sys

BUCKETS = [("8k",1),("16k",2),("32k",4),("64k",8),("128k",16),("256k",32),("512k",64),("1M",128)]

def load(path):
    rows = list(csv.DictReader(open(path)))
    configs = []
    for i, r in enumerate(rows):
        pts = []
        for name, blk in BUCKETS:
            v = r.get(f"{name}_pct")
            if v:
                s = float(v) / 100.0
                if s > 0.01:
                    pts.append((blk, s))
        if len(pts) >= 5:
            configs.append({"slug": r["model_slug"], "auc": float(r["auc_128k_pct"] or 0),
                            "row": i, "pts": pts})
    return configs

def sse(pts, lam, k):
    return sum((s - math.exp(-((lam*x)**k)))**2 for x, s in pts)

def fit_weibull(pts):
    """Grid + iterative refinement, least squares in score space."""
    best = (1e9, None, None)
    for ki in range(10, 260, 5):
        k = ki / 100
        for li in range(-70, 1):
            lam = math.exp(li / 10)
            v = sse(pts, lam, k)
            if v < best[0]:
                best = (v, lam, k)
    _, l0, k0 = best
    for _ in range(4):
        for dk in [x/1000 for x in range(-40, 41, 4)]:
            for dl in [x/100 for x in range(-25, 26, 5)]:
                lam = l0 * math.exp(dl); k = max(0.05, k0 + dk)
                v = sse(pts, lam, k)
                if v < best[0]:
                    best = (v, lam, k)
        _, l0, k0 = best
    v, lam, k = best
    obs = [s for _, s in pts]
    m = sum(obs) / len(obs)
    r2 = 1 - v / sum((o - m)**2 for o in obs)
    return lam, k, r2

def monotone_nonincreasing(pts, tol=0.02):
    return all(pts[i+1][1] <= pts[i][1] + tol for i in range(len(pts)-1))

def main(path):
    configs = load(path)
    print(f"{len(configs)} configs with >=5 buckets\n")
    fits = []
    for c in configs:
        lam, k, r2 = fit_weibull(c["pts"])
        fits.append({**c, "lam": lam, "k": k, "r2": r2,
                     "mono": monotone_nonincreasing(c["pts"])})

    # C2/C3: shape distribution
    ks = sorted(f["k"] for f in fits)
    n = len(fits)
    print(f"Shape k: median={ks[n//2]:.2f}, 90th pct={ks[9*n//10]:.2f}")
    accel = [f for f in fits if f["k"] > 1.15]
    print(f"k>1.15: {len(accel)}/{n}")
    for f in accel:
        sc = " ".join(f"{x*8}k:{s*100:.0f}" for x, s in f["pts"])
        print(f"  {f['slug']} (row {f['row']})  k={f['k']:.2f}  monotone={f['mono']}  [{sc}]")
    clean_accel = [f for f in accel if f["mono"]]
    print(f"k>1.15 AND monotone (genuine length-acceleration candidates): {len(clean_accel)}/{n}")

    # Best config per family table
    best = {}
    for f in fits:
        if f["slug"] not in best or f["auc"] > best[f["slug"]]["auc"]:
            best[f["slug"]] = f
    print(f"\n{'model (best-AUC config)':40s} {'lam/8k':>7s} {'k':>5s} {'R2':>6s} {'mono':>5s}")
    print("-" * 70)
    for slug in sorted(best, key=lambda s: best[s]["lam"]):
        f = best[slug]
        print(f"{slug:40s} {f['lam']:7.4f} {f['k']:5.2f} {f['r2']:6.3f} {str(f['mono']):>5s}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mrcr-leaderboard-8-needle-2026-06-10.csv")
