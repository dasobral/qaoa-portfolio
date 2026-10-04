# Benchmarks

Methodology and curated results for the QAOA Portfolio Optimizer benchmark
suites. Raw JSON artifacts live under `results/benchmarks/` (gitignored);
every number below can be regenerated with the listed command, except the
n > 20 points of §7.4, which were measured outside the harness (see there).

Two campaigns are reported: the **June 2026** Phase 5 baseline (§2–6,
RTX 3080 workstation, CPU `default.qubit` simulator) and the **October 2026**
re-measurement on the same workstation and an NVIDIA DGX Spark with faster
simulator backends (§7). The `front/` dashboard displays both datasets.

- **Module:** `qaoa_portfolio/benchmarks.py`
- **CLI:** `qaoa-portfolio benchmark --suite {quality,scaling,layers,market}`
- **Scope:** at most 20 assets — a harness limit (`MAX_EXACT_ASSETS = 20`):
  the Rust brute force that supplies the reference optimum accepts n ≤ 20.

## 1. Methodology

### 1.1 Problem instances

Each run generates a synthetic price matrix with `generate_synthetic_prices`
(seeded NumPy generator: per-asset drift/volatility ladder plus a shared
market factor, so the covariance matrix is non-trivial). The Rust core
builds the QUBO (`build_qubo`) with risk factor 0.5 and a cardinality
target of half the assets. Repeats are **paired**: every solver sees the
identical instance for repeat *i* (seed = base seed + *i*).

### 1.2 Solvers

| Solver | Implementation | Notes |
|--------|----------------|-------|
| `brute_force` | Rust, exhaustive | Defines the per-instance optimum (n ≤ 20) |
| `simulated_annealing` | Rust, seeded | Default schedule |
| `markowitz` | Rust continuous + top-k | Selects the `target_assets` largest weights, then evaluates on the QUBO |
| `random` | Python, seeded | Uniform cardinality-constrained sample — the floor any optimizer must beat |
| `qaoa` | PennyLane statevector | Benchmark default: 1 layer, Adam, ≤60 iterations, 2 restarts |

### 1.3 Quality metric: approximation ratio

Raw QUBO objectives are scale- and sign-dependent, so quality is reported
as an approximation ratio in (0, 1]:

```
ratio = 1                          if achieved == optimum (±1e-9)
ratio = 1 / (1 + gap)              otherwise, with
gap   = (achieved − optimum) / max(|optimum|, 1e-9)
```

1.0 means the solver found the brute-force optimum; the ratio decays with
the relative optimality gap and stays defined for negative objectives.
This is *not* the textbook `optimum / achieved` ratio (which is undefined
across sign changes); comparisons between solvers are unaffected because
the mapping is strictly monotone in the gap.

### 1.4 Statistics

`significance_test` runs a **paired Wilcoxon signed-rank test** on
approximation ratios over identical instances. With fewer than ~10 repeats
the p-values are indicative only. Timings are wall-clock per solver call
(`time.perf_counter`); Python-side peak memory comes from `tracemalloc`
(QAOA and random baselines only — Rust-internal allocations are invisible
to it; pure-Rust numbers come from `cargo bench`).

### 1.5 Reproducibility

Every record stores its seed, run index, and solver settings; artifacts
echo the full `BenchmarkConfig`. Within a campaign, every host and backend
solves the identical seeded instances, and QAOA results are bit-for-bit
identical across simulator backends (§7.2), so quality numbers pool across
hosts while timings stay hardware-specific.

## 2. Results — Solution Quality (5A)

Command:

```bash
uv run qaoa-portfolio benchmark --suite quality --assets 8 --target 4 --repeats 10 --seed 42
```

8 assets, select 4, 10 paired instances, QAOA at 1 layer / Adam / ≤60
iterations / 2 restarts:

| Solver | Mean ratio | Std | Optimal runs | Median time |
|--------|-----------:|----:|-------------:|------------:|
| Brute force | 1.000 | 0.000 | 10/10 | < 1 ms |
| Simulated annealing | 1.000 | 0.000 | 10/10 | 0.7 ms |
| Markowitz (top-k) | 0.886 | 0.276 | 6/10 | 0.1 ms |
| **QAOA** | **0.825** | **0.211** | **3/10** | **23.2 s** |
| Random | 0.558 | 0.206 | 0/10 | 0.1 ms |

Paired Wilcoxon signed-rank tests on the 10 shared instances:

| Comparison | p-value | Verdict |
|------------|--------:|---------|
| QAOA vs random | 0.002 | QAOA significantly better (+48 % relative ratio) |
| QAOA vs Markowitz | 0.297 | No significant difference |
| QAOA vs simulated annealing | 0.016 | SA significantly better at this size |

The roadmap targets are met: QAOA improves on random selection by far more
than the 15–25 % goal, and is statistically indistinguishable from the
classical Markowitz top-k baseline. Simulated annealing remains the
strongest heuristic at these sizes — the expected outcome recorded in the
roadmap risk log ("document the crossover point").

## 3. Results — Scaling (5B)

Command (current CLI; reproduces the June settings):

```bash
uv run qaoa-portfolio benchmark --suite scaling --asset-counts 4,8,12,16,20 --repeats 3 --seed 42 \
  --qaoa-layers 3 --qaoa-iterations 10 --qaoa-restarts 1
```

QAOA ran at **p = 3 layers**, 10 iterations, 1 restart so the full ladder fits
a practical budget (a *cost* study, not a converged-quality study; n = 20 ran
with 2 repeats, everything else with 3). The June artifacts record p = 3: at
the time, passing any `--qaoa-*` override replaced the benchmark preset with
the full QAOA defaults (layers = 3). The CLI now merges overrides into the
preset, hence the explicit `--qaoa-layers 3` above.
Median wall-clock per solve:

| n | Brute force | Sim. annealing | Markowitz | QAOA (10 iter) | QAOA peak Python mem |
|--:|------------:|---------------:|----------:|---------------:|---------------------:|
| 4 | < 0.1 ms | 0.5 ms | < 0.1 ms | 1.4 s | 1 MB |
| 8 | < 0.1 ms | 0.7 ms | 0.1 ms | 4.4 s | 3.4 MB |
| 12 | 0.4 ms | 1.0 ms | 0.1 ms | 10.4 s | 25 MB |
| 16 | 1.3 ms | 1.5 ms | 0.1 ms | 25.9 s | 676 MB |
| 20 | 17.6 ms | 1.9 ms | 0.2 ms | 396 s | 16.2 GB |

Two scaling regimes are visible:

- **Rust classical solvers** stay in milliseconds across the whole ladder
  (brute force doubles per asset as expected — 2^n enumeration — but the
  constant is tiny).
- **QAOA statevector simulation** grows ~2.4× per +4 assets in time and
  ~16× per +4 assets in memory beyond n = 12; at n = 20 a single solve
  needs ~16 GB. The October campaign identified the cause: `default.qubit`
  differentiates by backpropagation and stores intermediate states.
  Re-running it at n = 20, p = 3 reproduces the peak (16,207 MB), while the
  `lightning.*` backends (adjoint differentiation) need ~17 MB of Python
  memory at the same size (§7.3). The 16 GB was a backend cost, not a limit
  of the problem.

QAOA quality at fixed 10 iterations also decays with n (1.00 → 0.69 →
0.53 → 0.36 → 0.64 mean ratio): larger instances need more optimizer
iterations to converge, compounding the time scaling.

## 4. Results — QAOA Depth (5B)

Command:

```bash
uv run qaoa-portfolio benchmark --suite layers --assets 6 --target 3 --repeats 3 --seed 42
```

6 assets, select 3, 3 repeats, 30 iterations / 1 restart per depth. Every
depth reaches the optimum on these small instances (mean ratio 1.000), so
the depth study measures *cost*, which is linear in p as QAOA theory
predicts:

| Depth p | 1 | 2 | 3 | 5 | 10 |
|---------|--:|--:|--:|--:|---:|
| Median time | 3.5 s | 5.8 s | 8.0 s | 12.4 s | 23.9 s |

Extra depth buys nothing at n = 6 — one layer already solves these
instances. Depth becomes interesting only on instances QAOA cannot solve
at p = 1, which (per the scaling table) are also the instances where each
additional layer is expensive. For this problem family, shallow circuits
with more restarts are the better trade.

## 5. Results — Real Market Data (5C)

Out-of-sample protocol: optimize on the first 70 % of the window, score
the equal-weighted selection on the held-out 30 % with `FinancialMetrics`
(annualized return/volatility, Sharpe, max drawdown).

Command (example):

```bash
uv run qaoa-portfolio benchmark --suite market \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,JNJ,XOM \
  --start-date 2022-01-01 --end-date 2024-12-31 --assets 8 --target 4
```

All three studies use 2022-01-01 → 2024-12-31, 70/30 split, QAOA at the
benchmark default (1 layer, Adam, ≤60 iterations, 2 restarts), seed 42.

**S&P 500 subset** (8 large caps, select 4; 527 in-sample / 226 out-of-sample days):

| Solver | Selection | QUBO ratio | OOS return | OOS vol | Sharpe | Max DD |
|--------|-----------|-----------:|-----------:|--------:|-------:|-------:|
| Brute force / SA | NVDA, JPM, JNJ, XOM | 1.000 | +39.6 % | 16.0 % | 2.35 | −9.2 % |
| Markowitz | MSFT, NVDA, JPM, XOM | 0.611 | +42.8 % | 18.8 % | 2.17 | −11.3 % |
| QAOA | MSFT, NVDA, JNJ, XOM | 0.792 | +27.8 % | 16.0 % | 1.62 | −8.3 % |
| Random | AAPL, MSFT, GOOGL, JPM | 0.168 | +32.8 % | 16.1 % | 1.91 | −12.1 % |

**Crypto** (6 coins, select 3; 767 / 329 days):

| Solver | Selection | QUBO ratio | OOS return | OOS vol | Sharpe | Max DD |
|--------|-----------|-----------:|-----------:|--------:|-------:|-------:|
| Brute force / SA / **QAOA** | BTC, ETH, BNB | 1.000 | +83.4 % | 44.6 % | 1.83 | −32.3 % |
| Markowitz | ETH, BNB, ADA | 0.729 | +81.2 % | 49.9 % | 1.59 | −42.8 % |
| Random | BTC, ADA, SOL | 0.589 | +95.8 % | 53.5 % | 1.75 | −37.9 % |

**Mixed stocks + crypto** (6 assets, select 3; 527 / 226 days):

| Solver | Selection | QUBO ratio | OOS return | OOS vol | Sharpe | Max DD |
|--------|-----------|-----------:|-----------:|--------:|-------:|-------:|
| Brute force / SA / **QAOA** | MSFT, JPM, XOM | 1.000 | +19.9 % | 13.8 % | 1.30 | −7.8 % |
| Markowitz | MSFT, XOM, BTC | 0.259 | +45.0 % | 22.5 % | 1.92 | −11.4 % |
| Random | AAPL, BTC, ETH | 0.072 | +89.4 % | 42.2 % | 2.07 | −19.9 % |

QAOA found the exact QUBO optimum on both the crypto and the mixed study.
Note the deliberate distinction the tables expose: the QUBO ratio measures
*how well the solver optimized the formulated problem*; the out-of-sample
columns measure *how that selection fared afterwards*. The risk-averse
QUBO optimum (low volatility, low drawdown) is not the highest-return
portfolio out of sample — in the mixed study the random crypto-heavy pick
earned more while carrying 3× the volatility and 2.5× the drawdown.
Solver quality and portfolio-model quality are different questions, and
only the first is QAOA's job.

## 6. Interpretation

1. **Roadmap success metrics:** QAOA vs random: +48 % relative quality
   (target 15–25 %) with p ≈ 0.002 — met. QAOA vs classical: statistically
   indistinguishable from Markowitz top-k (p ≈ 0.30) — met; SA remains
   significantly better at simulator-reachable sizes — documented
   honestly. 8-asset end-to-end optimization: Rust path < 1 ms,
   QAOA ≈ 23 s (quantum simulation dominates).
2. **The crossover narrative:** at every size the classical Rust solvers
   are faster and at least as good. QAOA's value here is demonstrating a
   correct, end-to-end quantum formulation pipeline — QUBO → Hamiltonian →
   variational optimization → ranked portfolios — not beating classical
   solvers on classical hardware, which simulation cannot do.
3. **The 20-asset ceiling:** in June, n = 20 cost 396 s and 16 GB per QAOA
   solve versus 18 ms for exact brute force. The October campaign (§7)
   showed that limit was the `default.qubit` backend plus a harness guard
   (`MAX_EXACT_ASSETS = 20`), not the hardware: on `lightning.gpu`, n = 20
   takes 13 s and exact simulation reaches n = 26 (RTX 3080) and n = 28
   (DGX Spark).
4. **Real-data pipeline works end-to-end** on historical windows,
   including the mixed asset-class case (timezone alignment between
   equity and crypto bars was fixed during this phase — see
   `_combine_portfolio_data`).

## 7. October 2026 campaign — RTX 3080 and DGX Spark (GB10)

### 7.1 Setup

| Host | CPU | GPU / memory | QAOA backends |
|---|---|---|---|
| RTX 3080 workstation (same as §2–6) | x86_64, 24 threads, 31 GB RAM | RTX 3080, 10 GB GDDR6X (~760 GB/s) | `default.qubit`, `lightning.qubit`, `lightning.gpu` |
| NVIDIA DGX Spark | Grace, aarch64, 20 cores | GB10, 128 GB unified LPDDR5x (~273 GB/s) | `default.qubit`, `lightning.qubit`, `lightning.gpu` |

Same seeds (42), instance generator, and QUBO (risk 0.5, k = n/2) as §1;
the backend is selected with `--qaoa-backend` (install `lightning.gpu` with
`uv sync --extra dev --extra gpu`). Every run is logged with wall time, peak
memory, and whether another benchmark shared the host. Both campaigns ran
CPU-only tracks alongside the GPU preset runs at n ≤ 24, so the preset
timings in §7.5–7.6 are slightly pessimistic; the cost ladder (§7.3), the
n > 20 cost probes (§7.4), and the n = 26 preset solves ran alone. Example:

```bash
uv run qaoa-portfolio benchmark --suite quality --assets 12 --repeats 10 --seed 42 \
  --qaoa-backend lightning.gpu
```

### 7.2 Backends change speed, never results

At n = 8 (preset: p = 1, Adam, 60 iterations, 2 restarts) every backend on
both hosts returns the June result exactly — QAOA 0.825 mean ratio, 3/10
optimal, same selections — and the three market studies of §5 reproduce
exactly. Median QAOA time per solve:

| Backend | RTX 3080 | DGX Spark |
|---|---:|---:|
| `default.qubit` | 24.6 s (June: 23.2 s) | 16.1 s |
| `lightning.qubit` (CPU) | 7.7 s | 5.5 s |
| `lightning.gpu` | 8.3 s | 7.0 s |

At this size both GPUs are overhead-bound (Python / PennyLane dispatch on a
tiny statevector) and the CPU backends are as fast or faster.

### 7.3 Cost scaling, n = 4–20 (p = 3, 10 iterations, 1 restart — pairs with §3)

Median seconds per QAOA solve (3 repeats; June n = 20 used 2):

| n | June `default.qubit` | RTX `lightning.qubit` | RTX `lightning.gpu` | GB10 `lightning.qubit` | GB10 `lightning.gpu` |
|--:|---:|---:|---:|---:|---:|
| 4 | 1.36 | 0.62 | 0.65 | 0.45 | 0.56 |
| 8 | 4.38 | 1.75 | 1.83 | 1.21 | 1.40 |
| 12 | 10.4 | 3.50 | 3.75 | 2.48 | 2.68 |
| 16 | 25.9 | 7.72 | 6.30 | 5.45 | 4.53 |
| 20 | 396 | 78.6 | **12.7** | 45.4 | **13.6** |

- n = 20 is **31× faster** than June on the RTX 3080 GPU (29× on the GB10).
  On the Spark, the June figure splits into ~3× from hardware
  (`default.qubit` 130 s) and ~9.5× from the backend.
- Peak Python memory at n = 20 drops from 16.2 GB to ~17 MB with the
  `lightning.*` backends (adjoint differentiation instead of backprop, §3).
- GPUs overtake the CPU backends around n = 16.

### 7.4 Beyond 20 assets (measured outside the harness)

The harness refuses n > 20 (`MAX_EXACT_ASSETS`, and the Rust brute force
accepts n ≤ 20). The points below were produced by campaign scripts that
rebuild the identical seeded instances, call the same QAOA solver, and
compute the exact optimum by enumerating all 2^n bitstrings in NumPy
(validated equal to the Rust brute force at n = 12–20, and equal across the
two hosts to 1e-13). They are not regenerable with the CLI at this commit;
bringing n > 20 into the harness is planned.

Seconds per solve, same cost settings as §7.3 (1 instance, seed 42):

| n | RTX `lightning.gpu` | GB10 `lightning.gpu` | RTX `lightning.qubit` | GB10 `lightning.qubit` |
|--:|---:|---:|---:|---:|
| 22 | 26 | 52 | — | 191 |
| 24 | 83 | 215 | 1 917 (shared host) | 844 |
| 26 | 339 | 949 | — | 3 806 |
| 27 | out of memory | — | — | — |
| 28 | — | 4 270 | — | — |

- **Ceilings.** RTX 3080: n = 26 on the GPU (n = 27 fails with CUDA out of
  memory in the adjoint Jacobian), n = 24 on the CPU. DGX Spark: n = 28 on the
  GPU (71 min per 10-iteration solve), n = 26 on the CPU (63 min, 47 GB RSS).
  A 1-iteration memory probe ran at n = 29; n = 30 was projected at ~5 h per
  solve and ~100 GB and not attempted.
- **Memory.** `lightning.gpu` holds about five statevector copies: ~5.4 GB of
  VRAM at n = 26 on the RTX; on the GB10, 3 / 8 / 24 / 41 GB above baseline at
  n = 24 / 26 / 28 / 29. `lightning.qubit` needs ~6× more at the same n.
- **Hardware.** From n = 22 up, the RTX 3080 GPU is ~2.8× faster per solve than
  the GB10 — the ratio of their memory bandwidths. Exact statevector
  simulation streams the whole state for every gate (one cost layer is
  n(n+1)/2 rotations), so it is bandwidth-bound; the GB10's advantage is
  capacity (n = 28–29 vs 26). A prototype that applies the cost layer as one
  diagonal pass ran 17–28× faster per circuit evaluation than the current
  gate-by-gate path at n = 20–26 on the RTX 3080 (same energies to 1e-12);
  `qml.DiagonalQubitUnitary` is not a usable shortcut on `lightning.gpu`,
  which expands it to a dense 2^n × 2^n matrix.

### 7.5 Quality vs size (preset, pooled across hosts)

Mean approximation ratio (instances solved to the exact optimum):

| n | Instances | QAOA | Simulated annealing | Markowitz top-k | Random |
|--:|--:|---|---|---|---|
| 12 | 10 | 0.849 (7) | 0.945 (2) | 0.973 (5) | 0.532 (0) |
| 16 | 10 | 0.722 (4) | 0.870 (0) | 0.995 (8) | 0.579 (0) |
| 20 | 10 | 0.720 (5) | 0.845 (0) | 0.889 (1) | 0.611 (0) |
| 22 † | 5 | 0.630 (2) | 0.828 (0) | 0.806 (0) | 0.608 (0) |
| 24 † | 5 | 0.769 (3) | 0.827 (0) | 0.951 (0 of 3) | 0.464 (0 of 3) |
| 26 † | 3 | 0.453 (1) | 0.740 (0) | 1.000 (1 of 1) | 0.533 (0 of 1) |

† outside the harness (§7.4). QAOA time per solve on `lightning.gpu`
(RTX / GB10): n = 12 16 / 13 s, n = 20 57 / 66 s, n = 24 428 / 1 123 s,
n = 26 1 734 / 4 741 s.

How to read this table:

- **QAOA is all-or-nothing.** It either finds the exact optimum or returns a
  poor (often cardinality-infeasible) portfolio, so it hits the optimum more
  often than simulated annealing while its *mean* ratio is lower at every n.
- **The simulated-annealing baseline is untuned.** The harness calls the Rust
  solver with its defaults: one run of 10 000 single-bit-flip moves with a
  fixed geometric schedule, independent of n, finishing in ~3 ms — against
  minutes to over an hour for QAOA. Single flips always leave the
  cardinality-feasible set, so the budget penalty is a barrier for this move
  set. "QAOA reaches the optimum more often than SA" is therefore a statement
  about this default configuration only; a tuned, time-matched SA comparison
  has not been run yet.
- **Samples at n ≥ 22 are small** (1–5 instances); no significance test is
  meaningful there.
- Markowitz top-k is the strongest classical heuristic at n = 16–20.

### 7.6 Optimizer budget, optimizer choice, and depth

- **More Adam budget does not help** (n = 12, 10 instances): 60 / 150 / 300
  iterations with 2 restarts → 0.849 / 0.671 / 0.832; 5 restarts →
  0.642 / 0.601 / 0.736. Restarts are selected by expected energy while the
  answer is the best of the 64 most probable states; the two criteria
  disagree. At n = 20, 150 iterations raise hits from 5/10 to 6/10 for 2.5×
  the cost.
- **COBYLA beats Adam** at equal iteration cap (p = 1, 2 restarts):

  | n | COBYLA ratio (optimal) | time / solve | Adam ratio (optimal) | time / solve |
  |--:|---|---:|---|---:|
  | 12 | 0.908 (8/10) | 5.8 s (CPU) | 0.849 (7/10) | 14.5 s (CPU) |
  | 16 | 0.745 (6/10) | 12.6 s (CPU) | 0.722 (4/10) | 27.6 s (GPU) |
  | 20 | 0.785 (6/10) | 16 s (GPU) | 0.720 (5/10) | 57 s (GPU) |
  | 24 † | 0.702 (2/5) | 58 s (GPU) | 0.769 (3/5) | 428 s (GPU) |
  | 26 † | 0.775 (2/3) | 147 s (GPU) | 0.453 (1/3) | 1 734 s (GPU) |

  At n = 24 the two optimizers solve different instances (together 5/5).
- **Depth** (n = 10, 10 instances, Adam): p = 1 / 2 / 3 / 5 → 0.603 (2) /
  0.885 (8) / 0.697 (4) / 0.861 (7) — p ≥ 2 helps but not monotonically
  (5 instances at p = 8 / 10: 0.764 / 0.872). At n = 12 and 16 depth makes
  little difference. Cost is linear in p (~10.5 s per layer at n = 10 on CPU).

### 7.7 Summary

1. The June 20-asset limit was a backend and harness limit, not a hardware
   one: the same workstation now runs n = 20 in 13 s and reaches n = 26.
2. Per problem, the RTX 3080 is the faster simulator (bandwidth); the DGX
   Spark is the larger one (capacity, n = 28–29).
3. QAOA reaches the exact optimum at n = 24–26, where untuned simulated
   annealing does not — a promising but not yet fair comparison (§7.5).
4. COBYLA is the better optimizer for this problem: better or equal hit rates
   at every size except n = 24, at 2.5–12× less time on the same backend, and
   no gradient memory.
5. The current circuit construction, not the GPUs, dominates run time;
   expressing the cost layer as a single pass is the largest available
   speed-up.
