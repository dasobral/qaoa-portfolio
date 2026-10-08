# Algorithm

How the QAOA Portfolio Optimizer turns a price history into a portfolio
selection: Markowitz mean-variance model → QUBO → cost Hamiltonian → QAOA
circuit and variational loop → decoded portfolio, and how the result is
scored. Every formula below is the one the code evaluates; the table in
[§6](#6-notation-and-code-symbols) maps each symbol to its code name.

See also: [usage guide](usage_guide.md) (how to run each step),
[API reference](api_reference.md), [Rust core](rust_core.md) (QUBO
construction and classical solvers), [quantum backend](quantum_backend.md)
(QAOA implementation), [benchmarks](benchmarks.md) (measured results).

## 1. Portfolio selection as optimization

### 1.1 Returns and risk from prices

The input is a price matrix $P \in \mathbb{R}_{>0}^{(T+1) \times n}$: one row
per trading day, one column per asset. The Rust core (`ReturnSeries`) uses
log returns and annualizes with 252 trading days:

$$
r_{t,i} = \ln \frac{P_{t,i}}{P_{t-1,i}}, \qquad
\mu_i = 252 \cdot \frac{1}{T}\sum_{t=1}^{T} r_{t,i}, \qquad
\Sigma_{ij} = 252 \cdot \frac{1}{T-1}\sum_{t=1}^{T} (r_{t,i}-\bar r_i)(r_{t,j}-\bar r_j).
$$

$\mu$ is the annualized expected return vector and $\Sigma$ the annualized
sample covariance matrix. Reported metrics (Sharpe ratio, out-of-sample
returns, charts) use a different convention: simple returns with compound
annualization in `FinancialMetrics`. The two layers are never mixed in one
table; see [rust_core.md § Return and Annualization Conventions](rust_core.md#return-and-annualization-conventions).

### 1.2 Binary mean-variance selection

The optimizer chooses *which* assets to hold, not how much of each. Each
asset $i$ gets a binary variable $x_i \in \{0, 1\}$ (1 = selected), and a
valid portfolio selects exactly $k$ assets (`target_assets`), implicitly
equally weighted. A risk-aversion weight $q \in [0, 1]$ (`risk_aversion` in
the bridge, `risk_factor` in the benchmark harness) trades risk against
return. The objective the code builds is

$$
f(x) = q\left(\sum_{i} \Sigma_{ii}\,x_i + \sum_{i<j} \Sigma_{ij}\,x_i x_j\right) - (1-q)\sum_i \mu_i\,x_i .
$$

Note the cross terms: each pair $i<j$ contributes $q\,\Sigma_{ij}$ once. The
textbook mean-variance risk $q\,x^\top \Sigma x$ would contribute
$2q\,\Sigma_{ij}$ per pair, because $x^\top\Sigma x = \sum_i \Sigma_{ii}x_i + 2\sum_{i<j}\Sigma_{ij}x_ix_j$
for binary $x$. Equivalently,
$f(x) = \tfrac{q}{2}\left(x^\top\Sigma x + \operatorname{diag}(\Sigma)^\top x\right) - (1-q)\,\mu^\top x$:
the encoded risk term weights covariances between assets half as much as the
textbook form. All solvers (QAOA and the classical baselines) optimize this
same $f$, so solver comparisons are unaffected; the portfolio model is what
differs from the textbook.

### 1.3 Continuous Markowitz baseline

For comparison, the Rust `MarkowitzSolver` (bridge `solve_markowitz`) solves
the classical continuous problem: maximum-Sharpe weights

$$
w \propto \Sigma^{-1}(\mu - r_f \mathbf 1), \qquad \textstyle\sum_i w_i = 1,
$$

with $r_f = 0.02$ and a pseudo-inverse when $\Sigma$ is singular (minimum
variance $w \propto \Sigma^{-1}\mathbf 1$ when the weights cannot be
normalized). The benchmark's `markowitz` solver keeps the $k$ largest
weights and evaluates that selection on the QUBO.

## 2. QUBO encoding

### 2.1 Energy convention

A QUBO here is a symmetric matrix $Q \in \mathbb{R}^{n\times n}$ plus a
constant offset $c$, with energy

$$
E(x) = c + \sum_{i} Q_{ii}\,x_i + \sum_{i<j} Q_{ij}\,x_i x_j .
$$

$Q$ is stored symmetrically ($Q_{ij} = Q_{ji}$) but each off-diagonal pair is
counted **once**, from the upper triangle. In matrix form
$E(x) = x^\top U x + c$ with $U = \operatorname{triu}(Q)$ — not
$x^\top Q x + c$, which would count every pair twice. This is the convention
of `PyQUBOMatrix.evaluate`, `evaluate_qubo_bitstring`,
`benchmarks.exact_enumeration` and the cost Hamiltonian below.

### 2.2 Cardinality constraint as a penalty

"Select exactly $k$ assets" becomes the quadratic penalty
$\lambda\left(\sum_i x_i - k\right)^2$. With $x_i^2 = x_i$ it expands to

$$
\lambda\left(\sum_i x_i - k\right)^2
= \lambda(1-2k)\sum_i x_i + 2\lambda\sum_{i<j}x_ix_j + \lambda k^2 ,
$$

so `PenaltyBuilder::budget` adds $\lambda(1-2k)$ to every diagonal entry,
$2\lambda$ to every off-diagonal pair, and $\lambda k^2$ to the offset. The
complete QUBO built by `build_qubo(prices, symbols, risk_aversion, target_assets)` is

$$
Q_{ii} = q\,\Sigma_{ii} - (1-q)\,\mu_i + \lambda(1-2k), \qquad
Q_{ij} = q\,\Sigma_{ij} + 2\lambda \;\; (i \ne j), \qquad
c = \lambda k^2 ,
$$

and $E(x) = f(x) + \lambda\left(\sum_i x_i - k\right)^2$.

### 2.3 Penalty calibration

`QUBOFormulation::build_from_params` sets the penalty weight from the
largest objective coefficient:

$$
\lambda = 2 \max\Big(\max_i \big|q\,\Sigma_{ii} - (1-q)\,\mu_i\big|,\; \max_{i<j} \big|q\,\Sigma_{ij}\big|\Big),
$$

falling back to $\lambda = 1$ when every objective coefficient is zero. The
Rust API accepts an explicit value (`with_budget_penalty`, finite and
positive); the Python bridge always uses the automatic one. The factor 2 is
a scale heuristic, not a proven bound: it makes a single violated unit
($|\sum x_i - k| = 1$) cost at least twice any single objective coefficient,
but a state that drops or adds an asset can still gain more than $\lambda$
when many coefficients add up. The exact optimum of $E$ is therefore not
guaranteed to select $k$ assets, and heuristic answers (QAOA in particular)
can be infeasible — which is why every benchmark record reports feasibility
separately ([§4.2](#42-feasibility-and-probability-on-the-optimum)).

The Rust core also offers `position_limit` and `diversity` penalty terms;
they are not exposed through the Python bridge.

Inputs are validated before building: $q \in [0, 1]$ and finite,
$1 \le k \le n$, finite $\mu$ and $\Sigma$, at least two price rows with
strictly positive prices.

## 3. QAOA

### 3.1 Cost Hamiltonian

Each asset becomes one qubit. Substituting $x_i = (1 - Z_i)/2$ (so
$|0\rangle \mapsto x_i = 0$ and $|1\rangle \mapsto x_i = 1$) into $E(x)$ gives
the Ising Hamiltonian built by `build_cost_hamiltonian`:

$$
H_C = c' \, I + \sum_i h_i\, Z_i + \sum_{i<j} J_{ij}\, Z_i Z_j ,
$$

$$
c' = c + \sum_i \frac{Q_{ii}}{2} + \sum_{i<j}\frac{Q_{ij}}{4}, \qquad
h_i = -\frac{Q_{ii}}{2} - \sum_{j \ne i}\frac{Q_{ij}}{4}, \qquad
J_{ij} = \frac{Q_{ij}}{4}.
$$

Terms with a zero coefficient are dropped. $H_C$ is diagonal in the
computational basis with $H_C\,|x\rangle = E(x)\,|x\rangle$, so its ground
state is the QUBO minimum. Bitstring character $i$ is wire $i$ is asset $i$;
wire 0 is the most significant bit of the basis-state index (PennyLane's
ordering).

### 3.2 Mixer and ansatz

The mixer is the standard transverse field $H_M = \sum_i X_i$
(`build_mixer_hamiltonian`). With $p$ layers (`QAOAConfig.layers`) and angles
$\boldsymbol\gamma = (\gamma_1,\dots,\gamma_p)$,
$\boldsymbol\beta = (\beta_1,\dots,\beta_p)$, the circuit prepares

$$
|\psi(\boldsymbol\gamma,\boldsymbol\beta)\rangle =
\prod_{l=1}^{p} e^{-i\beta_l H_M}\, e^{-i\gamma_l H_C}\; |+\rangle^{\otimes n},
$$

starting from Hadamards on every wire and applying
`qml.qaoa.cost_layer(γ_l, H_C)` then `qml.qaoa.mixer_layer(β_l, H_M)` per
layer. The parameter vector is $(\gamma_1,\dots,\gamma_p,\beta_1,\dots,\beta_p)$.

### 3.3 Variational loop

The classical optimizer minimizes the expected energy

$$
F(\boldsymbol\gamma,\boldsymbol\beta) = \langle\psi(\boldsymbol\gamma,\boldsymbol\beta)|\,H_C\,|\psi(\boldsymbol\gamma,\boldsymbol\beta)\rangle
$$

on the configured PennyLane device (`default.qubit`, `lightning.qubit`, or
`lightning.gpu`; with `shots=None` the value is exact).

- **Initialization and restarts.** `num_restarts` independent runs start from
  $\gamma_l \sim U(0, \pi)$ and $\beta_l \sim U(0, \pi/2)$, drawn from one
  `numpy.random.default_rng(seed)` stream (so restart $r$ is reproducible from
  the seed). The run with the lowest observed $F$ wins; its best parameters
  are used for decoding.
- **Gradient optimizers** (`adam`, `gradient_descent`): PennyLane's
  `AdamOptimizer()` / `GradientDescentOptimizer()` with default step size
  0.01. Each step records the best $F$ seen so far; a run stops after
  `max_iterations` steps or after 5 consecutive steps whose change in $F$ is
  below `convergence_threshold`. Gradients use the device's default
  differentiation method (backpropagation on `default.qubit`, adjoint on the
  `lightning.*` devices).
- **Gradient-free optimizers** (`cobyla`, `nelder_mead`):
  `scipy.optimize.minimize` with `options={"maxiter": max_iterations}` (for
  COBYLA SciPy counts function evaluations). The best evaluated point is
  kept; the history records the best $F$ after each SciPy iteration callback.
- The benchmark harness preset is `layers=1`, `optimizer="cobyla"`,
  `max_iterations=60`, `num_restarts=2`; library code defaults to
  `QAOAConfig()` (`layers=3`, `adam`, 100 iterations, 3 restarts).

`QAOAResult.convergence_history` is the winning run's best-so-far $F$ per
iteration, `iterations` its length, and `metadata["expected_cost"]` the
winning $F$.

### 3.4 Decoding

After optimization the circuit is evaluated once more to obtain the full
distribution $\Pr(x) = |\langle x|\psi\rangle|^2$ (clipped at 0 and
renormalized). From it the backend records the probability mass per Hamming
weight (`metadata["hamming_weight_distribution"]`) and, when
`target_assets` $=k$ is set, the feasible mass
$\sum_{|x| = k}\Pr(x)$ (`metadata["feasible_probability"]`).

The answer is **not** simply the most probable state:

1. Keep the $m$ = `max_stored_solutions` (default 64) most probable basis
   states (`QAOAResult.probabilities`).
2. Evaluate $E(x)$ for each and sort by $(E(x), \text{bitstring})$
   ascending.
3. The first entry is the answer (`best_bitstring`, `selected_assets`,
   `objective_value`); the first 10 entries are `top_solutions`.

**Feasible decoding** (`QAOAConfig(feasible_decoding=True, target_assets=k)`,
CLI `--qaoa-feasible-decoding`) restricts step 1 to states of Hamming weight
$k$: the answer is the lowest-energy state among the $m$ most probable
*feasible* states, so it always selects exactly $k$ assets when $k \le n$.
It is off by default so earlier results stay reproducible.

## 4. Measuring quality

The benchmark harness ([benchmarks.md](benchmarks.md)) scores every solver
against the exact optimum $E^\ast = \min_x E(x)$ of the same QUBO: the Rust
brute force for $n \le 20$ and a chunked NumPy enumeration of all $2^n$
bitstrings for $20 < n \le 28$ (`exact_optimum`; each record names the method
in `metadata.optimum_reference`).

### 4.1 Approximation ratio

QUBO energies can be negative or cross zero, so the textbook ratio
$E^\ast/E$ is not usable. `approximation_ratio(achieved, optimum)` maps the
relative optimality gap onto $(0, 1]$:

$$
\rho =
\begin{cases}
1 & \text{if } |E - E^\ast| \le 10^{-9},\\[4pt]
\dfrac{1}{1 + g}, \quad g = \dfrac{E - E^\ast}{\max(|E^\ast|,\,10^{-9})} & \text{otherwise.}
\end{cases}
$$

$\rho = 1$ means the optimum was found; $\rho$ is strictly decreasing in the
gap, so solver rankings do not depend on the mapping. An achieved value
below the optimum by more than $10^{-9}$ raises `BenchmarkError`. A run
**hits the optimum** when $\rho \ge 1 - 10^{-9}$; `optimal_hit_rate` is the
share of such runs.

### 4.2 Feasibility and probability on the optimum

- **Feasible:** a record is feasible when its selection has exactly $k$
  assets (`metadata["feasible"]`); `summarize_quality` reports the
  `feasibility_rate`.
- **$p_\text{opt}$:** for QAOA, the final probability of the exact optimum
  bitstring $x^\ast$ returned by `exact_optimum`,
  $p_\text{opt} = |\langle x^\ast|\psi\rangle|^2$ (`metadata["p_opt"]`,
  summarized as `mean_p_opt`). It measures how concentrated the circuit is on
  the answer, independently of the top-$m$ decoding rule. When several
  bitstrings share the optimal energy only the returned one is counted.
- **Feasible mass:** `feasible_probability` as defined in §3.4, summarized
  as `mean_feasible_probability`.

### 4.3 Paired statistics

Every solver sees the identical instance for each repeat (instance seed =
base seed + run index), so comparisons are paired.

**Wilcoxon signed-rank test** (`significance_test`). Records of two solvers
are paired by `(num_assets, seed)` and SciPy's two-sided `wilcoxon` is
applied to the paired approximation ratios. It answers "is one solver's
ratio systematically higher?" and uses the size of the differences. When all
paired differences are within $10^{-9}$ the test is skipped and $p = 1$ is
reported with `identical=True`; with fewer than about 10 pairs the p-value is
indicative only.

**Exact McNemar test** (`mcnemar_test`). Records are paired by
`(num_assets, seed)` and reduced to hit/miss outcomes. Only discordant pairs
carry information: $b$ instances where only solver A hit the optimum and $c$
where only solver B did. Under the null hypothesis each discordant pair is a
fair coin, so the two-sided p-value is

$$
p = \min\left(1,\; 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i}\,2^{-(b+c)}\right),
$$

with $p = 1$ when $b + c = 0$. It answers "does one solver hit the optimum
more often?", the right question for QAOA, whose answers tend to be either
exactly optimal or far off.

## 5. Limits

- **Exact statevector simulation.** The simulators store all $2^n$ complex
  amplitudes (16 bytes each in double precision: 4 GiB per state at
  $n = 28$, and the optimizers hold several copies). Time and memory double
  per asset; measured costs per backend are in
  [benchmarks.md §7](benchmarks.md#7-october-2026-campaign--rtx-3080-and-dgx-spark-gb10).
  The cost layer is applied gate by gate ($n$ single-qubit and
  $n(n-1)/2$ two-qubit rotations per layer).
- **Harness cap $n \le 28$** (`MAX_EXACT_ASSETS`): the largest size measured
  with exact simulation. The Rust brute force stops at $n = 20$
  (`MAX_RUST_BRUTE_FORCE_ASSETS`); `exact_enumeration` is used above that and
  refuses $n > 40$.
- **Shots.** `QAOAConfig.shots` forwards a finite shot count to the device,
  which then *estimates* $F$ and the probabilities from samples, but the
  simulator still builds the full $2^n$ state and the harness always runs
  with `shots=None`. A sampling-based pipeline that scales beyond exact
  statevector simulation is not implemented.
- **Model scope.** Selections are equal-weight subsets of exactly $k$ assets;
  the QUBO carries no fractional weights, transaction costs, or sector
  constraints. The continuous Markowitz solver is a baseline only.
- **Penalty, not constraint.** The cardinality constraint is soft (§2.3);
  the X-mixer explores all $2^n$ states, so QAOA can return infeasible
  selections unless feasible decoding is enabled.

## 6. Notation and code symbols

| Symbol | Meaning | Code |
|---|---|---|
| $n$ | number of assets / qubits | `len(symbols)`, `PyQUBOMatrix.num_variables`, `metadata["num_variables"]` |
| $P$ | price matrix (periods × assets) | `prices` argument of `build_qubo` |
| $\mu$ | annualized mean log returns | `PyReturnSeries.mean_returns()` |
| $\Sigma$ | annualized covariance of log returns | `PyReturnSeries.covariance_matrix()` |
| $q$ | risk-aversion weight in $[0,1]$ | `risk_aversion` (`build_qubo`), `BenchmarkConfig.risk_factor`, CLI `--risk-factor` |
| $k$ | assets to select | `target_assets` (`build_qubo`, `BenchmarkConfig`, `QAOAConfig`), CLI `--target` |
| $x$ | binary selection vector | `PyOptimizationResult.solution`, `QAOAResult.best_solution`, bitstring characters |
| $\lambda$ | budget penalty weight | computed in `QUBOFormulation::build_from_params`; `with_budget_penalty` (Rust only) |
| $Q$ | symmetric QUBO matrix | `PyQUBOMatrix.to_numpy()` |
| $c$ | QUBO offset ($=\lambda k^2$) | `PyQUBOMatrix.offset`, `metadata["offset"]` |
| $E(x)$ | QUBO energy | `PyQUBOMatrix.evaluate`, `evaluate_qubo_bitstring`, `objective_value` |
| $E^\ast$ | exact optimum | `exact_optimum(qubo)[0]`, `solve_brute_force(qubo).objective_value` |
| $H_C$, $H_M$ | cost and mixer Hamiltonians | `build_cost_hamiltonian`, `build_mixer_hamiltonian` |
| $p$ | QAOA depth | `QAOAConfig.layers`, CLI `--qaoa-layers` |
| $\boldsymbol\gamma, \boldsymbol\beta$ | QAOA angles | `QAOAResult.optimal_parameters["gammas"]`, `["betas"]` |
| $F$ | expected energy minimized by the optimizer | `convergence_history`, `metadata["expected_cost"]` |
| $\Pr(x)$ | final state probabilities | `QAOAResult.probabilities` (top $m$) |
| $m$ | decoding candidates kept | `QAOAConfig.max_stored_solutions` |
| $\rho$ | approximation ratio | `approximation_ratio`, `BenchmarkRecord.approximation_ratio` |
| $p_\text{opt}$ | probability of the optimum bitstring | `metadata["p_opt"]`, `mean_p_opt` |
| $b, c$ (McNemar) | discordant hit counts | `only_a`, `only_b` in `mcnemar_test` output |

## 7. Checking the mapping yourself

The cost Hamiltonian is diagonal with the QUBO energy on every basis state.
This snippet builds a 4-asset QUBO, checks $H_C|x\rangle = E(x)|x\rangle$ for
all 16 states against both evaluators, and evaluates the approximation ratio
on two examples (runs in about a second):

```python
import numpy as np
import pennylane as qml
import qaoa_portfolio_core as core
from qaoa_portfolio import (
    approximation_ratio,
    build_cost_hamiltonian,
    evaluate_qubo_bitstring,
    generate_synthetic_prices,
)

prices = generate_synthetic_prices(num_assets=4, periods=120, seed=7)
qubo = core.build_qubo(prices, ["A", "B", "C", "D"], 0.5, 2)
Q, c = qubo.to_numpy(), qubo.offset

H = build_cost_hamiltonian(Q, c)
energies = np.real(np.diag(qml.matrix(H, wire_order=range(4))))
for index in range(16):
    bits = format(index, "04b")
    assert np.isclose(energies[index], evaluate_qubo_bitstring(Q, bits, c))
    assert np.isclose(energies[index], qubo.evaluate([b == "1" for b in bits]))

print("ground state:", format(int(energies.argmin()), "04b"))
print(approximation_ratio(-0.9, -1.0))  # gap 0.1 -> 0.909...
print(approximation_ratio(1.2, 1.0))    # gap 0.2 -> 0.833...
```

Output:

```text
ground state: 1010
0.9090909090909091
0.8333333333333334
```
