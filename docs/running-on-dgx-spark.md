# Running QOPO on the NVIDIA DGX Spark (GB10)

Run the QAOA side of the benchmark on the Spark's GPU — using NVIDIA's
ready-to-run CUDA stack for GB10 — and compare it against the classical
baselines (brute force, simulated annealing, Markowitz) to answer the open
question: does QAOA actually gain anything over standard optimization
algorithms on this hardware?

## 1. Does QOPO need custom CUDA kernels? No.

This is the key fact that shapes the whole setup: **the project does not
compile or ship any CUDA code.**

- The Rust core (`qaoa_portfolio_core`: QUBO formulation, brute force,
  simulated annealing, Markowitz) is pure CPU (nalgebra / ndarray / rayon /
  PyO3). It simply has to be *rebuilt for aarch64* on the Spark.
- The QAOA side is written against PennyLane's device abstraction
  (`qml.device(...)`). All GPU kernels live inside a **prebuilt third-party
  simulator plugin**, not in this repository (no `.cu`/`.cuh` files, no
  CUDA build flags anywhere).

So this is not a "port the kernels to GB10" task — it is a
"run the code as-is in the right environment" task. The only GB10
compatibility question is whether the *prebuilt* GPU stack you pick
(PyTorch for LightningGPU, cuQuantum wheels) was built to run on
**sm_121** (GB10's Blackwell compute capability). NVIDIA's ready-to-run
resources exist precisely to remove that burden: the DGXOS host image ships
CUDA 12.8 plus a GB10-tuned PyTorch, and NVIDIA's prebuilt wheels
(LightningGPU aarch64 wheels, cuQuantum `-cu12` wheels) already embed
Blackwell cubins (verified: cuStateVec 1.15 ships `sm_100` + `sm_120`
targets, and sm_120 SASS is binary-compatible with sm_121 within the 12.x
major version).

## 2. Why the Spark needs a specific runtime

The Spark differs from the RTX 3080 machine in two ways that change how the
software must be built and run:

| | RTX 3080 (baseline runs) | DGX Spark (this guide) |
|---|---|---|
| CPU arch | x86_64 | **aarch64** (ARM) — no reusable envs or wheels |
| GPU | Ampere, compute capability 8.6 | **GB10 Grace Blackwell, sm_121**, 128 GB *unified* CPU+GPU LPDDR5x |
| CUDA stack | system CUDA 12.0 | **CUDA 12.8 preinstalled** on DGXOS, plus a GB10-tuned PyTorch |
| QAOA device | `default.qubit` / `lightning.qubit` (CPU) | **`lightning.gpu`** (PennyLane LightningGPU, CUDA) |

Key consequences:

- `qaoa-env/`, `target/` and any prebuilt wheels from the 3080 box are
  x86_64-only — build the environment from scratch on the Spark
  (`uv.lock` and `data/cache/` are portable).
- The GB10 has **one shared 128 GB memory pool** for CPU and GPU. There is
  no separate GPU HBM: everything (Python state, CUDA context, PyTorch
  allocator, *and the locally deployed coding agent*) competes for the same
  pool. See §7 before running anything big.
- The repo's backend allowlist (`QAOAParams.SUPPORTED_BACKENDS` in
  `qaoa_portfolio/params.py`) currently only accepts `default.qubit` and
  `lightning.qubit`, and the `qaoa-portfolio benchmark` CLI has no
  `--backend` flag yet — so the GPU run is driven by the small script in §6
  (it reuses the Phase 5 harness, keeping results directly comparable with
  the 3080 artifacts in `docs/benchmarks.md`).

## 3. Pre-flight checks on the Spark (5 min)

```bash
nvidia-smi        # GB10 (Grace Blackwell) + driver
nvcc --version    # CUDA 12.8 on DGXOS
uname -m          # aarch64
python3 - <<'EOF'
import torch
print(torch.__version__, torch.version.cuda)
print("sm:", torch.cuda.get_device_capability(0))   # expect (12, 1)
EOF
```

DGXOS ships a PyTorch build tuned for GB10. If `torch` is missing, install
an **aarch64 CUDA build that includes sm_121 (Blackwell consumer) kernels**
before anything else — LightningGPU fails at runtime without it.

## 4. Fresh aarch64 environment

```bash
git clone https://github.com/dasobral/qaoa-portfolio.git
cd qaoa-portfolio
git checkout feature/runtime-spark

# Rust toolchain (aarch64) + uv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
python3 -m pip install uv

# Project env (repo convention: qaoa-env/, not .venv/)
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev        # builds qaoa_portfolio_core (maturin) for aarch64
uv run qaoa-portfolio --help
uv run pytest -q           # optional CPU sanity pass

# The GPU plugin — pin the same minor as the installed PennyLane (0.45.x)
uv pip install "pennylane-lightning-gpu==0.45.0"
```

Device smoke test (finishes in seconds):

```bash
uv run python - <<'EOF'
import pennylane as qml
dev = qml.device("lightning.gpu", wires=4)
print(dev)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    return qml.probs(wires=range(4))

print(circuit())          # [0.25 x8] means the CUDA path is alive
EOF
```

## 5. Enable `lightning.gpu` in QOPO

**Option A — one-line allowlist change** (recommended; it is the natural
first commit of this branch):

```diff
--- a/qaoa_portfolio/params.py
+++ b/qaoa_portfolio/params.py
-    SUPPORTED_BACKENDS = ("default.qubit", "lightning.qubit")
+    SUPPORTED_BACKENDS = ("default.qubit", "lightning.qubit", "lightning.gpu")
```

**Option B — no repo change:** the driver script below patches the
allowlist in memory before the backend module is first imported.

## 6. Run the example: QAOA on the GB10 vs classical baselines

Save as `spark_example.py` in the repo root, then:

```bash
uv run python spark_example.py
```

```python
"""DGX Spark example: QAOA on the GPU vs classical baselines.

Reproduces the Phase 5 quality suite (docs/benchmarks.md section 2) with
the QAOA solver on the lightning.gpu device, so results line up with the
RTX 3080 baseline artifacts.
"""
from __future__ import annotations

# Option B from docs/running-on-dgx-spark.md section 5: enable the GPU
# device without a repo change. Must run before qaoa_portfolio.quantum_backend
# (or .benchmarks, which imports it) is first imported.
import qaoa_portfolio.params as _params

if "lightning.gpu" not in _params.QAOAParams.SUPPORTED_BACKENDS:
    _params.QAOAParams.SUPPORTED_BACKENDS = (
        _params.QAOAParams.SUPPORTED_BACKENDS + ("lightning.gpu",)
    )

import pennylane as qml

from qaoa_portfolio.benchmarks import (
    BenchmarkConfig,
    run_quality_benchmark,
    save_benchmark_results,
    significance_test,
    summarize_quality,
)
from qaoa_portfolio.quantum_backend import QAOAConfig

BACKEND = "lightning.gpu"

# 1) Device smoke test — fail fast if CUDA is broken.
dev = qml.device(BACKEND, wires=4)
print(f"[spark] device ok: {dev}")
try:
    import torch

    print(f"[spark] sm: {torch.cuda.get_device_capability(0)}")
except Exception as exc:  # torch is optional for the benchmark itself
    print(f"[spark] torch not introspectable: {exc}")

# 2) Same instance settings and QAOA preset as the 3080 quality run.
qaoa = QAOAConfig(backend=BACKEND, layers=1, optimizer="adam",
                  max_iterations=60, num_restarts=2)
config = BenchmarkConfig(num_assets=8, target_assets=4, repeats=10,
                         seed=42, qaoa=qaoa)
solvers = ("brute_force", "simulated_annealing", "markowitz", "random", "qaoa")

# 3) Paired quality benchmark: every solver sees the identical seeded instances.
records = run_quality_benchmark(config, solvers=solvers)
summary = summarize_quality(records)

print(f"{'solver':<20}{'mean ratio':>12}{'optimal':>10}{'mean time':>14}")
for name, stats in summary.items():
    optimal = int(round(stats["optimal_hit_rate"] * stats["runs"]))
    print(f"{name:<20}{stats['mean_approximation_ratio']:>12.3f}"
          f"{optimal}/{stats['runs']:<6}{stats['mean_elapsed_ms'] / 1000:>11.2f} s")

for baseline in ("simulated_annealing", "markowitz"):
    test = significance_test(records, "qaoa", baseline)
    print(f"[spark] qaoa vs {baseline}: p={test['p_value']:.3f} "
          f"({test['num_pairs']} pairs)")

qaoa_rec = max((r for r in records if r.solver_name == "qaoa"),
               key=lambda r: r.approximation_ratio)
print(f"[spark] best QAOA selection: {qaoa_rec.selected_assets}")

path = save_benchmark_results(
    records,
    suite="spark-quality",
    config=config,
    extra={"summary": summary, "device": BACKEND, "host": "DGX Spark GB10"},
)
print(f"[spark] artifact: {path}")
```

Reference points from the 3080 CPU run (8 assets / 4 selected / 10 repeats,
`docs/benchmarks.md`): QAOA 23.2 s median, mean ratio 0.825; simulated
annealing 0.7 ms, ratio 1.000; Markowitz top-k 0.1 ms, ratio 0.886. The
experiment is meaningful if `lightning.gpu` cuts the QAOA wall time while
keeping the quality ratio where it was — and whether that finally beats the
classical baselines in practice.

For the scaling picture, raise `num_assets` (12, 16, 20 — the exact-simulation
ceiling) and re-run; each is a fresh, seeded, paired comparison.

## 7. RAM budgeting (read before big runs)

- The GB10's 128 GB pool is **shared between CPU and GPU**. The locally
  deployed coding agent holds a large slice of it while active — **close or
  suspend agent sessions before running the experiment**, as the GPU needs
  as much of the pool as possible.
- Measured Python-side peaks on the 3080 (CPU simulator, `tracemalloc`):
  n = 8 → 3.4 MB, n = 16 → 676 MB, **n = 20 → 16.2 GB**. Add the CUDA
  context + PyTorch allocator (a few GB) when on `lightning.gpu`.
- Run one solve at a time; no parallel `pytest` or second benchmark.
- Monitor with `free -g` (the unified pool) and `nvidia-smi` (SM
  utilization; memory counters are shared on GB10 and may read N/A).
- If OOM at n = 20: lower `repeats`/`max_iterations`, or step down to
  n = 16 first.

## 8. Alternative runtime stacks (if the host setup is not enough)

Two other ways to get the "NVIDIA-prepared for GB10" environment, listed
because they were explicitly considered for this experiment. Neither is
required to run the example — the host setup in §4 already is — but they
are the fallbacks / upgrade paths.

### 8.1 cuQuantum (NVIDIA's quantum SDK) — maximum GPU optimization, real integration work

- `pip install cuquantum` is a meta package that pulls `cuquantum-cu12`
  (or `-cu13`), which installs the prebuilt NVIDIA libraries —
  cuStateVec, cuStabilizer, cuPauliProp, cuTensorNet, cuDensityMat — as
  **aarch64 wheels** (e.g. `custatevec-cu12`, ~73 MB prebuilt
  `libcustatevec.so`). Verified: the current wheels embed Blackwell cubins
  (`sm_100` + `sm_120`), which run on GB10's sm_121 via within-major-version
  binary compatibility.
- `pip install cuquantum-python` adds NVIDIA's official Python bindings
  (low-level: state-vector propagation, gates, expectation values).
- **But there is no drop-in PennyLane device built on cuQuantum** on PyPI —
  adopting it means rewriting the QAOA loop (`quantum_backend.py`: cost
  Hamiltonian evolutions, X-mixer, optimizer steps, probability extraction)
  against the cuStateVec API instead of `qml.qnode`. That is a separate
  engineering effort, not a config flag.
- **Verdict:** run the LightningGPU path first (§4–§6). If its throughput on
  GB10 is the bottleneck for the comparison, then open the cuQuantum
  integration as a follow-up phase of this branch.

### 8.2 NVIDIA ready-to-use containers on DGXOS — the reproducible path

DGXOS includes NGC access and Docker with the NVIDIA runtime. If you want
the whole stack to come from NVIDIA's GB10-validated containers instead of
the host packages (reproducible across machines, or the host toolchain is
incomplete):

```bash
# Pick the newest 12.8.x arm64 tag in the NGC catalog (verify the tag has
# an arm64 entry before pulling):
docker manifest inspect nvcr.io/nvidia/cuda:12.8-devel-ubuntu24.04

docker run --rm --gpus all -it \
  -v $PWD:/work -w /work \
  -e UV_PROJECT_ENVIRONMENT=qaoa-env \
  nvcr.io/nvidia/cuda:12.8-devel-ubuntu24.04 bash
```

Inside the container the workflow is unchanged: install rustup + uv,
`uv sync --extra dev`, `uv pip install pennylane-lightning-gpu` (plus a
GB10-capable aarch64 PyTorch — an NGC PyTorch container can substitute for
the CUDA base image if you prefer PyTorch to be baked in), then run
`spark_example.py`. The Rust extension and the PennyLane env build inside
the container's aarch64 toolchain exactly as in §4; nothing about the
project code changes.

Decision guide:

| Path | When | Effort |
|---|---|---|
| Host DGXOS + `lightning.gpu` (§4) | Default — run the experiment this week | ~30 min setup |
| NVIDIA container (§8.2) | Reproducibility / host stack incomplete | +30 min |
| cuQuantum rewrite (§8.1) | Only if LightningGPU's GB10 throughput disappoints | Separate project |

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `lightning.gpu` device not found / plugin import error | Plugin missing from `qaoa-env`, or PennyLane/plugin minor mismatch — both must be 0.45.x (`uv pip list \| grep -i pennylane`). |
| `QuantumBackendError: Unable to create PennyLane backend 'lightning.gpu'` | Allowlist — apply §5 option A (option B in the script already covers this). |
| torch kernel errors mentioning sm_121 / "no kernel image" | PyTorch build lacks Blackwell-consumer kernels — use the GB10 PyTorch that ships with DGXOS (or an aarch64 cu128 wheel built with sm_121). |
| CUDA version mismatch errors between torch and system toolkit | Align on 12.8 (DGXOS default); do not mix with the 3080's 12.0 assumptions. |
| cuQuantum wheels refuse to initialize on the GB10 | Check `nvidia-smi` driver vs the wheel's CUDA major version (use the `-cu12` wheels on CUDA 12.8); if sm_120 SASS does not JIT on your driver, wait for/force a cuQuantum build that lists sm_121 explicitly. |
| `lightning.gpu` does not work on GB10 at all | Fall back to `lightning.qubit` (CPU) so the classical comparison still runs, and document the gap; escalate to the §8.1 cuQuantum path. |

## 10. Recording results & next steps for this branch

- Keep seed 42, the 1-layer benchmark preset, and the solver list identical
  to the 3080 runs so hardware/backend is the only variable; the JSON
  artifact schema already matches (`results/benchmarks/spark-quality-*.json`
  vs `quality-*.json`).
- Suggested follow-up commits on `feature/runtime-spark`:
  1. One-line `params.py` allowlist change (§5 option A).
  2. `--backend` flag on `qaoa-portfolio benchmark` so the CLI can drive the
     GPU device directly.
  3. Add the measured Spark numbers (n = 8 / 12 / 16 / 20) to
     `docs/benchmarks.md` and the README headline table.
  4. (If needed) cuQuantum-based backend prototype, following §8.1.
