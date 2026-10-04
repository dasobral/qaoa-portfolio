# Running QOPO on the NVIDIA DGX Spark (GB10)

Run the QAOA side of the benchmark on the Spark's GPU — inside an NGC
container built on NVIDIA's ready-to-run stack for GB10 — and compare it
against the classical baselines (brute force, simulated annealing,
Markowitz) to answer the open question: does QAOA actually gain anything
over standard optimization algorithms on this hardware?

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
(the LightningGPU wheel's `-cu12` libraries, or the cuQuantum wheels) was
built to run on **sm_121** (GB10's Blackwell compute capability).
NVIDIA's ready-to-run resources exist precisely to remove that burden:
the LightningGPU aarch64 wheels pull their whole CUDA stack from PyPI as
prebuilt aarch64 `-cu12` wheels (`custatevec-cu12` plus
`nvidia-cublas/cusparse/cuda-runtime/nvjitlink-cu12`), and the GB10-tuned
PyTorch used for pre-flight checks and introspection comes from NVIDIA's
NGC PyTorch image (`nvcr.io/nvidia/pytorch:26.09-py3`), not from a host
install. Verified: cuStateVec 1.15 embeds `sm_100` + `sm_120` Blackwell
targets, and sm_120 SASS is binary-compatible with sm_121 within the 12.x
major version).

## 2. Why the Spark needs a specific runtime

The Spark differs from the RTX 3080 machine in two ways that change how the
software must be built and run:

| | RTX 3080 (baseline runs) | DGX Spark (this guide) |
|---|---|---|
| CPU arch | x86_64 | **aarch64** (ARM) — no reusable envs or wheels |
| GPU | Ampere, compute capability 8.6 | **GB10 Grace Blackwell, sm_121**, 128 GB *unified* CPU+GPU LPDDR5x |
| CUDA stack | system CUDA 12.0 | **CUDA 12.8** on DGXOS; the GB10-tuned PyTorch comes from the **NGC PyTorch image** (no host torch install) |
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
  `--backend` flag yet — so the GPU run is driven by the small script in §7
  (it reuses the Phase 5 harness, keeping results directly comparable with
  the 3080 artifacts in `docs/benchmarks.md`).
- The Spark has **no global PyTorch**, and installing one host-wide is not
  recommended. The runtime gets PyTorch from the NGC base image (§4). The
  QAOA benchmark itself does not need torch at runtime — LightningGPU's
  CUDA dependencies are prebuilt aarch64 `-cu12` wheels (§1).

## 3. Pre-flight checks on the Spark (5 min)

```bash
nvidia-smi        # GB10 (Grace Blackwell) + driver
uname -m          # aarch64
docker --version  # Docker + NVIDIA runtime (DGXOS default)
```

The Spark has **no global PyTorch**, and installing one host-wide is not
recommended. Run the PyTorch / sm_121 pre-flight *inside* the NGC PyTorch
image that the runtime is built on:

```bash
# 1) Verify the tag has an arm64 entry:
docker manifest inspect nvcr.io/nvidia/pytorch:26.09-py3

# 2) Check the GB10 compute capability from inside the image:
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.09-py3 python - <<'EOF'
import torch
print(torch.__version__, torch.version.cuda)
print("sm:", torch.cuda.get_device_capability(0))   # expect (12, 1)
EOF
```

If `sm` does not report `(12, 1)`, stop — nothing GPU-side in this guide
will work on the wrong architecture.

## 4. Fresh runtime: NGC PyTorch container (primary path)

Build the runtime on top of the base PyTorch image — the standard DGXOS
workflow: the image supplies the GB10-tuned CUDA stack and a GB10-tuned
PyTorch (for pre-flights and introspection), and the QOPO environment is
built inside it with `uv`, exactly as on any dev box.

```bash
git clone https://github.com/dasobral/qaoa-portfolio.git
cd qaoa-portfolio
git checkout feature/runtime-spark

# 1) Pull the base image (verify the arm64 entry as in section 3):
docker pull nvcr.io/nvidia/pytorch:26.09-py3

# 2) Open an interactive container with the repo mounted:
docker run --rm --gpus all -it \
  -v $PWD:/work -w /work \
  nvcr.io/nvidia/pytorch:26.09-py3 bash

# 3) Inside the container (aarch64) — the usual repo setup:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Keep the project env outside the mount so `docker commit` can bake it in:
export UV_PROJECT_ENVIRONMENT=/opt/qaoa-env
uv sync --extra dev        # builds qaoa_portfolio_core (maturin) for aarch64
uv run qaoa-portfolio --help
uv run pytest -q           # optional CPU sanity pass
uv pip install "pennylane-lightning-gpu==0.45.0"
```

No torch install is needed anywhere in this flow: LightningGPU does not
depend on PyTorch — its CUDA stack (`custatevec-cu12` + the
`nvidia-*-cu12` runtime libraries) arrives as prebuilt aarch64 wheels
from PyPI. The image's PyTorch is only used for pre-flights/introspection
(and future cuQuantum work, section 9).

Device smoke test (finishes in seconds), still inside the container:

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

To avoid rebuilding the env on every run, commit the container after step 3
(env + built Rust extension are baked in; the mounted repo stays the single
source of truth):

```bash
docker commit <container-id> qopo-spark:spark
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

## 6. Alternative: host venv (no container)

If you deliberately skip the container (the Spark host itself, or a different
aarch64 box), the same `uv` setup works on the host — and, as of
LightningGPU 0.45, **no PyTorch is involved at all** (its CUDA dependencies
are prebuilt aarch64 `-cu12` wheels). Do not install torch globally on the
Spark:

```bash
git clone https://github.com/dasobral/qaoa-portfolio.git
cd qaoa-portfolio
git checkout feature/runtime-spark

# Rust toolchain (aarch64) + uv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Project env (repo convention: qaoa-env/, not .venv/)
export UV_PROJECT_ENVIRONMENT=qaoa-env
uv sync --extra dev        # builds qaoa_portfolio_core (maturin) for aarch64
uv run qaoa-portfolio --help
uv run pytest -q           # optional CPU sanity pass

# The GPU plugin — pin the same minor as the installed PennyLane (0.45.x)
uv pip install "pennylane-lightning-gpu==0.45.0"
```

The device smoke test from section 4 (`uv run python - <<'EOF' ...`) is the
same, and the torch introspection in the example script is optional there
(no global torch — it will just print a skip note).

## 7. Run the example: QAOA on the GB10 vs classical baselines

Save as `spark_example.py` in the repo root, then run it in the section 4
container (with the env already built):

```bash
uv run python spark_example.py
```

If you committed the image (`qopo-spark:spark`), the whole run is one line:

```bash
docker run --rm --gpus all -v $PWD:/work -w /work \
  -e UV_PROJECT_ENVIRONMENT=/opt/qaoa-env \
  qopo-spark:spark uv run python spark_example.py
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

## 8. RAM budgeting (read before big runs)

- The GB10's 128 GB pool is **shared between CPU and GPU**. The locally
  deployed coding agent holds a large slice of it while active — **close or
  suspend agent sessions before running the experiment**, as the GPU needs
  as much of the pool as possible.
- Measured Python-side peaks on the 3080 (CPU simulator, `tracemalloc`):
  n = 8 → 3.4 MB, n = 16 → 676 MB, **n = 20 → 16.2 GB**. Add the CUDA
  context + cuStateVec workspace (a few GB) when on `lightning.gpu`.
- Run one solve at a time; no parallel `pytest` or second benchmark.
- Monitor with `free -g` (the unified pool) and `nvidia-smi` (SM
  utilization; memory counters are shared on GB10 and may read N/A).
- If OOM at n = 20: lower `repeats`/`max_iterations`, or step down to
  n = 16 first.

## 9. cuQuantum (NVIDIA's quantum SDK) — maximum GPU optimization, real integration work

- `pip install cuquantum` is a meta package that pulls `cuquantum-cu12`
  (or `-cu13`), which installs the prebuilt NVIDIA libraries —
  cuStateVec, cuStabilizer, cuPauliProp, cuTensorNet, cuDensityMat — as
  **aarch64 wheels** (e.g. `custatevec-cu12`, ~73 MB prebuilt
  `libcustatevec.so`). Verified: the current wheels embed Blackwell cubins
  (`sm_100` + `sm_120`), which run on GB10's sm_121 via within-major-version
  binary compatibility.
- `pip install cuquantum-python` adds NVIDIA's official Python bindings
  (low-level: state-vector propagation, gates, expectation values).
- Note that LightningGPU 0.45 already *links* cuStateVec — `custatevec-cu12`
  is its core dependency — so the same prebuilt library powers both paths;
  what changes is the QAOA loop itself (cost Hamiltonian evolutions,
  X-mixer, optimizer steps, probability extraction).
- **But there is no drop-in PennyLane device built on cuQuantum** on PyPI —
  adopting it means rewriting the QAOA loop (`quantum_backend.py`: cost
  Hamiltonian evolutions, X-mixer, optimizer steps, probability extraction)
  against the cuStateVec API instead of `qml.qnode`. That is a separate
  engineering effort, not a config flag.
- **Verdict:** run the LightningGPU path first (sections 4 and 7). If its
  throughput on GB10 is the bottleneck for the comparison, then open the
  cuQuantum integration as a follow-up phase of this branch.


Decision guide:

| Path | When | Effort |
|---|---|---|
| NGC PyTorch container + `lightning.gpu` (section 4) | Default — run the experiment this week | ~30–45 min setup |
| Host venv (section 6) | No container on the box / lighter isolation | ~30 min |
| cuQuantum rewrite (section 9) | Only if LightningGPU's GB10 throughput disappoints | Separate project |

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `lightning.gpu` device not found / plugin import error | Plugin missing from `qaoa-env`, or PennyLane/plugin minor mismatch — both must be 0.45.x (`uv pip list \| grep -i pennylane`). |
| `QuantumBackendError: Unable to create PennyLane backend 'lightning.gpu'` | Allowlist — apply §5 option A (option B in the script already covers this). |
| `no kernel image` / kernel errors from `custatevec-cu12` or `nvidia-*-cu12` | The wheels embed `sm_100` + `sm_120` cubins (sm_120 SASS runs on sm_121). Re-run the section 3 pre-flight (expect `(12, 1)`), confirm you installed the **aarch64** wheels, and if it still fails, escalate to the section 9 cuQuantum path. |
| CUDA version mismatch between the `-cu12` wheels and the system | Everything is CUDA 12.x (12.8 on DGXOS); do not mix in `-cu13` wheels or the 3080's 12.0 assumptions. |
| GPU not visible inside the container (`nvidia-smi` fails in the container) | The container was started without GPU access — re-run with `--gpus all` (and verify: `docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.09-py3 nvidia-smi`). |
| cuQuantum wheels refuse to initialize on the GB10 | Check `nvidia-smi` driver vs the wheel's CUDA major version (use the `-cu12` wheels on CUDA 12.8); if sm_120 SASS does not JIT on your driver, wait for/force a cuQuantum build that lists sm_121 explicitly. |
| `lightning.gpu` does not work on GB10 at all | Fall back to `lightning.qubit` (CPU) so the classical comparison still runs, and document the gap; escalate to the section 9 cuQuantum path. |

## 11. Recording results & next steps for this branch

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
  4. (If needed) cuQuantum-based backend prototype, following section 9.
