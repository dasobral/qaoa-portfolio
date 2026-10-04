# QOPO benchmark dashboard

A static, dependency-free page (`index.html` + `styles.css` + `charts.js` +
`app.js`) that visualises the benchmark artifacts under `results/benchmarks/`.
Light theme following the dasobral.github.io / PQC-course tokens.

## Use

```bash
python front/build_data.py          # bundle results/ -> front/data/qopo-data.js
xdg-open front/index.html           # works from file://, no server needed
```

Re-run `build_data.py` whenever new artifacts land. The bundle is generated
and gitignored (like `results/`).

## Data layout

| Location | Host label |
|---|---|
| `results/benchmarks/*.json` | this machine (`--local-host`, default `rtx3080`) |
| `results/benchmarks/spark-gb10/*.json` | DGX Spark artifacts copied from the Spark |
| `results/benchmarks/**/runs-*.jsonl` | campaign manifests: host, backend, wall time, peak memory, status; failed/aborted runs without an artifact still appear in the run log |

Backend comes from the manifest, else the artifact's `config.qaoa.backend`,
else `default.qubit`. Campaign = artifact month (`created_utc`); archived
campaigns are drawn with hollow markers.

## Sections

1. **Quality** — approximation ratio per solver (benchmark preset only).
2. **Scaling** — time, `tracemalloc` peak, QAOA ratio and speed-up over the
   June 2026 RTX 3080 baseline vs portfolio size, per QAOA setting (`pL · it · restarts`).
3. **Optimizer budget** — ratio vs iterations, one panel per restart count.
4. **Depth** — `layers` suite: ratio and time vs p.
5. **Market** — out-of-sample tables per study.
6. **Run log** — every artifact and manifest entry.

Series colour follows host · backend (validated categorical palette); marker
shape encodes host (circle = RTX 3080, square = DGX Spark).
