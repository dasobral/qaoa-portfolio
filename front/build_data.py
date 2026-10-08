#!/usr/bin/env python3
"""Bundle benchmark artifacts into ``front/data/qopo-data.js`` for the dashboard.

Scans ``results/benchmarks/`` (artifacts produced on this host) and every
direct sub-directory of it (datasets copied from other hosts, e.g.
``results/benchmarks/spark-gb10/``). Run manifests (``runs-*.jsonl``) written
by the experiment campaigns are joined by artifact filename to attach host,
backend, wall time and memory. Standard library only; the output is a plain
script (``window.QOPO_DATA = {...}``) so ``index.html`` works from ``file://``.

Usage::

    python front/build_data.py [--results results/benchmarks] [--local-host rtx3080]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
SUITES = {"quality", "scaling", "layers", "market", "probe-large"}

HOSTS = {
    "rtx3080": {"label": "RTX 3080", "hardware": "RTX 3080 10 GB · x86_64 · 31 GB RAM"},
    "spark-gb10": {
        "label": "DGX Spark",
        "hardware": "GB10 Grace Blackwell · aarch64 · 128 GB unified",
    },
}


def _read_manifests(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Map artifact filename -> manifest entry for every runs-*.jsonl in a directory."""
    entries: Dict[str, Dict[str, Any]] = {}
    for path in sorted(directory.glob("runs-*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get("artifact") or f"__run__{entry.get('run_id')}"
            entries[Path(key).name if entry.get("artifact") else key] = entry
    return entries


def _campaign(created: str) -> str:
    try:
        return datetime.strptime(created[:8], "%Y%m%d").strftime("%Y-%m")
    except ValueError:
        return "unknown"


def _record(rec: Dict[str, Any], artifact: int) -> Dict[str, Any]:
    meta = rec.get("metadata") or {}
    solver = rec.get("solver_name")
    is_qaoa = solver == "qaoa"
    if solver == "brute_force" and (
        meta.get("method") == "exact_enumeration"
        or ("method" not in meta and (rec.get("num_assets") or 0) > 20)
    ):
        # The Rust brute force refuses n > 20. The harness records its NumPy
        # enumeration as method "exact_enumeration"; older campaign-wrapper
        # artifacts carry no method, so n > 20 identifies them. Either way the
        # timings are not Rust timings.
        solver = "exact_enum"
    return {
        "a": artifact,
        "s": solver,
        "n": rec.get("num_assets"),
        "r": rec.get("approximation_ratio"),
        "t": rec.get("elapsed_ms"),
        "m": rec.get("peak_memory_kb"),
        "i": rec.get("run_index"),
        "sd": rec.get("seed"),
        "L": meta.get("layers") if is_qaoa else None,
        "it": meta.get("max_iterations") if is_qaoa else None,
        "rs": meta.get("num_restarts") if is_qaoa else None,
        "op": meta.get("optimizer") if is_qaoa else None,
    }


def _probe_records(data: Dict[str, Any], artifact: int) -> List[Dict[str, Any]]:
    """Flatten a ``probe-large`` artifact (n > 20, written by the campaign's
    out-of-harness probe script) into the same record shape as the CLI suites.

    Ratios there are taken against an exact NumPy 2^n enumeration; that
    enumeration is kept as solver ``exact_enum`` (not ``brute_force``, whose
    timings come from the Rust core).
    """
    q = data.get("qaoa_config") or {}
    out: List[Dict[str, Any]] = []
    for rec in data.get("records") or []:
        base = {
            "a": artifact,
            "n": rec.get("num_assets"),
            "i": rec.get("run_index"),
            "sd": rec.get("seed"),
            "m": None,
            "L": None,
            "it": None,
            "rs": None,
            "op": None,
        }
        if rec.get("qaoa_ratio") is not None:
            out.append(
                {
                    **base,
                    "s": "qaoa",
                    "r": rec["qaoa_ratio"],
                    "t": rec.get("qaoa_ms"),
                    "m": rec.get("qaoa_peak_python_kb"),
                    "L": q.get("layers"),
                    "it": q.get("max_iterations"),
                    "rs": q.get("num_restarts"),
                    "op": q.get("optimizer"),
                }
            )
        if rec.get("sa_ratio") is not None:
            out.append(
                {
                    **base,
                    "s": "simulated_annealing",
                    "r": rec["sa_ratio"],
                    "t": rec.get("sa_ms"),
                }
            )
        if rec.get("exact_enum_s") is not None:
            out.append(
                {**base, "s": "exact_enum", "r": 1.0, "t": rec["exact_enum_s"] * 1000.0}
            )
    return out


def collect(results: Path, local_host: str) -> Dict[str, Any]:
    artifacts: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    studies: List[Dict[str, Any]] = []
    orphans: List[Dict[str, Any]] = []

    directories = [(results, local_host)] + [
        (d, d.name) for d in sorted(results.iterdir()) if d.is_dir() and d.name in HOSTS
    ]
    for directory, host in directories:
        manifest = _read_manifests(directory)
        seen = set()
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            suite = str(data.get("suite", "")).replace("spark-", "")
            if suite not in SUITES:
                continue
            probe = suite == "probe-large"
            if probe:  # n > 20 probe: same instances/metric as the quality suite
                suite = "quality"
                args_ = data.get("args") or {}
                first = (data.get("records") or [{}])[0]
                data.setdefault(
                    "config",
                    {
                        "num_assets": args_.get("assets"),
                        "target_assets": first.get("target_assets"),
                        "repeats": args_.get("repeats"),
                        "qaoa": data.get("qaoa_config"),
                    },
                )
            seen.add(path.name)
            entry = manifest.get(path.name, {})
            config = data.get("config") or {}
            qaoa = config.get("qaoa") or {}
            backend = (
                entry.get("backend")
                or data.get("device")
                or qaoa.get("backend")
                or "default.qubit"
            )
            created = str(data.get("created_utc", ""))
            idx = len(artifacts)
            artifacts.append(
                {
                    "id": idx,
                    "file": (
                        f"{directory.relative_to(ROOT)}/{path.name}"
                        if directory.is_relative_to(ROOT)
                        else path.name
                    ),
                    "suite": suite,
                    "host": entry.get("host") or host,
                    "backend": backend,
                    "created": created,
                    "campaign": _campaign(created),
                    "n": config.get("num_assets"),
                    "target": config.get("target_assets"),
                    "repeats": config.get("repeats"),
                    "wall_s": entry.get("wall_s"),
                    "max_rss_kb": entry.get("max_rss_kb"),
                    "peak_pool_mb": entry.get("peak_pool_used_mb"),
                    "status": entry.get("status", "ok"),
                    "command": entry.get("command"),
                    "notes": entry.get("notes"),
                    "run_id": entry.get("run_id"),
                    "probe": probe,
                    "concurrent": bool(entry.get("concurrent_with")),
                }
            )
            if probe:
                records.extend(_probe_records(data, idx))
            else:
                for rec in data.get("records") or []:
                    records.append(_record(rec, idx))
            study = data.get("study")
            if study:
                studies.append(
                    {
                        "a": idx,
                        "symbols": study.get("symbols"),
                        "start": study.get("start_date"),
                        "end": study.get("end_date"),
                        "split": study.get("split"),
                        "in_rows": study.get("in_sample_rows"),
                        "out_rows": study.get("out_of_sample_rows"),
                        "solvers": {
                            name: {
                                "ratio": s["record"].get("approximation_ratio"),
                                "assets": s["record"].get("selected_assets"),
                                "ms": s["record"].get("elapsed_ms"),
                                **(s.get("out_of_sample") or {}),
                            }
                            for name, s in (study.get("solvers") or {}).items()
                        },
                    }
                )
        # Failed / aborted runs that never wrote an artifact are still results.
        for key, entry in manifest.items():
            if key.startswith("__run__") or (
                entry.get("artifact") and Path(entry["artifact"]).name not in seen
            ):
                orphans.append({**entry, "host": entry.get("host") or host})

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "hosts": HOSTS,
        "artifacts": artifacts,
        "records": records,
        "studies": studies,
        "runs_without_artifact": orphans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "benchmarks")
    parser.add_argument(
        "--local-host", default="rtx3080", help="host label for top-level artifacts"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "front" / "data" / "qopo-data.js"
    )
    args = parser.parse_args()

    bundle = collect(args.results.resolve(), args.local_host)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "window.QOPO_DATA = " + json.dumps(bundle, separators=(",", ":")) + ";\n"
    )
    print(
        f"{len(bundle['artifacts'])} artifacts, {len(bundle['records'])} records, "
        f"{len(bundle['studies'])} market studies -> {args.out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
