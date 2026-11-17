#!/usr/bin/env python3
"""Thin orchestration layer for the stage Snakefiles.

Examples
--------
    python workflow/run.py configs --kind all
    python workflow/run.py align --cores 64
    python workflow/run.py simulate --role train --cores 100
    python workflow/run.py simulate --role test --cores 100
    python workflow/run.py simulate --role benchmark --cores 100
    python workflow/run.py predict --pairs 45 54
    python workflow/run.py analysis --step figures
    python workflow/run.py all --dry-run

Every command is printed before it runs; ``--dry-run`` only prints.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, dry=False, cwd=None):
    line = " ".join(str(x) for x in cmd)
    print(("DRY-RUN: " if dry else "RUN: ") + line, flush=True)
    if not dry:
        subprocess.run([str(x) for x in cmd], check=True, cwd=cwd)


def load_cfg(path):
    if yaml is None:
        raise SystemExit("PyYAML required")
    with open(path) as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[
        "configs", "align", "simulate", "predict", "analysis", "all", "doctor"
    ])
    ap.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    ap.add_argument("--kind", default="all",
                    choices=["align", "simulation", "benchmark", "consistency", "twist_hybrid", "all"])
    ap.add_argument("--role", default="train", choices=["train", "test", "benchmark"])
    ap.add_argument("--pairs", nargs="*", default=["45", "54"])
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--step", default="all")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg_path = Path(a.config)
    if not cfg_path.exists():
        if a.command == "doctor":
            print("No config file at", cfg_path)
            print("Copy config/config.example.yaml to config/config.yaml and edit paths.")
            return
        raise SystemExit(f"Config not found: {cfg_path}. Copy config/config.example.yaml first.")
    cfg = load_cfg(cfg_path)
    data_root = Path(cfg.get("data_root", "data"))
    work_root = Path(cfg.get("work_root", "data/work"))
    if not data_root.is_absolute():
        data_root = (REPO / data_root).resolve()
    if not work_root.is_absolute():
        work_root = (REPO / work_root).resolve()
    env = os.environ.copy()
    env.setdefault("CPSNV_DATA_ROOT", str(data_root))
    env.setdefault("CPSNV_WORK_ROOT", str(work_root))
    env.setdefault("CPSNV_RESULTS", str((REPO / cfg.get("results_root", "results")).resolve()))

    if a.command == "doctor":
        checks = {
            "config": cfg_path,
            "genome": Path(cfg["reference"]["genome_fasta"]) if Path(cfg["reference"]["genome_fasta"]).is_absolute()
                      else REPO / cfg["reference"]["genome_fasta"],
            "panel_bed": REPO / cfg["reference"]["panel_bed"],
            "posmix": REPO / cfg["tools"]["posmix"],
            "posfeature": REPO / cfg["tools"]["posfeature"],
            "varscan_jar": REPO / cfg["tools"]["varscan_jar"],
            "errorRates": REPO / cfg["tools"]["errorRates"],
            "SomaticXGB model": REPO / cfg["somaticxgb"]["model"],
        }
        for k, v in checks.items():
            print(f"{k:18s} {'OK ' if Path(v).exists() else 'MISSING'} {v}")
        return

    make = [PY, REPO / "workflow" / "drivers" / "make_configs.py", "--config", cfg_path]

    if a.command in ("configs", "all"):
        kinds = ["align", "simulation", "benchmark", "consistency", "twist_hybrid"] if a.kind == "all" else [a.kind]
        for k in kinds:
            run(make + ["--kind", k], a.dry_run)

    if a.command in ("align", "all"):
        run(make + ["--kind", "align"], a.dry_run)
        run(["snakemake", "-s", REPO / "workflow" / "align" / "align.smk",
             "--configfile", work_root / "0.align" / "config.json",
             "--directory", work_root / "0.align", "--cores", a.cores], a.dry_run)

    if a.command in ("simulate", "all"):
        if a.command == "all":
            roles = ["train", "test", "benchmark"]
        else:
            roles = [a.role]
        for role in roles:
            if role == "train":
                dirs = sorted((work_root / "4.train").glob("*/config.json"))
            elif role == "test":
                dirs = sorted((work_root / "4.test").glob("*/config.json"))
            else:
                dirs = sorted((work_root / "3.performance").glob("*/config.json"))
            for cfg_file in dirs:
                run(["snakemake", "-s", REPO / "workflow" / "posmix" / "somatic_snv_posmix.smk",
                     "--configfile", cfg_file, "--directory", cfg_file.parent,
                     "--cores", a.cores], a.dry_run)

    if a.command in ("predict", "all"):
        for pair in a.pairs:
            pdir = work_root / "4.test" / pair
            run(["Rscript", REPO / "workflow" / "somaticxgb" / "predict_SomaticXGB.R",
                 "--pairs", pdir, "--outdir", pdir / "pred",
                 "--model", REPO / cfg["somaticxgb"]["model"]], a.dry_run)

    if a.command in ("analysis", "all"):
        steps = ["figures"] if a.step == "all" else [a.step]
        for step in steps:
            script = REPO / "analysis" / "07_figures_tables" / "make_all.py"
            run([PY, script, "--step", "all" if step == "figures" else step], a.dry_run)


if __name__ == "__main__":
    main()
