#!/usr/bin/env python3
"""Fast smoke tests: no real sequencing data and no full pipeline run.

Run with:
    python tests/test_smoke.py -v
or
    bash tests/run_smoke.sh
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


class Workspace:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="cpsnv_smoke_"))
        self.data = self.root / "data"
        self.work = self.root / "work"
        self.results = self.root / "results"
        for p in (self.data, self.work, self.results):
            p.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env.update({
            "CPSNV_DATA_ROOT": str(self.data),
            "CPSNV_WORK_ROOT": str(self.work),
            "CPSNV_RESULTS": str(self.results),
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(REPO / "analysis"),
        })

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class CommonTests(unittest.TestCase):
    def test_position_table_and_distance(self):
        sys.path.insert(0, str(REPO / "analysis"))
        import common
        import pandas as pd
        p = Path(tempfile.mkdtemp()) / "pf.txt"
        pd.DataFrame([
            [1, 100, 0.01, "A", "G", "0/1", "Inf", 100, 2, 0, 0.005],
            [1, 200, 0.02, "C", "T", "0/1", 100, "Inf", 2, 1, 0.01],
        ]).to_csv(p, sep="\t", header=False, index=False)
        df = common.read_position_table(p)
        self.assertEqual(df.loc[0, "var"], "1:100:A:G")
        out = common.add_target_distance(df)
        self.assertEqual(int(out.loc[0, "dist"]), 100)
        self.assertEqual(common.platform_name("SURFSeq_5000"), "SURFSeq 5000")
        self.assertAlmostEqual(common.metrics(8, 2, 2)["F1"], 0.8)


class ErrorRateTests(unittest.TestCase):
    def test_one_file(self):
        ws = Workspace()
        try:
            sys.path.insert(0, str(REPO / "analysis" / "01_error_rates"))
            import build_error_rates_by_q as b
            d = ws.work / "FA_HG001_GM1"
            d.mkdir(parents=True)
            f = d / "FA_HG001_GM1.er.txt.gz"
            with gzip.open(f, "wt") as fh:
                fh.write("ref\tbase\tqual\tcount\nA\tA\t0\t100\nA\tC\t0\t10\n"
                         "A\tA\t30\t50\nA\tC\t30\t1\n")
            rows = b.one_file(f)
            q0 = [r for r in rows if r["q"] == 0][0]
            q30 = [r for r in rows if r["q"] == 30][0]
            self.assertAlmostEqual(q0["err"], 11 / 161)
            self.assertAlmostEqual(q30["err"], 1 / 51)
            self.assertEqual(q0["platform"], "FA")
        finally:
            ws.cleanup()


class TwistTruthTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bcftools"), "bcftools not installed")
    def test_truth_extraction(self):
        ws = Workspace()
        try:
            sys.path.insert(0, str(REPO / "analysis" / "06_twist"))
            import derive_twist_tables as d
            bed = ws.root / "panel.bed"
            bed.write_text("1\t99\t200\n")
            vcf = ws.root / "truth.vcf"
            vcf.write_text(textwrap.dedent("""\
                ##fileformat=VCFv4.2
                ##contig=<ID=1>
                ##contig=<ID=2>
                ##FILTER=<ID=PASS,Description="All filters passed">
                #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
                1\t150\t.\tA\tG\t.\tPASS\t.
                2\t150\t.\tA\tG\t.\tPASS\t.
                2\t29940529\t.\tA\tG\t.\tPASS\t.
            """))
            out = ws.root / "truth.tsv"
            truth = d.twist_truth(vcf, bed, out)
            self.assertEqual(truth, {"1:150:A:G"})
        finally:
            ws.cleanup()


class ConfigTests(unittest.TestCase):
    @staticmethod
    def _config_file(ws):
        cfg = ws.root / "config.yaml"
        cfg.write_text(textwrap.dedent(f"""\
            data_root: {ws.data}
            work_root: {ws.work}
            results_root: {ws.results}
            reference:
              genome_fasta: {ws.data}/genome.fa
              panel_bed: {REPO}/reference/panel/T364V1.hg19.bed
        """))
        return cfg

    def test_generate_simulation_configs(self):
        ws = Workspace()
        try:
            cfg = self._config_file(ws)
            subprocess.run([PY, REPO / "workflow" / "drivers" / "make_configs.py",
                            "--config", cfg, "--kind", "simulation"],
                           check=True, env=ws.env, capture_output=True, text=True)
            out = ws.work / "4.test" / "45" / "config.json"
            self.assertTrue(out.exists())
            with open(out) as fh:
                obj = json.load(fh)
            self.assertEqual(len(obj["data"]), 6)
            self.assertTrue(obj["posfreq"].endswith("pf45_low.txt"))
        finally:
            ws.cleanup()

    def test_all_generated_bed_paths_exist(self):
        ws = Workspace()
        try:
            cfg = self._config_file(ws)
            for kind in ("simulation", "benchmark", "consistency", "twist_hybrid"):
                subprocess.run([PY, REPO / "workflow" / "drivers" / "make_configs.py",
                                "--config", cfg, "--kind", kind],
                               check=True, env=ws.env, capture_output=True, text=True)
            configs = sorted(ws.work.rglob("config.json"))
            self.assertEqual(len(configs), 16)
            for path in configs:
                with open(path) as fh:
                    bed = json.load(fh)["bed_file"]
                self.assertTrue(os.path.exists(bed), f"{path}: missing {bed}")
            with open(ws.work / "4.train" / "21" / "config.json") as fh:
                pair21 = json.load(fh)
            self.assertEqual(Path(pair21["bed_file"]).name,
                             "intersect_HG001_HG002_panel.sort.bed")
            with open(ws.work / "8.twist_HG001_mix" / "config.json") as fh:
                twist = json.load(fh)
            self.assertEqual(Path(twist["bed_file"]).name,
                             "intersect_HG001_panel.sort.bed")
        finally:
            ws.cleanup()


class FigureScriptTests(unittest.TestCase):
    def test_authoritative_r_scripts(self):
        figdir = REPO / "analysis" / "r_figures"
        f1 = figdir / "01_fig1b_fig2d_fig3.R"
        f4 = figdir / "02_fig4.R"
        self.assertTrue(f1.exists())
        self.assertTrue(f4.exists())
        s1 = f1.read_text()
        self.assertIn('OUT <- "fig_260610"', s1)
        for name in ("fig1b.png", "fig2.png", "fig3.png"):
            self.assertIn(name, s1)
        s4 = f4.read_text()
        self.assertIn('OUT <- "fig_261028"', s4)
        self.assertIn("fig4.png", s4)
        for helper in ("readSamstats.R", "nbinit.R", "install.pkgs.R", "fsave.R"):
            self.assertTrue((figdir / "R" / helper).exists())
        # The approximate Python plotting scripts were removed; data-prep scripts remain.
        for old in ("analysis/03_benchmark/make_fig2d.py",
                    "analysis/04_model/make_fig3.py",
                    "analysis/06_twist/make_fig4.py"):
            self.assertFalse((REPO / old).exists())


class WorkflowTests(unittest.TestCase):
    def test_dry_run(self):
        ws = Workspace()
        try:
            cfg = ws.root / "config.yaml"
            cfg.write_text(textwrap.dedent(f"""\
                data_root: {ws.data}
                work_root: {ws.work}
                results_root: {ws.results}
                reference:
                  genome_fasta: {ws.data}/genome.fa
                  panel_bed: {REPO}/reference/panel/T364V1.hg19.bed
                tools:
                  posmix: {REPO}/software/bin/posmix
                  posfeature: {REPO}/software/bin/posfeature
                  varscan_jar: {REPO}/software/varscan/VarScan.v2.4.3.jar
                somaticxgb:
                  model: {REPO}/software/somaticxgb/models/SomaticXGB
            """))
            subprocess.run([PY, REPO / "workflow" / "run.py", "configs", "--config", cfg,
                            "--kind", "simulation"], env=ws.env, check=True,
                           capture_output=True, text=True)
            r = subprocess.run([PY, REPO / "workflow" / "run.py", "simulate", "--role", "test",
                                "--config", cfg, "--dry-run"], env=ws.env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("somatic_snv_posmix.smk", r.stdout)
            r = subprocess.run([PY, REPO / "workflow" / "run.py", "all", "--config", cfg,
                                "--dry-run"], env=ws.env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("align.smk", r.stdout)
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
