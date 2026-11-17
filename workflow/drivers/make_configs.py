#!/usr/bin/env python3
"""Generate per-stage JSON configs from the sample / simulation tables.

Usage
-----
    python workflow/drivers/make_configs.py --config config/config.yaml --kind align
    python workflow/drivers/make_configs.py --config config/config.yaml --kind simulation
    python workflow/drivers/make_configs.py --config config/config.yaml --kind benchmark
    python workflow/drivers/make_configs.py --config config/config.yaml --kind consistency
    python workflow/drivers/make_configs.py --config config/config.yaml --kind twist_hybrid

The generated files use only paths under ``data_root`` / ``work_root``; no
absolute internal path is written.  FASTQ file names are derived from the
``sample_id`` column of ``config/samples.tsv``:

    $data_root/fastq/{sample_id}_R1.fq.gz
    $data_root/fastq/{sample_id}_R2.fq.gz

For the Twist/HG001 simulation the tumour BAMs are the marked-duplicate BAMs
from the Twist runs, by default under
``$work_root/5.twist_errorRates_mdup/pipeline/bwamem``.  Override with
``--twist-bam-dir`` if they are elsewhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REPO = Path(__file__).resolve().parents[2]
PLATFORMS = [
    ("FASTASeq_300", "FASTASeq 300", "FA"),
    ("GenoLab_M", "GenoLab M", "GL"),
    ("SURFSeq_5000", "SURFSeq 5000", "SF"),
    ("NovaSeq_6000", "NovaSeq 6000", "NV"),
    ("NovaSeq_X", "NovaSeq X", "NX"),
    ("MGISEQ-T7", "DNBSEQ-T7", "T7"),
]


def load_yaml(path):
    if yaml is None:
        raise SystemExit("PyYAML is required: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(p, root):
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def read_tsv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_samples(repo):
    rows = read_tsv(repo / "config" / "samples.tsv")
    return {(r["platform"], r["sample"], r["library_id"]): r["sample_id"] for r in rows}


def sample_id(samples, platform_display, sample, library):
    key = (platform_display, sample, library)
    if key not in samples:
        raise KeyError(f"No sample_id for {key}; check config/samples.tsv")
    return samples[key]


def bam_path(work_root, sid):
    return str(work_root / "0.align" / "pipeline" / "bwamem" / sid / f"{sid}.bam")


def pair_rows(sims, pair):
    for r in sims:
        if r["pair_id"] == pair:
            return r
    raise KeyError(pair)


def pair_data(samples, work_root, sim, depths, platform_ids=None):
    data = []
    for pid, display, _ in PLATFORMS:
        if platform_ids and pid not in platform_ids:
            continue
        donor_lib = sim["donor_dnbseq_library"] if pid == "MGISEQ-T7" else sim["donor_truseq_library"]
        bg_lib = sim["background_dnbseq_library"] if pid == "MGISEQ-T7" else sim["background_truseq_library"]
        t_sid = sample_id(samples, display, sim["donor"], donor_lib)
        n_sid = sample_id(samples, display, sim["background"], bg_lib)
        data.append({
            "id": pid,
            "tumor": bam_path(work_root, t_sid),
            "normal": bam_path(work_root, n_sid),
        })
    return data


def intersect_bed(panel_bed, *samples):
    """Return the panel BED intersected with the given GIAB samples.

    The released BED names sort the sample IDs, so an unordered donor/background
    pair (e.g. HG002/HG001) resolves to the same file as its sorted counterpart
    (HG001/HG002).
    """
    label = "_".join(sorted(samples))
    path = panel_bed.parent / f"intersect_{label}_panel.sort.bed"
    if not path.exists():
        raise FileNotFoundError(f"Missing intersection BED: {path}")
    return path


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    ap.add_argument("--kind", required=True,
                    choices=["align", "simulation", "benchmark", "consistency", "twist_hybrid"])
    ap.add_argument("--twist-bam-dir", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    repo = REPO
    data_root = resolve(cfg.get("data_root", "data"), repo)
    work_root = resolve(cfg.get("work_root", "data/work"), repo)
    ref = cfg.get("reference", {})
    panel_bed = resolve(ref.get("panel_bed", "reference/panel/T364V1.hg19.bed"), repo)
    genome = resolve(ref.get("genome_fasta", "data/reference/GRCh37/genome.fasta"), repo)
    samples = load_samples(repo)
    sims = read_tsv(repo / "config" / "simulations.tsv")

    if a.kind == "align":
        rows = read_tsv(repo / "config" / "samples.tsv")
        data = []
        for r in rows:
            sid = r["sample_id"]
            data.append({
                "sampleid": sid,
                "read1": [str(data_root / "fastq" / f"{sid}_R1.fq.gz")],
                "read2": [str(data_root / "fastq" / f"{sid}_R2.fq.gz")],
            })
        out = work_root / "0.align" / "config.json"
        write_json(out, {
            "genome": "GRCh37",
            "genome_fasta": str(genome),
            "bed_file": str(panel_bed),
            "fastp": "",
            "data": data,
        })

    elif a.kind == "simulation":
        for sim in sims:
            pair = sim["pair_id"]
            role = sim["role"]
            depths = [int(x) for x in sim["depths_M"].split(",")]
            if role == "train":
                posfreq = str(repo / "workflow" / "pos_files" / "train" / pair)
                out = work_root / "4.train" / pair / "config.json"
            else:
                posfreq = str(repo / "workflow" / "pos_files" / "test" / pair / f"pf{pair}_low.txt")
                out = work_root / "4.test" / pair / "config.json"
            bed = intersect_bed(panel_bed, sim["donor"], sim["background"])
            write_json(out, {
                "name": f"snv{pair}_{role}",
                "genome": "GRCh37",
                "genome_fasta": str(genome),
                "min_freq": 0.0025,
                "callers": ["mutect2", "strelka2", "varscan", "varscan_novo", "features"],
                "bed_file": str(bed),
                "posfreq": posfreq,
                "seed": int(sim["seed"]),
                "tumor_reads": depths,
                "normal_reads": 5,
                "data": pair_data(samples, work_root, sim, depths),
            })

    elif a.kind == "benchmark":
        pair = "21"
        sim = pair_rows(sims, pair)
        for row in read_tsv(repo / "config" / "benchmark.tsv"):
            depths = [int(x) for x in row["depths_M"].split(",")]
            out = work_root / "3.performance" / row["vaf_id"] / "config.json"
            write_json(out, {
                "name": f"snv21_{row['vaf_id']}",
                "genome": "GRCh37",
                "genome_fasta": str(genome),
                "min_freq": 0.0025,
                "callers": ["strelka2", "mutect2", "varscan_novo", "features"],
                "bed_file": str(intersect_bed(panel_bed, sim["donor"], sim["background"])),
                "posfreq": str(repo / row["posfreq"]),
                "seed": int(row["seed"]),
                "tumor_reads": depths,
                "normal_reads": 5,
                "data": pair_data(samples, work_root, sim, depths),
            })

    elif a.kind == "consistency":
        pair = "21"
        sim = pair_rows(sims, pair)
        out = work_root / "2.consistency" / "config.json"
        write_json(out, {
            "name": "snv21_actual",
            "genome": "GRCh37",
            "genome_fasta": str(genome),
            "min_freq": 0.0025,
            "callers": ["varscan"],
            "bed_file": str(intersect_bed(panel_bed, sim["donor"], sim["background"])),
            "posfreq": str(repo / "workflow" / "pos_files" / "consistency" / "pf_actual_0910.txt"),
            "seed": 6600,
            "tumor_reads": [150],
            "normal_reads": 5,
            "data": pair_data(samples, work_root, sim, [150]),
        })

    elif a.kind == "twist_hybrid":
        twist_dir = Path(a.twist_bam_dir) if a.twist_bam_dir else work_root / "5.twist_errorRates_mdup" / "pipeline" / "bwamem"
        normal_sid = sample_id(samples, "FASTASeq 300", "HG001", "GM23112901-1")
        data = []
        for r in read_tsv(repo / "config" / "twist_hybrid.tsv"):
            rid = r["replicate_id"]
            plat = r["platform"]
            plat_display = {"FASTASeq_300": "FASTASeq 300", "GenoLab_M": "GenoLab M",
                            "SURFSeq_5000": "SURFSeq 5000", "NovaSeq_6000": "NovaSeq 6000",
                            "NovaSeq_X": "NovaSeq X"}[plat]
            prefix = {"FASTASeq_300": "FA", "GenoLab_M": "GL", "SURFSeq_5000": "SF",
                      "NovaSeq_6000": "NV", "NovaSeq_X": "NX"}[plat]
            tumor_dir = f"{prefix}_1_{r['tumor_library']}_PE70M"
            tumor_bam = twist_dir / tumor_dir / f"{tumor_dir}.marked.bam"
            n_sid = sample_id(samples, plat_display, "HG001", r["normal_library"])
            data.append({
                "id": f"{prefix}_1_{rid.split('_')[-1]}",
                "tumor": str(tumor_bam),
                "normal": bam_path(work_root, n_sid),
            })
        out = work_root / "8.twist_HG001_mix" / "config.json"
        write_json(out, {
            "name": "twist_hg001_mix",
            "genome": "GRCh37",
            "genome_fasta": str(genome),
            "min_freq": 0.0025,
            "callers": ["mutect2", "strelka2", "varscan_novo", "features"],
            "bed_file": str(intersect_bed(panel_bed, "HG001")),
            "posfreq": str(repo / "workflow" / "pos_files" / "twist_hg001" / "posfile.txt"),
            "seed": 1000,
            "tumor_reads": [140],
            "normal_reads": 5,
            "data": data,
        })


if __name__ == "__main__":
    main()
