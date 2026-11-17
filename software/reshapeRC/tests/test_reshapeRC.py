#!/usr/bin/env python3
"""Tests for reshapeRC.py.

Run from the repository root:

    python3 -m unittest discover -s software/reshapeRC/tests -v

The tests use a tiny synthetic genome for format/edge-case checks.  The two
large published input/expected pairs and the GRCh37 FASTA are also checked
when those paths are available; otherwise those tests are skipped.
"""

from __future__ import annotations

import gzip
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reshapeRC.py"
sys.path.insert(0, str(ROOT))

import reshapeRC  # noqa: E402  (path is set up above)

# Hard-coded on purpose: changing the script's template must fail the tests.
ZERO = "0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00"

A_REST = "1:60.00:35.00:0.10:1:0:0.50:0.01:35.00:1:0.50:100.00:0.49"
C_REST = "0:" + ZERO[len("0:") :]
G_REST = "2:60.00:36.00:0.20:2:0:0.60:0.02:36.00:2:0.60:101.00:0.50"
T_REST = "7:59.90:34.50:0.30:4:3:0.40:0.01:34.50:4:0.40:102.00:0.48"
N_REST = "0:" + ZERO[len("0:") :]
DEL_T_REST = "1:60.00:0.00:0.00:1:0:0.30:0.01:1.00:1:0.10:90.00:0.40"
INS_A_REST = "1:60.00:0.00:0.00:0:1:0.70:0.01:1.00:0:0.20:90.00:0.40"

GENOME_RECORDS: Tuple[Tuple[str, Sequence[str]], ...] = (
    ("chr1", ("AACCGGTT", "AACCGGTT")),
    ("chr2", ("ACGTN",)),
    ("chr3", ("acgtn",)),
)


def write_fasta_and_fai(path: Path) -> None:
    """Write a small FASTA plus a matching .fai index (wrapped lines)."""
    entries: List[Tuple[str, int, int, int, int]] = []
    offset = 0
    with path.open("wb") as handle:
        for name, lines in GENOME_RECORDS:
            header = (">" + name + " description\n").encode("ascii")
            handle.write(header)
            offset += len(header)
            seq_offset = offset
            length = 0
            line_bases = 0
            line_width = 0
            for i, line in enumerate(lines):
                encoded = line.encode("ascii") + b"\n"
                handle.write(encoded)
                offset += len(encoded)
                length += len(line)
                if i == 0:
                    line_bases = len(line)
                    line_width = len(encoded)
            entries.append((name, length, seq_offset, line_bases, line_width))
    with path.with_name(path.name + ".fai").open("w", encoding="ascii") as handle:
        for entry in entries:
            handle.write("\t".join(str(value) for value in entry) + "\n")


def readcount_line(
    chrom: str,
    pos: int,
    ref: str,
    depth: int,
    entries: Sequence[Tuple[str, str]],
) -> str:
    return "\t".join([chrom, str(pos), ref, str(depth)] + [
        name + ":" + rest for name, rest in entries
    ])


def expected_row(
    chrom: str,
    pos: int,
    ref: str,
    depth: int,
    left: str,
    right: str,
    base: str,
    rest: str,
) -> str:
    return "\t".join(
        [chrom, str(pos), ref, str(depth), left, right, base] + rest.split(":")
    )


class SyntheticGenomeTest(unittest.TestCase):
    """Unit tests using a tiny synthetic reference."""

    fasta_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="reshapeRC-test-")
        cls.fasta_path = Path(cls._tmp.name) / "genome.fasta"
        write_fasta_and_fai(cls.fasta_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def open_fasta(self, backend: str) -> reshapeRC.FastaAccess:
        return reshapeRC.open_fasta(str(self.fasta_path), backend)

    def convert(
        self,
        text: str,
        backend: str = "fai",
        include_extra_entries: bool = True,
    ) -> List[str]:
        fasta = self.open_fasta(backend)
        try:
            return list(
                reshapeRC.iter_rows(
                    io.StringIO(text), fasta, include_extra_entries
                )
            )
        finally:
            fasta.close()

    def test_fetch_all_backends(self) -> None:
        backends = ["fai"]
        try:
            import pysam  # noqa: F401
        except ImportError:
            pass
        else:
            backends.append("pysam")

        expected = {
            "chr1": "AACCGGTTAACCGGTT",
            "chr2": "ACGTN",
            "chr3": "ACGTN",  # context is returned upper-cased
        }
        for backend in backends:
            with self.subTest(backend=backend):
                fasta = self.open_fasta(backend)
                try:
                    for chrom, sequence in expected.items():
                        self.assertEqual(
                            fasta.get_length(chrom),
                            {"chr1": 16, "chr2": 5, "chr3": 5}[chrom],
                        )
                        self.assertEqual(fasta.fetch(chrom, 0, 100), sequence)
                        self.assertEqual(fasta.fetch(chrom, 3, 7), sequence[3:7])
                finally:
                    fasta.close()

    def test_fai_fetch_across_line_boundary(self) -> None:
        fasta = self.open_fasta("fai")
        try:
            # chr1 is wrapped as 8 bases per line.
            self.assertEqual(fasta.fetch("chr1", 6, 10), "TTAA")
            self.assertEqual(fasta.fetch("chr1", 7, 9), "TA")
        finally:
            fasta.close()

    def test_full_output_and_indels(self) -> None:
        line = readcount_line(
            "chr1",
            8,
            "T",
            10,
            [
                ("=", ZERO),
                ("A", A_REST),
                ("C", C_REST),
                ("G", G_REST),
                ("T", T_REST),
                ("N", N_REST),
                ("-T", DEL_T_REST),
            ],
        )
        expected = [
            expected_row("chr1", 8, "T", 10, "T", "A", "A", A_REST),
            expected_row("chr1", 8, "T", 10, "T", "A", "C", C_REST),
            expected_row("chr1", 8, "T", 10, "T", "A", "G", G_REST),
            expected_row("chr1", 8, "T", 10, "T", "A", "T", T_REST),
            expected_row("chr1", 8, "T", 10, "T", "A", "N", N_REST),
            expected_row("chr1", 8, "T", 10, "T", "A", "-T", DEL_T_REST),
        ]
        self.assertEqual(self.convert(line), expected)

    def test_zero_rows_for_missing_bases(self) -> None:
        line = readcount_line(
            "chr1",
            8,
            "T",
            3,
            [
                ("=", ZERO),
                ("A", A_REST),
                ("-A", INS_A_REST),
            ],
        )
        expected = [
            expected_row("chr1", 8, "T", 3, "T", "A", "A", A_REST),
            expected_row("chr1", 8, "T", 3, "T", "A", "C", ZERO),
            expected_row("chr1", 8, "T", 3, "T", "A", "G", ZERO),
            expected_row("chr1", 8, "T", 3, "T", "A", "T", ZERO),
            expected_row("chr1", 8, "T", 3, "T", "A", "N", ZERO),
            expected_row("chr1", 8, "T", 3, "T", "A", "-A", INS_A_REST),
        ]
        self.assertEqual(self.convert(line), expected)

        canonical_only = [
            row for row in expected if row.split("\t")[6] in reshapeRC.CANONICAL_BASES
        ]
        self.assertEqual(
            self.convert(line, include_extra_entries=False), canonical_only
        )

    def test_base_order_is_fixed_and_equals_is_skipped(self) -> None:
        line = readcount_line(
            "chr1",
            8,
            "T",
            9,
            [
                ("N", N_REST),
                ("G", G_REST),
                ("T", T_REST),
                ("=", "5:" + ZERO[2:]),  # nonzero '=' must still be skipped
                ("A", A_REST),
                ("C", C_REST),
                ("-A", INS_A_REST),
                ("+G", INS_A_REST),
            ],
        )
        rows = self.convert(line)
        self.assertEqual(
            [row.split("\t")[6] for row in rows],
            ["A", "C", "G", "T", "N", "-A", "+G"],
        )
        self.assertTrue(all("=" not in row.split("\t")[6] for row in rows))

    def test_comment_and_blank_lines(self) -> None:
        line = readcount_line("chr1", 8, "T", 1, [("A", A_REST)])
        text = "\n# a comment\n\t\n" + line + "\n\n"
        self.assertEqual(len(self.convert(text)), 5)

    def test_boundary_context(self) -> None:
        line1 = readcount_line("chr1", 1, "A", 1, [("A", A_REST)])
        rows1 = self.convert(line1)
        self.assertEqual(rows1[0].split("\t")[4:6], ["N", "A"])

        line_end = readcount_line("chr1", 16, "T", 1, [("T", T_REST)])
        rows_end = self.convert(line_end)
        self.assertEqual(rows_end[0].split("\t")[4:6], ["T", "N"])

    def test_bad_position_and_contig(self) -> None:
        with self.assertRaises(reshapeRC.ReshapeError):
            self.convert(readcount_line("chr1", 17, "A", 1, [("A", A_REST)]))
        with self.assertRaises(reshapeRC.ReshapeError):
            self.convert(readcount_line("chrZ", 1, "A", 1, [("A", A_REST)]))
        with self.assertRaises(reshapeRC.ReshapeError):
            self.convert(readcount_line("chr1", 0, "A", 1, [("A", A_REST)]))

    def test_no_entries_gives_five_zero_rows(self) -> None:
        line = readcount_line("chr1", 8, "T", 0, [("=", ZERO)])
        rows = self.convert(line)
        self.assertEqual([row.split("\t")[6] for row in rows], ["A", "C", "G", "T", "N"])
        for row in rows:
            self.assertEqual(row.split("\t")[7:], ZERO.split(":"))

    def test_cli_gzip_stdin_stdout_and_fai_backend(self) -> None:
        line = readcount_line(
            "chr1",
            8,
            "T",
            10,
            [
                ("=", ZERO),
                ("A", A_REST),
                ("C", C_REST),
                ("G", G_REST),
                ("T", T_REST),
                ("N", N_REST),
                ("-T", DEL_T_REST),
            ],
        )
        expected = "\n".join(
            [
                expected_row("chr1", 8, "T", 10, "T", "A", "A", A_REST),
                expected_row("chr1", 8, "T", 10, "T", "A", "C", C_REST),
                expected_row("chr1", 8, "T", 10, "T", "A", "G", G_REST),
                expected_row("chr1", 8, "T", 10, "T", "A", "T", T_REST),
                expected_row("chr1", 8, "T", 10, "T", "A", "N", N_REST),
                expected_row("chr1", 8, "T", 10, "T", "A", "-T", DEL_T_REST),
            ]
        ) + "\n"

        with tempfile.TemporaryDirectory(prefix="reshapeRC-cli-") as tmp:
            input_gz = Path(tmp) / "input.rc.gz"
            output_gz = Path(tmp) / "output.rc.tsv.gz"
            with gzip.open(input_gz, "wt", encoding="utf-8", newline="") as handle:
                handle.write(line + "\n")

            command = [
                sys.executable,
                str(SCRIPT),
                "-i",
                str(input_gz),
                "-g",
                str(self.fasta_path),
                "-o",
                str(output_gz),
                "--fasta-backend",
                "fai",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with gzip.open(output_gz, "rt", encoding="utf-8", newline="") as handle:
                self.assertEqual(handle.read(), expected)

            # Plain stdin, as in the required gzip -dc | reshapeRC.py -i - case.
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "-i",
                    "-",
                    "-g",
                    str(self.fasta_path),
                    "--fasta-backend",
                    "fai",
                ],
                input=line + "\n",
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)

            # gzip-compressed stdin is detected by magic bytes too.
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "-i",
                    "-",
                    "-g",
                    str(self.fasta_path),
                    "--fasta-backend",
                    "fai",
                ],
                input=gzip.compress((line + "\n").encode("utf-8")),
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout.decode("utf-8"), expected)

    def test_help_version_and_errors(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--canonical-only", help_result.stdout)

        version_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--version"], capture_output=True, text=True
        )
        self.assertEqual(version_result.returncode, 0)
        self.assertIn(reshapeRC.__version__, version_result.stdout)

        missing = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("required", missing.stderr)

        unknown_contig = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "-i",
                "-",
                "-g",
                str(self.fasta_path),
                "--fasta-backend",
                "fai",
            ],
            input=readcount_line("chrZ", 1, "A", 1, [("A", A_REST)]) + "\n",
            capture_output=True,
            text=True,
        )
        self.assertEqual(unknown_contig.returncode, 1)
        self.assertIn("chrZ", unknown_contig.stderr)


_VALIDATION_ROOT = os.environ.get("CPSNV_RESHAPE_VALIDATION_ROOT", "")
PAIR_DIRS = {}
PAIR_FILES = {}
if _VALIDATION_ROOT:
    _root = Path(_VALIDATION_ROOT)
    PAIR_DIRS = {
        "pair1": _root / "8.twist_HG001_mix" / "varscan_novo",
        "pair2": _root / "6.twist_pair" / "varscan_novo",
    }
    PAIR_FILES = {
        "pair1": ("FASTASeq_300-1_reads140M.raw.snp.vcf.rc.gz",
                  "FASTASeq_300-1_reads140M.raw.snp.vcf.rc.tsv.gz"),
        "pair2": ("FA_1_GM23063001_GM23062701.raw.snp.vcf.rc.gz",
                  "FA_1_GM23063001_GM23062701.raw.snp.vcf.rc.tsv.gz"),
    }

GRCH37 = Path(os.environ.get("CPSNV_GENOME", "/nonexistent/GRCh37/genome.fasta"))


def _gzip_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


@unittest.skipUnless(GRCH37.exists(), "GRCh37 FASTA is not available")
class PublishedPairsTest(unittest.TestCase):
    """Byte-for-byte checks against the two published input/expected pairs."""

    def _run_pair(self, pair: str, backend: str = "auto") -> None:
        input_path = PAIR_DIRS[pair] / PAIR_FILES[pair][0]
        expected_path = PAIR_DIRS[pair] / PAIR_FILES[pair][1]
        if not input_path.exists() or not expected_path.exists():
            self.skipTest("published pair %s is not available" % pair)

        with tempfile.TemporaryDirectory(prefix="reshapeRC-pair-") as tmp:
            output_path = Path(tmp) / "output.tsv.gz"
            command = [
                sys.executable,
                str(SCRIPT),
                "-i",
                str(input_path),
                "-g",
                str(GRCH37),
                "-o",
                str(output_path),
                "--fasta-backend",
                backend,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_gzip_bytes(output_path), _gzip_bytes(expected_path))

    def test_pair1(self) -> None:
        self._run_pair("pair1")

    def test_pair2(self) -> None:
        self._run_pair("pair2")

    def test_pair1_fai_backend(self) -> None:
        self._run_pair("pair1", backend="fai")

    def test_pair2_fai_backend(self) -> None:
        self._run_pair("pair2", backend="fai")

    def test_pair1_stdin(self) -> None:
        input_path = PAIR_DIRS["pair1"] / PAIR_FILES["pair1"][0]
        expected_path = PAIR_DIRS["pair1"] / PAIR_FILES["pair1"][1]
        if not input_path.exists() or not expected_path.exists():
            self.skipTest("published pair pair1 is not available")

        raw_input = _gzip_bytes(input_path)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "-i",
                "/dev/stdin",
                "-g",
                str(GRCH37),
            ],
            input=raw_input,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, _gzip_bytes(expected_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
