#!/usr/bin/env python3
"""reshapeRC.py - convert bam-readcount output into the reshapeRC TSV layout.

This is a standalone implementation of the ``reshapeRC`` transformation.  It reads ``bam-readcount`` output (plain text or gzip-compressed),
looks up the reference context in an indexed genome FASTA, and writes one
tab-separated row per allele.  There is no header.

Output columns
--------------
1  CHROM
2  POS
3  REF (verbatim from the bam-readcount line)
4  depth (verbatim from the bam-readcount line)
5  reference base immediately left of POS
6  reference base immediately right of POS
7  allele name (A, C, G, T, N, or an indel such as -G)
8-20  the 13 numeric bam-readcount fields after the allele name, verbatim

For every position the rows for A, C, G, T and N are emitted in that order.
Bases that are missing from the input line are emitted with the canonical
zero-filled record:

    0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00

The ``=`` entry is always skipped.  By default all other allele entries
(including indels such as ``+A`` and ``-G``) are emitted after A/C/G/T/N in
their original input order; this is the published output layout and is
required to reproduce the published .rc.tsv.gz files byte for byte.  Use
``--canonical-only`` to emit only A, C, G, T and N.

Numeric fields are copied as strings, so the original bam-readcount
formatting (for example ``60.00``) is preserved.

Boundary positions (POS=1 or the last base of a contig) have no base on one
side; ``N`` is used for the unavailable context base.  FASTA context is
returned in upper case.

Usage
-----
    reshapeRC.py -i input.rc[.gz] -g genome.fasta > output.rc.tsv
    reshapeRC.py -i input.rc.gz -g genome.fasta -o output.rc.tsv.gz
    gzip -dc input.rc.gz | reshapeRC.py -i /dev/stdin -g genome.fasta

Only the standard library and (optionally) pysam are required.  If pysam is
not available, a small random-access reader based on the FASTA ``.fai``
index is used instead; the whole genome is never loaded into memory.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import os
import signal
import sys
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

__version__ = "1.0.0"

CANONICAL_BASES: Tuple[str, ...] = ("A", "C", "G", "T", "N")
ZERO_FIELDS = "0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00"
GZIP_SUFFIXES = (".gz", ".bgz", ".bgzf")
STDIN_NAMES = ("-", "/dev/stdin")
STDOUT_NAMES = ("-", "/dev/stdout")


class ReshapeError(Exception):
    """A user-facing error (bad input, unavailable reference, etc.)."""


def _warn(message: str) -> None:
    sys.stderr.write("reshapeRC.py: warning: %s\n" % message)


# ---------------------------------------------------------------------------
# FASTA access
# ---------------------------------------------------------------------------


class FastaAccess:
    """Minimal random-access FASTA interface used by the converter."""

    def get_length(self, chrom: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def fetch(self, chrom: str, start: int, end: int) -> str:  # pragma: no cover
        """Return the 0-based half-open sequence [start, end), upper-cased."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class PysamFasta(FastaAccess):
    """FASTA access through pysam.FastaFile (htslib + .fai)."""

    def __init__(self, path: str) -> None:
        import pysam  # imported lazily so --fasta-backend fai works without it

        self._fa = pysam.FastaFile(path)
        self._lengths: Dict[str, int] = {}

    def _length(self, chrom: str) -> int:
        if chrom not in self._lengths:
            try:
                self._lengths[chrom] = int(self._fa.get_reference_length(chrom))
            except (KeyError, ValueError) as exc:
                raise ReshapeError(
                    "reference sequence %r is not present in the genome FASTA" % chrom
                ) from exc
        return self._lengths[chrom]

    def get_length(self, chrom: str) -> int:
        return self._length(chrom)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length = self._length(chrom)
        if start < 0:
            start = 0
        if end > length:
            end = length
        if start >= end:
            return ""
        try:
            return self._fa.fetch(chrom, start, end).upper()
        except (KeyError, ValueError) as exc:
            raise ReshapeError(
                "cannot fetch %s:%d-%d from the genome FASTA: %s"
                % (chrom, start + 1, end, exc)
            ) from exc

    def close(self) -> None:
        self._fa.close()


class FaiFasta(FastaAccess):
    """Random-access FASTA reader that uses only a ``.fai`` index.

    Only the bytes covering the requested interval are read, so memory use is
    independent of genome size.  The FASTA itself must be uncompressed.
    """

    def __init__(self, path: str) -> None:
        if path.lower().endswith(GZIP_SUFFIXES):
            raise ReshapeError(
                "the .fai fallback cannot read a compressed FASTA (%s); "
                "install pysam or decompress the FASTA" % path
            )
        fai_path = path + ".fai"
        if not os.path.exists(fai_path):
            raise ReshapeError(
                "FASTA index not found: %s (create it with 'samtools faidx %s')"
                % (fai_path, path)
            )
        self._index: Dict[str, Tuple[int, int, int, int]] = {}
        try:
            with open(fai_path, "rt", encoding="ascii", newline="") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        raise ReshapeError(
                            "malformed .fai line %d in %s: expected at least "
                            "5 tab-separated columns" % (line_no, fai_path)
                        )
                    name, length, offset, line_bases, line_width = parts[:5]
                    try:
                        self._index[name] = (
                            int(length),
                            int(offset),
                            int(line_bases),
                            int(line_width),
                        )
                    except ValueError as exc:
                        raise ReshapeError(
                            "malformed .fai line %d in %s: non-integer field"
                            % (line_no, fai_path)
                        ) from exc
        except OSError as exc:
            raise ReshapeError("cannot read FASTA index %s: %s" % (fai_path, exc)) from exc

        try:
            with open(path, "rb") as probe:
                if probe.read(2) == b"\x1f\x8b":
                    raise ReshapeError(
                        "the .fai fallback cannot read a gzip/bgzip-compressed "
                        "FASTA (%s); install pysam or decompress the FASTA" % path
                    )
            self._fh = open(path, "rb")
        except OSError as exc:
            raise ReshapeError("cannot open genome FASTA %s: %s" % (path, exc)) from exc

    def _record(self, chrom: str) -> Tuple[int, int, int, int]:
        try:
            return self._index[chrom]
        except KeyError as exc:
            raise ReshapeError(
                "reference sequence %r is not present in the genome FASTA" % chrom
            ) from exc

    def get_length(self, chrom: str) -> int:
        return self._record(chrom)[0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        length, offset, line_bases, line_width = self._record(chrom)
        if start < 0:
            start = 0
        if end > length:
            end = length
        if start >= end:
            return ""
        if line_bases <= 0:
            raise ReshapeError("invalid line_bases in .fai for %r" % chrom)

        chunks: List[bytes] = []
        pos = start
        while pos < end:
            line_offset = (pos // line_bases) * line_width + (pos % line_bases)
            take = min(end - pos, line_bases - (pos % line_bases))
            try:
                self._fh.seek(offset + line_offset)
                data = self._fh.read(take)
            except OSError as exc:
                raise ReshapeError(
                    "cannot read %s:%d-%d from the genome FASTA: %s"
                    % (chrom, pos + 1, pos + take, exc)
                ) from exc
            if len(data) != take:
                raise ReshapeError(
                    "unexpected end of genome FASTA while reading %s:%d-%d; "
                    "is the .fai index stale?" % (chrom, pos + 1, end)
                )
            chunks.append(data)
            pos += take
        return b"".join(chunks).decode("ascii").upper()

    def close(self) -> None:
        self._fh.close()


def open_fasta(path: str, backend: str = "auto") -> FastaAccess:
    """Open *path* with pysam or the .fai fallback."""
    if not os.path.exists(path):
        raise ReshapeError("genome FASTA does not exist: %s" % path)

    pysam_error: Optional[BaseException] = None
    if backend in ("auto", "pysam"):
        try:
            return PysamFasta(path)
        except ImportError as exc:
            pysam_error = exc
            if backend == "pysam":
                raise ReshapeError(
                    "pysam is not installed; use --fasta-backend fai or install pysam"
                ) from exc
        except Exception as exc:  # pysam/htslib errors vary by version
            pysam_error = exc
            if backend == "pysam":
                raise ReshapeError(
                    "failed to open genome FASTA with pysam (%s): %s" % (path, exc)
                ) from exc

    try:
        return FaiFasta(path)
    except ReshapeError as exc:
        if pysam_error is not None:
            raise ReshapeError(
                "%s (the pysam backend also failed: %s)" % (exc, pysam_error)
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Input / output helpers
# ---------------------------------------------------------------------------


def _stdin_is_gzip() -> bool:
    """Peek at stdin without consuming data, if the stream supports it."""
    raw = getattr(sys.stdin, "buffer", None)
    if raw is None:
        return False
    peek = getattr(raw, "peek", None)
    if peek is not None:
        try:
            return peek(2)[:2] == b"\x1f\x8b"
        except (OSError, ValueError):
            return False
    return False


def _is_gzip_file(path: str) -> bool:
    if path.lower().endswith(GZIP_SUFFIXES):
        return True
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def open_input_text(path: str, stack: contextlib.ExitStack) -> TextIO:
    """Open a readcount file (or stdin) as text, using gzip when appropriate."""
    if path in STDIN_NAMES:
        if _stdin_is_gzip():
            raw = sys.stdin.buffer
            gz = gzip.GzipFile(fileobj=raw, mode="rb")
            wrapper = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            stack.callback(wrapper.close)
            return wrapper
        # sys.stdin already provides a text stream; do not close it.
        return sys.stdin

    try:
        if _is_gzip_file(path):
            handle = gzip.open(path, "rt", encoding="utf-8", newline="")
        else:
            handle = open(path, "rt", encoding="utf-8", newline="")
    except OSError as exc:
        raise ReshapeError("cannot open input file %s: %s" % (path, exc)) from exc
    stack.callback(handle.close)
    return handle


def open_output_text(path: Optional[str], stack: contextlib.ExitStack) -> TextIO:
    """Open the output as text, gzip-compressing files with a .gz suffix."""
    if path is None or path in STDOUT_NAMES:
        return sys.stdout
    try:
        if path.lower().endswith(GZIP_SUFFIXES):
            handle = gzip.open(path, "wt", encoding="utf-8", newline="\n")
        else:
            handle = open(path, "wt", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ReshapeError("cannot open output file %s: %s" % (path, exc)) from exc
    stack.callback(handle.close)
    return handle


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def _parse_entries(
    tokens: Sequence[str], line_no: int, include_extra_entries: bool
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """Split allele entries into canonical and extra (indel) entries.

    The value is the original bam-readcount field string after the allele
    name (13 colon-separated numeric fields).
    """
    canonical: Dict[str, str] = {}
    extras: List[Tuple[str, str]] = []
    for token in tokens:
        if not token:
            continue
        name, sep, rest = token.partition(":")
        if not sep:
            _warn(
                "line %d: ignoring allele field without numeric data: %r"
                % (line_no, token)
            )
            continue
        if name == "=":
            continue
        if len(name) == 1 and name.upper() in CANONICAL_BASES:
            name = name.upper()
        if name in CANONICAL_BASES:
            # Real bam-readcount output has exactly one entry per base.  Keep
            # the first if a malformed file repeats one.
            if name not in canonical:
                canonical[name] = rest
            continue
        if include_extra_entries:
            extras.append((name, rest))
    return canonical, extras


def _context_bases(fasta: FastaAccess, chrom: str, pos: int) -> Tuple[str, str]:
    """Return the reference bases at POS-1 and POS+1 (1-based POS)."""
    if pos > 1:
        sequence = fasta.fetch(chrom, pos - 2, pos + 1)
        left = sequence[0] if len(sequence) >= 1 else "N"
        right = sequence[2] if len(sequence) >= 3 else "N"
    else:
        # No base to the left; fetch POS and POS+1 only.
        sequence = fasta.fetch(chrom, 0, 2)
        left = "N"
        right = sequence[1] if len(sequence) >= 2 else "N"
    return left, right


def iter_rows(
    lines: Iterator[str],
    fasta: FastaAccess,
    include_extra_entries: bool = True,
) -> Iterator[str]:
    """Yield output rows (without trailing newline) for readcount *lines*."""
    for line_no, raw_line in enumerate(lines, 1):
        # The original trims leading/trailing whitespace (including CR/LF).
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) < 4:
            raise ReshapeError(
                "line %d: expected at least 4 tab-separated fields "
                "(CHROM, POS, REF, depth), got %d" % (line_no, len(fields))
            )

        chrom, pos_text, ref, depth = fields[:4]
        if not chrom:
            raise ReshapeError("line %d: empty chromosome name" % line_no)
        try:
            pos = int(pos_text)
        except ValueError as exc:
            raise ReshapeError(
                "line %d: POS is not an integer: %r" % (line_no, pos_text)
            ) from exc
        if pos < 1:
            raise ReshapeError("line %d: POS must be >= 1, got %d" % (line_no, pos))

        contig_length = fasta.get_length(chrom)
        if pos > contig_length:
            raise ReshapeError(
                "line %d: %s:%d is beyond the end of the reference (%d bp)"
                % (line_no, chrom, pos, contig_length)
            )

        canonical, extras = _parse_entries(fields[4:], line_no, include_extra_entries)
        left, right = _context_bases(fasta, chrom, pos)
        prefix = "\t".join((chrom, pos_text, ref, depth, left, right))

        for base in CANONICAL_BASES:
            rest = canonical.get(base, ZERO_FIELDS)
            yield prefix + "\t" + base + "\t" + rest.replace(":", "\t")
        for name, rest in extras:
            yield prefix + "\t" + name + "\t" + rest.replace(":", "\t")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reshapeRC.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Convert bam-readcount output into the reshapeRC TSV layout "
            "(no header, one row per allele)."
        ),
        epilog="""\
output columns:
  1 CHROM   2 POS   3 REF   4 depth   5 left base   6 right base
  7 allele (A,C,G,T,N or indel)   8-20 numeric bam-readcount fields

For every position, A/C/G/T/N rows are emitted in that order, with the exact
zero record for bases missing from the input.  The '=' entry is skipped.
By default, other entries (indels such as +A/-G) are also emitted, matching
the published files; use --canonical-only to suppress them.

examples:
  reshapeRC.py -i input.rc.gz -g genome.fasta > output.rc.tsv
  reshapeRC.py -i input.rc.gz -g genome.fasta -o output.rc.tsv.gz
  gzip -dc input.rc.gz | reshapeRC.py -i /dev/stdin -g genome.fasta
""",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FILE",
        help="bam-readcount output (.rc[.gz]); '-' or /dev/stdin for stdin",
    )
    parser.add_argument(
        "-g",
        "--genome",
        required=True,
        metavar="FASTA",
        help="indexed reference genome FASTA (.fai, chromosome names as in input)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="output TSV (default: stdout); a .gz suffix enables gzip output",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        metavar="N",
        help="accepted for compatibility with the published CLI; ignored",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="emit only A,C,G,T,N rows (drop +A/-A style indel entries)",
    )
    parser.add_argument(
        "--fasta-backend",
        choices=("auto", "pysam", "fai"),
        default="auto",
        help="FASTA random-access backend (default: auto; 'fai' forces the "
        "pure-Python .fai reader)",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.threads < 1:
        raise ReshapeError("-t/--threads must be >= 1")

    if args.output and args.output not in STDOUT_NAMES:
        output_real = os.path.realpath(args.output)
        if args.input not in STDIN_NAMES and os.path.realpath(args.input) == output_real:
            raise ReshapeError("input and output are the same file: %s" % args.input)
        for protected in (args.genome, args.genome + ".fai"):
            if os.path.realpath(protected) == output_real:
                raise ReshapeError(
                    "refusing to overwrite the reference file: %s" % args.output
                )

    with contextlib.ExitStack() as stack:
        fasta = open_fasta(args.genome, args.fasta_backend)
        stack.callback(fasta.close)
        instream = open_input_text(args.input, stack)
        outstream = open_output_text(args.output, stack)

        for row in iter_rows(instream, fasta, not args.canonical_only):
            outstream.write(row)
            outstream.write("\n")
        outstream.flush()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ReshapeError as exc:
        sys.stderr.write("reshapeRC.py: error: %s\n" % exc)
        return 1
    except BrokenPipeError:
        # A downstream consumer such as `head` closed the pipe.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0
    except KeyboardInterrupt:
        sys.stderr.write("reshapeRC.py: interrupted\n")
        return 130


if __name__ == "__main__":
    # Behave like a conventional Unix filter when a downstream pipe closes.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
