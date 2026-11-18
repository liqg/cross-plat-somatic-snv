#!/usr/bin/env python3
"""Pre-submission checks for the cross-plat-somatic-snv manuscript package.

Run from the repository root:
    python analysis/preflight.py
or:
    make preflight

Checks are local only.  External services (iThenticate/Turnitin, Proofig/
ImageTwin, professional language editing) still need to be run separately.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import openpyxl
from docx import Document
from PIL import Image

REPO = Path(os.environ.get("CPSNV_REPO_ROOT", Path(__file__).resolve().parents[1]))
PROJECT = Path(os.environ.get("CPSNV_PROJECT_ROOT", REPO.parent))
RESP = PROJECT / "response"
SUB = RESP / "submission-v4"


class Checker:
    def __init__(self):
        self.fail = 0
        self.warn = 0
        self.ok = 0

    def check(self, label, condition, detail="", level="FAIL"):
        if condition:
            self.ok += 1
            print(f"PASS  {label}")
        elif level == "WARN":
            self.warn += 1
            print(f"WARN  {label}" + (f" — {detail}" if detail else ""))
        else:
            self.fail += 1
            print(f"FAIL  {label}" + (f" — {detail}" if detail else ""))

    def summary(self):
        print(f"\nSummary: {self.ok} passed, {self.warn} warnings, {self.fail} failures")
        return 1 if self.fail else 0


def docx_texts(path):
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def pdf_pages(path):
    try:
        out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return None


def pdf_fonts_embedded(path):
    try:
        out = subprocess.check_output(["pdffonts", str(path)], text=True, stderr=subprocess.DEVNULL)
        rows = [line for line in out.splitlines() if line.strip() and not line.startswith("name") and not line.startswith("----")]
        return bool(rows) and all(re.search(r"\byes\b", line) for line in rows)
    except Exception:
        return None


def main():
    c = Checker()
    print(f"Repository: {REPO}")
    print(f"Manuscript project: {PROJECT}\n")

    manuscript = SUB / "Manuscript.docx"
    highlighted = RESP / "manuscript-submit-revised-v4-highlighted.pdf"
    response_md = RESP / "response-letter-point-by-point-v4.md"
    response_docx = SUB / "Response_to_Reviewers.docx"
    response_pdf = SUB / "Response_to_Reviewers.pdf"
    af1 = SUB / "Additional_file_1_Supplementary_tables.xlsx"
    af2 = SUB / "Additional_file_2_Supplementary_figures.pdf"
    cover_md = RESP / "cover-letter-v4.md"
    cover_docx = SUB / "Cover_letter.docx"
    cover_pdf = SUB / "Cover_letter.pdf"

    # ---- files
    for p in [manuscript, highlighted, response_md, response_docx, response_pdf,
              af1, af2, cover_md, cover_docx, cover_pdf]:
        c.check(f"exists: {p.relative_to(PROJECT)}", p.exists())

    if not manuscript.exists():
        return c.summary()
    texts = docx_texts(manuscript)
    full = "\n".join(texts)
    headings = set(texts)

    # ---- journal format
    abstract = next((t for t in texts if t.startswith("Background:") and "Results:" in t and "Conclusions:" in t), "")
    abstract_words = len(abstract.split())
    c.check("abstract <= 350 words", 0 < abstract_words <= 350, f"{abstract_words} words")
    c.check("abstract has Background/Results/Conclusions",
            all(k in abstract for k in ("Background:", "Results:", "Conclusions:")))
    for h in ["Abstract", "Background", "Methods", "Results", "Discussion", "Conclusions",
              "Declarations", "Additional files", "References", "Figure legends"]:
        c.check(f"section heading: {h}", h in headings)

    for h in ["Ethics approval and consent to participate", "Consent for publication",
              "Availability of data and materials", "Competing interests", "Funding",
              "Authors' contributions", "Acknowledgements"]:
        c.check(f"declaration heading: {h}", h in headings)

    # ---- citations and table ranges
    table_refs = sorted(set(int(x) for x in re.findall(r"Table\s+S(\d+)", full)))
    c.check("tables S1-S35 all cited", table_refs == list(range(1, 36)),
            f"missing={[i for i in range(1, 36) if i not in table_refs]}")
    fig_refs = sorted(set(int(x) for x in re.findall(r"Fig\.\s*S(\d+)", full)))
    c.check("Figs. S1-S12 all cited", fig_refs == list(range(1, 13)),
            f"missing={[i for i in range(1, 13) if i not in fig_refs]}")
    c.check("Additional file 1 description says S1-S35", "Tables S1-S35" in full)
    c.check("Additional file 2 description present", "Additional file 2" in full)

    # ---- stale/internal strings
    for bad in ["Somatic Mutation Detection", "liqg3/cross-plat", "Materials and Methods",
                "same trio", "Supplementary Table", "/mnt/nfsp", "/biobase"]:
        c.check(f"no stale string: {bad}", bad not in full, bad)

    # ---- response letter
    if response_md.exists():
        letter = response_md.read_text(encoding="utf-8")
        c.check("response: 35 reviewer comments", len(re.findall(r"^## Reviewer", letter, re.M)) == 35)
        c.check("response: 35 change summaries", letter.count("**Changes in the manuscript.**") == 35)
        c.check("response: no old repository URL", "liqg3/cross-plat" not in letter)
        c.check("response: references Table S35", "Table S35" in letter)
        c.check("response: cites new repository", "github.com/liqg/cross-plat-somatic-snv" in letter)

    # ---- cover letter
    if cover_md.exists():
        cover = cover_md.read_text(encoding="utf-8")
        c.check("cover letter: <=500 words", len(cover.split()) <= 500, f"{len(cover.split())} words")
        for term in ["ce52d314-ef89-4aca-8fea-7b0e22a3b080", "Somatic SNV Detection",
                     "github.com/liqg/cross-plat-somatic-snv", "CNP0009527"]:
            c.check(f"cover letter mentions: {term}", term in cover)
        c.check("cover letter: no liqg3 URL", "liqg3/cross-plat" not in cover)

    # ---- PDFs
    for name, expected in [(highlighted, 30), (response_pdf, 26), (af2, 13), (cover_pdf, 1)]:
        pages = pdf_pages(name) if name.exists() else None
        c.check(f"PDF pages: {name.name}", pages == expected, f"got {pages}, expected {expected}")
    for name in [highlighted, response_pdf, af2]:
        emb = pdf_fonts_embedded(name) if name.exists() else None
        c.check(f"PDF fonts embedded: {name.name}", emb is not False, f"embedded={emb}")

    # ---- supplementary workbook
    if af1.exists():
        wb = openpyxl.load_workbook(af1, read_only=True, data_only=True)
        c.check("workbook: 36 sheets", len(wb.sheetnames) == 36, str(len(wb.sheetnames)))
        c.check("workbook: Table S35 exists", "Table S35" in wb.sheetnames)
        if "Table S35" in wb.sheetnames:
            c.check("workbook: S35 has 480 data rows", wb["Table S35"].max_row == 484,
                    str(wb["Table S35"].max_row))
        if "Table S1" in wb.sheetnames:
            s1 = wb["Table S1"]
            c.check("workbook: S1 preparation columns",
                    s1.cell(2, 15).value == "Preparation type" and
                    s1.cell(2, 16).value == "Independent preparation")
        bad_cells = 0
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                for v in row:
                    if isinstance(v, str) and any(e in v for e in ("#REF!", "#VALUE!", "#DIV/0!", "#N/A")):
                        bad_cells += 1
        c.check("workbook: no error cells", bad_cells == 0, str(bad_cells))

    # ---- figures
    figdir = SUB / "Figures"
    c.check("main figures directory exists", figdir.is_dir())
    for n in range(1, 5):
        for ext in ("pdf", "png", "tif"):
            c.check(f"main figure: Fig{n}.{ext}", (figdir / f"Fig{n}.{ext}").exists())
        tif = figdir / f"Fig{n}.tif"
        if tif.exists():
            im = Image.open(tif)
            dpi = im.info.get("dpi")
            c.check(f"Fig{n}.tif 300 dpi", bool(dpi) and abs(float(dpi[0]) - 300) < 2, str(dpi))

    # ---- repository assets and cleanliness
    c.check("main plot_data present", (REPO / "analysis" / "r_figures" / "plot_data").is_dir())
    c.check("supplementary figure data present", (REPO / "analysis" / "supp_figure_data").is_dir())
    c.check("schematics present", (REPO / "analysis" / "r_figures" / "schematics" / "Fig1a.png").exists())
    c.check("complete figures present", (REPO / "analysis" / "r_figures" / "complete_figures" / "Fig4.tif").exists())

    try:
        status = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
        c.check("git working tree clean", not status, "uncommitted changes", level="WARN")
    except Exception as e:
        c.check("git status available", False, str(e), level="WARN")

    try:
        r = subprocess.run([sys.executable, "tests/test_smoke.py", "-v"], cwd=REPO,
                           capture_output=True, text=True)
        c.check("repo smoke tests", r.returncode == 0, r.stdout[-500:] + r.stderr[-500:])
    except Exception as e:
        c.check("repo smoke tests", False, str(e))

    return c.summary()


if __name__ == "__main__":
    sys.exit(main())
