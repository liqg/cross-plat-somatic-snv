#!/usr/bin/env python3
"""Assemble complete Fig. 1-4 from the R data panels and schematic panels.

Fig. 1 and Fig. 2 are vertical composites of a schematic panel and the
corresponding R-rendered data panel.  Fig. 3 and Fig. 4 are already complete
data figures from the R scripts and are exported as PDF/PNG/TIFF at 300 dpi.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

WIDTH = 2000
GAP = 28


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compose(images: list[Image.Image], out: Path) -> None:
    scaled = []
    for im in images:
        h = round(im.height * WIDTH / im.width)
        scaled.append(im.resize((WIDTH, h), Image.LANCZOS))
    height = sum(im.height for im in scaled) + GAP * (len(scaled) - 1)
    canvas = Image.new("RGB", (WIDTH, height), "white")
    y = 0
    for im in scaled:
        canvas.paste(im, (0, y))
        y += im.height + GAP
    save(canvas, out)


def save(im: Image.Image, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out.with_suffix(".png"), dpi=(300, 300))
    im.save(out.with_suffix(".tif"), dpi=(300, 300), compression="tiff_lzw")
    im.save(out.with_suffix(".pdf"), resolution=300.0)
    print("wrote", out.with_suffix(".png"), im.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-figures", required=True, help="directory containing fig_260610/ and fig_261028/")
    ap.add_argument("--schematics", required=True, help="directory containing Fig1a.png and Fig2a-c.png")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = Path(a.r_figures)
    s = Path(a.schematics)
    out = Path(a.out)

    compose([load(s / "Fig1a.png"), load(r / "fig_260610" / "fig1b.png")], out / "Fig1")
    compose([load(s / "Fig2a-c.png"), load(r / "fig_260610" / "fig2.png")], out / "Fig2")
    save(load(r / "fig_260610" / "fig3.png"), out / "Fig3")
    save(load(r / "fig_261028" / "fig4.png"), out / "Fig4")


if __name__ == "__main__":
    main()
