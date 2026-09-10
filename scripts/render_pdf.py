"""Render a Markdown document to a print-quality PDF.

Uses markdown-it-py (already a dependency of jupytext) plus headless Chrome or
Edge, both of which are present on this machine -- no LaTeX, no new packages.

    python scripts/render_pdf.py docs/datasets/DATASET_DOSSIER.md

Writes alongside the source unless --out is given. Keep the HTML with --keep-html
if you want to inspect or tweak the intermediate.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Cambria", serif;
  font-size: 10.2pt; line-height: 1.5; color: #14171a; margin: 0;
}
h1 {
  font-size: 21pt; line-height: 1.2; margin: 0 0 2mm; letter-spacing: -0.01em;
  font-weight: 600;
}
h2 {
  font-size: 14pt; margin: 9mm 0 3mm; padding-bottom: 1.6mm; font-weight: 600;
  border-bottom: 1.4px solid #c8ced4; break-after: avoid;
}
h3 {
  font-size: 11.4pt; margin: 6.5mm 0 2mm; font-weight: 600; color: #1d3f66;
  break-after: avoid;
}
h2 + p, h3 + p { margin-top: 0; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 2.6mm; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1.2mm; }
strong { font-weight: 600; color: #05070a; }
em { color: #33393f; }
code {
  font-family: "Cascadia Mono", "Consolas", monospace; font-size: 8.6pt;
  background: #eef1f4; padding: 0.4mm 1.1mm; border-radius: 2px; color: #23303d;
}
a { color: #1d4e89; text-decoration: none; word-break: break-word; }
table {
  border-collapse: collapse; width: 100%; margin: 3mm 0 4.5mm;
  font-size: 8.7pt; break-inside: avoid;
}
th, td {
  border: 0.7px solid #ccd2d8; padding: 1.5mm 2mm; text-align: left;
  vertical-align: top;
}
th { background: #eef1f4; font-weight: 600; color: #05070a; }
tr:nth-child(even) td { background: #fafbfc; }
hr { border: none; border-top: 0.8px solid #d8dde2; margin: 7mm 0; }
blockquote {
  margin: 3mm 0; padding: 0 0 0 4mm; border-left: 2.5px solid #c8ced4;
  color: #414850;
}
"""


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit("No Chrome or Edge found; cannot render PDF.")


def render_html(md_path: Path) -> str:
    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    body = md.render(md_path.read_text(encoding="utf-8"))
    title = md_path.stem.replace("_", " ").title()
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep-html", action="store_true")
    args = parser.parse_args()

    md_path = args.markdown.resolve()
    if not md_path.is_file():
        raise SystemExit(f"not found: {md_path}")
    out_pdf = (args.out or md_path.with_suffix(".pdf")).resolve()

    html_path = md_path.with_suffix(".html") if args.keep_html else Path(
        tempfile.mkdtemp(prefix="fpe_pdf_")) / f"{md_path.stem}.html"
    html_path.write_text(render_html(md_path), encoding="utf-8")

    browser = find_browser()
    # A fresh profile dir keeps this off the user's real browser state.
    profile = Path(tempfile.mkdtemp(prefix="fpe_pdf_profile_"))
    cmd = [
        str(browser), "--headless=new", "--disable-gpu", "--no-sandbox",
        f"--user-data-dir={profile}", "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf}", html_path.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not out_pdf.is_file():
        sys.stderr.write(result.stderr or result.stdout)
        raise SystemExit(f"{browser.name} did not produce {out_pdf}")

    shutil.rmtree(profile, ignore_errors=True)
    if not args.keep_html:
        shutil.rmtree(html_path.parent, ignore_errors=True)

    try:
        shown = out_pdf.relative_to(ROOT)
    except ValueError:
        shown = out_pdf
    print(f"{shown}  ({out_pdf.stat().st_size / 1024:,.0f} KB, via {browser.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
