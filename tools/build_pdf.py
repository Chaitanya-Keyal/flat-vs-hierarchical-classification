"""Builds report/results_matrix.tex with latexmk and copies the PDF to the repo root, where the
submission takes it from.

    uv run python -m tools.build_pdf      (after tools.run_all has written results/numbers.tex)
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def main():
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build"]
    subprocess.run([*command, "results_matrix.tex"], cwd=REPORT, check=True, capture_output=True)
    pdf = REPORT / "build" / "results_matrix.pdf"
    text = subprocess.run(["pdftotext", pdf, "-"], check=True, capture_output=True, text=True)
    if "??" in text.stdout:
        raise SystemExit("results_matrix.pdf has ?? where a number is missing")
    shutil.copyfile(pdf, ROOT / "results_matrix.pdf")
    print(f"wrote {(ROOT / 'results_matrix.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
