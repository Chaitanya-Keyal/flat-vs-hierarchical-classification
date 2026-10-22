"""Builds report/results_matrix.tex with latexmk and copies the PDF to the repo root, where the
submission takes it from.

    uv run python -m tools.build_pdf      (after tools.run_all has written results/numbers.tex)
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
COMMAND = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build"]


def build(report):
    """Run latexmk in the folder `report` and return the PDF it wrote. On a LaTeX error, stop
    and show the error from the log."""
    run = subprocess.run([*COMMAND, "results_matrix.tex"], cwd=report, capture_output=True)
    log = report / "build" / "results_matrix.log"
    if run.returncode != 0:
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        first = next((i for i, line in enumerate(lines) if line.startswith("!")), None)
        error = "\n".join(lines[first : first + 3]) if first is not None else "(no error line)"
        raise SystemExit(f"latexmk failed:\n{error}\nThe full log is {log}")
    return report / "build" / "results_matrix.pdf"


def main():
    pdf = build(REPORT)
    text = subprocess.run(["pdftotext", pdf, "-"], check=True, capture_output=True, text=True)
    if "??" in text.stdout:
        raise SystemExit("results_matrix.pdf has ?? where a number is missing")
    shutil.copyfile(pdf, ROOT / "results_matrix.pdf")
    print(f"wrote {(ROOT / 'results_matrix.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
