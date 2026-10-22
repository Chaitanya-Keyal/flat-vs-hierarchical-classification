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


class BuildError(Exception):
    """latexmk failed; the message holds LaTeX's error lines."""


def build(report):
    """Run latexmk in the folder `report` and return the PDF it wrote. On a LaTeX error, or a
    number missing from results/numbers.tex, raise BuildError with the lines that say why."""
    run = subprocess.run([*COMMAND, "results_matrix.tex"], cwd=report, capture_output=True)
    log = report / "build" / "results_matrix.log"
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    if run.returncode != 0:
        first = next((i for i, line in enumerate(lines) if line.startswith("!")), None)
        raise BuildError("\n".join(lines[first : first + 3]) if first is not None else "no log")
    missing = [line for line in lines if line.startswith("MISSING NUMBER")]
    if missing:  # the PDF shows ?? in their place
        raise BuildError("\n".join(missing))
    return report / "build" / "results_matrix.pdf"


def main():
    try:
        pdf = build(REPORT)
    except BuildError as error:
        log = REPORT / "build" / "results_matrix.log"
        raise SystemExit(
            f"results_matrix.pdf not written:\n{error}\nThe full log is {log}"
        ) from None
    shutil.copyfile(pdf, ROOT / "results_matrix.pdf")
    print(f"wrote {(ROOT / 'results_matrix.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
