"""Builds the submission zip and checks it the way a grader will use it.

    uv run python -m tools.package                  dist/2023B4A70727H_A1.zip
    uv run python -m tools.package ID [ID ...]      one zip per student ID
    uv run python -m tools.package --python "CMD"   run the scripts with another interpreter, e.g.
        "uv run --no-project --python 3.9 --with numpy==1.26.4 --with pandas==2.2.3 python"

Each zip holds exactly the eight files the handout names, flat at its root, dated by the last
commit that changed one of them. It is then checked: results_matrix.pdf is what report/ builds
from results/numbers.tex now; the scripts import only numpy, pandas, the standard library and
each other; no prohibited library is named anywhere and no path is absolute; and, unzipped into
an empty folder next to a copy of Frogs_MFCCs.csv, every script runs with one plain command and
prints exactly what results/partN.txt recorded.
"""

import argparse
import ast
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from common import DATA_FILE
from tools import build_pdf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "part1_decision_tree.py",
    "part2_bagging_boosting.py",
    "part3_random_forest.py",
    "part4_hierarchical.py",
    "part5_naive_bayes.py",
]
FILES = ["common.py", *SCRIPTS, "README.txt", "results_matrix.pdf"]
OWN = {Path(f).stem for f in FILES if f.endswith(".py")}
ALLOWED = {"numpy", "pandas"} | set(sys.stdlib_module_names) | OWN
FORBIDDEN = ("sklearn", "scikit", "scipy", "statsmodels", "torch", "tensorflow", "keras",
             "xgboost", "lightgbm", "matplotlib")  # fmt: skip


def stamp():
    """The date of the last commit that changed one of the files, in the committer's time zone,
    so that the same commit of a full clone always gives the same zip, byte for byte (a shallow
    clone may not hold that commit, and dates the files by its oldest one)."""
    command = ["git", "log", "-1", "--format=%cd", "--date=format:%Y %m %d %H %M %S", "--", *FILES]
    try:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    except OSError:  # git is not installed
        run = None
    if run is None or run.returncode != 0 or not run.stdout.strip():
        return (1980, 1, 1, 0, 0, 0)  # no git history: the earliest date a zip can hold
    return tuple(int(v) for v in run.stdout.split())


def build(student_id):
    path = ROOT / "dist" / f"{student_id}_A1.zip"
    path.parent.mkdir(exist_ok=True)
    date = stamp()
    with zipfile.ZipFile(path, "w") as zf:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=date)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            zf.writestr(info, (ROOT / name).read_bytes())
    return path


def check_contents(folder):
    """Problems with the unzipped files, as messages."""
    problems = []
    texts = {name: (folder / name).read_text() for name in FILES if not name.endswith(".pdf")}
    if shutil.which("pdftotext") is None:
        print("  results_matrix.pdf: text NOT SCANNED, pdftotext (poppler-utils) is not installed")
    else:
        pdf = folder / "results_matrix.pdf"
        run = subprocess.run(["pdftotext", pdf, "-"], check=True, capture_output=True, text=True)
        texts["results_matrix.pdf"] = run.stdout
    for name, text in texts.items():
        lowered = text.lower()
        problems += [f"{name} names {word}" for word in FORBIDDEN if word in lowered]
        problems += [f"{name} has an absolute path" for p in ("/home/", "c:\\") if p in lowered]
    for name in FILES:
        if name.endswith(".py"):
            for node in ast.walk(ast.parse(texts[name])):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                problems += [
                    f"{name} imports {m}" for m in modules if m.split(".")[0] not in ALLOWED
                ]
    return problems


def check_pdf():
    """Rebuild the PDF from report/ and results/ in an empty folder and compare it, byte for
    byte, with the one that ships (the build has no date or random ID in it)."""
    if shutil.which("latexmk") is None:
        print("  results_matrix.pdf: NOT CHECKED, latexmk is not installed")
        return []
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("report/results_matrix.tex", "results/numbers.tex"):
            (Path(tmp) / name).parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / name, Path(tmp) / name)
        try:
            fresh = build_pdf.build(Path(tmp) / "report").read_bytes()
        except build_pdf.BuildError as error:
            print("  results_matrix.pdf: FAILED")
            return [f"report/ does not build (run python -m tools.build_pdf for the log):\n{error}"]
    same = fresh == (ROOT / "results_matrix.pdf").read_bytes()
    print(f"  results_matrix.pdf: {'ok' if same else 'FAILED'}")
    return [] if same else ["results_matrix.pdf is out of date: run python -m tools.build_pdf"]


def check_runs(folder, python):
    """Run every script the way a grader would and compare with the recorded output."""
    shutil.copyfile(ROOT / DATA_FILE, folder / DATA_FILE)
    problems = []
    for i, script in enumerate(SCRIPTS, start=1):
        run = subprocess.run([*python, script], cwd=folder, capture_output=True, text=True)
        problem = None
        if run.returncode != 0:
            problem = f"{script} failed:\n{run.stderr}"
        elif run.stdout != (ROOT / "results" / f"part{i}.txt").read_text():
            problem = f"{script} prints something other than results/part{i}.txt"
        elif run.stderr:
            problem = f"{script} wrote to stderr:\n{run.stderr}"
        print(f"  {script}: {'ok' if problem is None else 'FAILED'}")
        problems += [problem] if problem else []
    return problems


def main():
    parser = argparse.ArgumentParser(description="Build and check the submission zip.")
    parser.add_argument("ids", nargs="*", default=["2023B4A70727H"], help="student IDs")
    parser.add_argument("--python", help="interpreter command (default: this Python)")
    args = parser.parse_args()
    python = shlex.split(args.python) if args.python else [sys.executable]

    problems = check_pdf()
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    failed = bool(problems)
    for student_id in args.ids:
        path = build(student_id)
        print(f"{path.relative_to(ROOT)}")
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as zf:
            assert sorted(zf.namelist()) == sorted(FILES)
            zf.extractall(tmp)
            problems = check_contents(Path(tmp)) + check_runs(Path(tmp), python)
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        failed |= bool(problems)
    if failed:
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
