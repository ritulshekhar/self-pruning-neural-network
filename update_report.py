#!/usr/bin/env python3
"""
update_report.py
================
Run this script AFTER the full training completes.
It reads experiment_log.json and automatically patches the results
tables in both report.md and README.md with real numbers,
then optionally commits and pushes to GitHub.

Usage
-----
  python3 update_report.py                  # update tables only
  python3 update_report.py --push           # update + git commit + push
"""
import json
import re
import subprocess
import sys
from pathlib import Path

RESULTS_DIR  = Path("results")
REPORT_PATH  = Path("report.md")
README_PATH  = Path("README.md")
LOG_PATH     = RESULTS_DIR / "experiment_log.json"


def load_results() -> list[dict]:
    if not LOG_PATH.exists():
        print(f"[ERROR] {LOG_PATH} not found. Has the full training run completed?")
        sys.exit(1)
    with open(LOG_PATH) as f:
        data = json.load(f)
    return data["runs"]


def format_results_table(runs: list[dict]) -> str:
    """Build a markdown results table from real numbers."""
    notes_map = {
        0: "Minimal pruning; near-baseline accuracy",
        1: "Best sparsity–accuracy trade-off",
        2: "Heavy pruning; accuracy drops noticeably",
    }
    header = (
        "| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Notes |\n"
        "|:----------:|:-----------------:|:------------------:|:------|\n"
    )
    rows = ""
    for i, run in enumerate(sorted(runs, key=lambda r: r["lam"])):
        lam     = run["lam"]
        acc     = run["accuracy"]
        spars   = run["sparsity"]
        note    = notes_map.get(i, "")
        rows += f"| `{lam:.0e}` | **{acc:.2f}** | **{spars:.2f}** | {note} |\n"
    return header + rows


def build_per_layer_table(runs: list[dict]) -> str:
    """Build a per-layer sparsity table for the best (lowest lam) run."""
    # use the medium lambda run for the per-layer table
    sorted_runs = sorted(runs, key=lambda r: r["lam"])
    mid_run = sorted_runs[min(1, len(sorted_runs) - 1)]
    per_layer = mid_run.get("per_layer", [])
    if not per_layer:
        return ""
    lam = mid_run["lam"]
    header = (
        f"\n### Per-Layer Sparsity (λ = `{lam:.0e}`)\n\n"
        "| Layer | Total Weights | Pruned | Sparsity (%) | Mean Gate |\n"
        "|-------|:-------------:|:------:|:------------:|:---------:|\n"
    )
    rows = ""
    for row in per_layer:
        rows += (
            f"| `{row['layer_name']}` "
            f"| {row['n_weights']:,} "
            f"| {row['n_pruned']:,} "
            f"| {row['sparsity']:.1f} "
            f"| {row['mean_gate']:.4f} |\n"
        )
    return header + rows


def patch_file(path: Path, old_pattern: str, new_content: str) -> bool:
    """Replace a block in a file matching old_pattern with new_content."""
    text = path.read_text()
    new_text, count = re.subn(old_pattern, new_content, text, flags=re.DOTALL)
    if count == 0:
        print(f"  [WARN] Pattern not found in {path}. Manual edit needed.")
        return False
    path.write_text(new_text)
    print(f"  [OK]  Updated {path}")
    return True


def update_report(runs: list[dict]) -> None:
    table = format_results_table(runs)
    per_layer = build_per_layer_table(runs)

    # -- report.md: replace the results table (lines between the header and next ---)
    report_pattern = (
        r"### Results Table\n\n"
        r"\| Lambda.*?(?=\n###|\n---)"
    )
    report_replacement = (
        "### Results Table\n\n"
        + table.rstrip()
        + (("\n" + per_layer) if per_layer else "")
        + "\n"
    )
    patch_file(REPORT_PATH, report_pattern, report_replacement)

    # -- README.md: replace the placeholder table
    readme_pattern = (
        r"\| Lambda \(λ\) \| Test Accuracy.*?(?=\n###|\n---)"
    )
    readme_replacement = table.rstrip() + "\n"
    patch_file(README_PATH, readme_pattern, readme_replacement)


def git_commit_and_push() -> None:
    files = [
        str(REPORT_PATH),
        str(README_PATH),
        str(RESULTS_DIR / "gate_distribution.png"),
        str(RESULTS_DIR / "training_curves.png"),
        str(LOG_PATH),
    ]
    print("\n  Staging files...")
    subprocess.run(["git", "add"] + files, check=True)
    subprocess.run(
        ["git", "commit", "-m", "results and plots from full training run"],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("\n  [OK] Pushed to GitHub!")


def main() -> None:
    push = "--push" in sys.argv

    print("Loading experiment log...")
    runs = load_results()
    print(f"  Found {len(runs)} run(s): λ = {[r['lam'] for r in runs]}")

    print("\nUpdating report.md and README.md...")
    update_report(runs)

    print("\n--- Results Summary ---")
    for r in sorted(runs, key=lambda r: r["lam"]):
        print(f"  λ={r['lam']:.0e}  acc={r['accuracy']:.2f}%  sparsity={r['sparsity']:.2f}%")

    if push:
        print("\nCommitting and pushing to GitHub...")
        git_commit_and_push()
    else:
        print("\n  [INFO] Run with --push to commit and push automatically.")
        print("  Or manually:")
        print("    git add results/ report.md README.md")
        print("    git commit -m 'results and plots from full training run'")
        print("    git push origin main")


if __name__ == "__main__":
    main()
