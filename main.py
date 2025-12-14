"""
main.py

Activate the whole pipeline:
- runs src/part_3.py (data build -> vectorize -> baseline -> ablation -> retrain)
- then runs src/part_5.py (EDA plots -> grid search -> RFECV -> scores)

Usage:
    python main.py
    python main.py --step part_3
    python main.py --step part_5
"""

from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


def run_part_3_and_part_5(project_root: Path, step: str = "all") -> None:
    src_dir = project_root / "src"
    part_3_path = src_dir / "part_3.py"
    part_5_path = src_dir / "part_5.py"

    if not part_3_path.exists():
        raise FileNotFoundError(f"Missing: {part_3_path}")
    if not part_5_path.exists():
        raise FileNotFoundError(f"Missing: {part_5_path}")

    # 1) Change CWD so that Path.cwd().parent inside part_3.py resolves correctly
    old_cwd = Path.cwd()
    os.chdir(src_dir)

    try:
        # 2) Run part_3 and capture its globals
        globals_after_part_3 = {}
        if step in ("all", "part_3"):
            print("\n========== Running PART 3 ==========\n")
            globals_after_part_3 = runpy.run_path(str(part_3_path))

        # 3) Run part_5 in the SAME namespace (so it can see X_train_array, vec, test_tree, etc.)
        if step in ("all", "part_5"):
            print("\n========== Running PART 5 ==========\n")

            # If user runs only part_5, we must first run part_3 to create required variables.
            if step == "part_5" and not globals_after_part_3:
                globals_after_part_3 = runpy.run_path(str(part_3_path))

            runpy.run_path(str(part_5_path), init_globals=globals_after_part_3)

        print("\nDone.\n")

    finally:
        os.chdir(old_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["all", "part_3", "part_5"],
        default="all",
        help="Which step to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    run_part_3_and_part_5(project_root, step=args.step)
