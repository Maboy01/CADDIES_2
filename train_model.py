#!/usr/bin/env python3
"""
Small launcher for the separated training scripts.

Default:
    python train_model.py

Runs the CNN+LSTM video model. Use --csv-baseline to run the metadata-only
RandomForest baseline.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Caddies training launcher.")
    parser.add_argument(
        "--csv-baseline",
        action="store_true",
        help="Run the CSV-only RandomForest baseline instead of CNN+LSTM.",
    )
    args, remaining_args = parser.parse_known_args()

    target_script = "train_csv_model.py" if args.csv_baseline else "train_cnn_lstm.py"
    target_path = BASE_DIR / target_script
    sys.argv = [str(target_path), *remaining_args]
    runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
    main()
