import argparse
import csv
from pathlib import Path


def best_value(csv_path: Path, metric: str):
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(float(row[metric]) for row in rows if row.get(metric))


def main():
    parser = argparse.ArgumentParser(description="Summarize task CSV metrics")
    parser.add_argument("--task1-dir", type=Path, default=Path("runs/task1"))
    parser.add_argument("--task3-dir", type=Path, default=Path("runs/task3"))
    args = parser.parse_args()

    print("Task 1 classification:")
    for csv_path in sorted(args.task1_dir.glob("*.csv")):
        print(f"  {csv_path.name}: best_val_acc={best_value(csv_path, 'best_val_acc'):.4f}")

    print("Task 3 segmentation:")
    for csv_path in sorted(args.task3_dir.glob("*.csv")):
        print(f"  {csv_path.name}: best_val_miou={best_value(csv_path, 'best_val_miou'):.4f}")


if __name__ == "__main__":
    main()
