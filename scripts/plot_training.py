"""Parse the training log and print an ASCII loss curve.

Usage:
    python scripts/plot_training.py [--log /tmp/train_long.log]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_TRAIN_RE = re.compile(r"step\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)")
_VAL_RE = re.compile(r">>\s+val_loss=([\d.]+)")


def parse(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    last_step = 0
    for line in path.read_text(errors="replace").splitlines():
        m = _TRAIN_RE.search(line)
        if m:
            last_step = int(m.group(1))
            train.append((last_step, float(m.group(2))))
            continue
        m = _VAL_RE.search(line)
        if m:
            val.append((last_step, float(m.group(1))))
    return train, val


def ascii_plot(points: list[tuple[int, float]], *, height: int = 12, width: int = 60) -> str:
    if not points:
        return "(no data)"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        y_max = y_min + 1
    grid = [[" "] * width for _ in range(height)]
    for x, y in points:
        col = int((x - x_min) / max(1, x_max - x_min) * (width - 1))
        row = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = "*"
    lines: list[str] = []
    for r, row in enumerate(grid):
        # Y axis label every other row
        y_at_row = y_max - (r / (height - 1)) * (y_max - y_min)
        label = f"{y_at_row:5.2f} |"
        lines.append(label + "".join(row))
    lines.append(" " * 7 + "+" + "-" * width)
    lines.append(" " * 8 + f"{x_min}" + " " * (width - len(str(x_min)) - len(str(x_max))) + f"{x_max}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=Path("/tmp/train_long.log"))
    args = p.parse_args()

    train, val = parse(args.log)
    if not train:
        print("No training points found in log.")
        return

    print(f"Training points: {len(train)}, val points: {len(val)}")
    print()
    print("=== Train loss ===")
    print(ascii_plot(train))
    if val:
        print()
        print("=== Val loss ===")
        print(ascii_plot(val))

    print()
    print(f"latest train loss: {train[-1][1]:.3f} (step {train[-1][0]})")
    if val:
        print(f"latest val loss:   {val[-1][1]:.3f} (step {val[-1][0]})")
        print(f"best val loss:     {min(v for _, v in val):.3f}")


if __name__ == "__main__":
    main()
