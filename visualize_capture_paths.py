"""Visualize drone and quadruped-robot capture trajectories from CSV metadata."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_xy(path: Path, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {x_key, y_key}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        for row in reader:
            try:
                x = float(row[x_key])
                y = float(row[y_key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)

    if not xs:
        raise ValueError(f"{path}: no valid coordinate rows")
    return xs, ys


def svg_panel(
    xs: list[float],
    ys: list[float],
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    labels: tuple[str, str],
) -> str:
    pad = 55.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)
    scale = min(plot_w / dx, plot_h / dy)
    used_w, used_h = dx * scale, dy * scale
    left = x0 + pad + (plot_w - used_w) / 2
    top = y0 + pad + (plot_h - used_h) / 2

    points = [
        (left + (x - min_x) * scale, top + used_h - (y - min_y) * scale)
        for x, y in zip(xs, ys)
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="#0f172a" opacity="0.65"/>'
        for x, y in points
    )
    start_x, start_y = points[0]
    end_x, end_y = points[-1]

    return f"""
    <g>
      <rect x="{x0}" y="{y0}" width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#cbd5e1"/>
      <text x="{x0 + width / 2}" y="{y0 + 30}" text-anchor="middle" font-size="18" font-weight="700">{title} ({len(xs)} captures)</text>
      <rect x="{left:.2f}" y="{top:.2f}" width="{used_w:.2f}" height="{used_h:.2f}" fill="#f8fafc" stroke="#cbd5e1"/>
      <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>
      {dots}
      <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="6" fill="#16a34a"/>
      <circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="6" fill="#dc2626"/>
      <text x="{x0 + width / 2}" y="{y0 + height - 14}" text-anchor="middle" font-size="13">{labels[0]}</text>
      <text x="{x0 + 16}" y="{y0 + height / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 {x0 + 16} {y0 + height / 2})">{labels[1]}</text>
    </g>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", type=Path, default=Path("poses(1IPS).csv"))
    parser.add_argument("--waypoints", type=Path, default=Path("waypoint_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("capture_paths.svg"))
    args = parser.parse_args()

    drone_x, drone_y = read_xy(args.drone, "x_north_m", "y_east_m")
    robot_x, robot_y = read_xy(args.waypoints, "x", "y")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">
  <rect width="1200" height="560" fill="#f1f5f9"/>
  <text x="600" y="38" text-anchor="middle" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#0f172a">Image Capture Paths</text>
  <g font-family="Arial, sans-serif" fill="#0f172a">
    {svg_panel(drone_x, drone_y, 25, 60, 560, 470, "Drone trajectory", ("North (m)", "East (m)"))}
    {svg_panel(robot_x, robot_y, 615, 60, 560, 470, "Quadruped waypoints", ("Local x", "Local y"))}
    <circle cx="500" cy="548" r="5" fill="#16a34a"/><text x="512" y="552" font-size="12">Start</text>
    <circle cx="565" cy="548" r="5" fill="#dc2626"/><text x="577" y="552" font-size="12">End</text>
  </g>
</svg>"""
    args.output.write_text(svg, encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
