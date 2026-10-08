from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_ascii_ply(path: str | Path) -> NDArray[np.float64]:
    source = Path(path)
    with source.open("r", encoding="ascii") as ply_file:
        vertex_count: int | None = None
        properties: list[str] = []
        header_lines = 0
        in_vertex_element = False
        for line in ply_file:
            header_lines += 1
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "format" and fields[1] != "ascii":
                raise ValueError("Only ASCII PLY files are supported.")
            if fields[0] == "element":
                in_vertex_element = fields[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(fields[2])
            elif fields[0] == "property" and in_vertex_element:
                if fields[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported.")
                properties.append(fields[2])
            elif fields[0] == "end_header":
                break
        else:
            raise ValueError(f"PLY header is incomplete: {source}")

    if vertex_count is None:
        raise ValueError(f"PLY file has no vertex element: {source}")
    try:
        xyz_columns = [properties.index(axis) for axis in ("x", "y", "z")]
    except ValueError as exc:
        raise ValueError(f"PLY file must have x, y, and z vertex properties: {source}") from exc
    if vertex_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.loadtxt(
        source,
        dtype=np.float64,
        skiprows=header_lines,
        max_rows=vertex_count,
        usecols=xyz_columns,
        ndmin=2,
    )
    if points.shape != (vertex_count, 3):
        raise ValueError(
            f"Expected {vertex_count} 3D vertices in {source}, read {points.shape[0]}."
        )
    if not np.isfinite(points).all():
        raise ValueError(f"PLY file contains non-finite vertex coordinates: {source}")
    return points


def show_point_cloud(points: NDArray[np.float64]) -> None:
    if points.size == 0:
        raise ValueError("PLY file contains no points to display.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required to display the point cloud. Install it with: "
            ".\\.venv\\Scripts\\python.exe -m pip install matplotlib"
        ) from exc

    figure = plt.figure("SLAM point cloud")
    axes = figure.add_subplot(111, projection="3d")
    axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c=points[:, 2], cmap="viridis")
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    axes.set_title(f"Initial map ({len(points)} points)")

    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2, 1e-3)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display an ASCII PLY point cloud with Matplotlib.")
    parser.add_argument("ply_file", nargs="?", default="initial_map.ply")
    args = parser.parse_args()
    points = load_ascii_ply(args.ply_file)
    print(f"Loaded {len(points)} points from {args.ply_file}")
    show_point_cloud(points)


if __name__ == "__main__":
    main()
