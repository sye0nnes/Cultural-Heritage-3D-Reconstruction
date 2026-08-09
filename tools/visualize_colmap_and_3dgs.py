"""Inspect COLMAP cameras, sparse points, and a 3DGS PLY in one Viser scene."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Optional

import imageio.v3 as iio
import numpy as np
from plyfile import PlyData
from tqdm.auto import tqdm
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

SH_C0 = 0.28209479177387814


def sample_indices(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    """Return at most ``limit`` unique indices without failing on small inputs."""
    size = min(count, max(1, limit))
    return rng.choice(count, size=size, replace=False)


def load_gs_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 3DGS PLY as XYZ positions and displayable RGB colors."""
    vertices = PlyData.read(str(ply_path))["vertex"].data
    names = set(vertices.dtype.names or ())

    required_xyz = {"x", "y", "z"}
    if not required_xyz.issubset(names):
        raise ValueError(f"{ply_path} is missing XYZ fields: {sorted(required_xyz - names)}")

    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        dc = np.stack(
            [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1
        ).astype(np.float32)
        rgb = (np.clip(0.5 + SH_C0 * dc, 0.0, 1.0) * 255).astype(np.uint8)
    elif {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(
            np.uint8
        )
    else:
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)
    return xyz, rgb


def validate_colmap_directory(path: Path) -> None:
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"COLMAP directory {path} is missing: {', '.join(missing)}")


def camera_focal_y(camera, image_height: int, image_width: int) -> float:
    if camera.model == "PINHOLE" and len(camera.params) >= 2:
        return float(camera.params[1])
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and len(camera.params) >= 1:
        return float(camera.params[0])
    return float(max(image_height, image_width))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap", type=Path, required=True, help="COLMAP sparse/0 directory")
    parser.add_argument("--gs-ply", type=Path, required=True, help="3DGS point_cloud.ply")
    parser.add_argument("--images", type=Path, help="Optional source-image directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--image-downsample", type=int, default=2)
    parser.add_argument("--max-colmap-points", type=int, default=50_000)
    parser.add_argument("--max-gs-points", type=int, default=200_000)
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reorient", action="store_true")
    args = parser.parse_args()

    validate_colmap_directory(args.colmap)
    if not args.gs_ply.is_file():
        raise FileNotFoundError(f"3DGS PLY not found: {args.gs_ply}")
    if args.image_downsample < 1:
        raise ValueError("--image-downsample must be at least 1")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    cameras = read_cameras_binary(args.colmap / "cameras.bin")
    images = read_images_binary(args.colmap / "images.bin")
    points3d = read_points3d_binary(args.colmap / "points3D.bin")
    if not points3d:
        raise ValueError("COLMAP sparse model contains no 3D points")
    if not images:
        raise ValueError("COLMAP sparse model contains no registered images")

    colmap_points = np.asarray([point.xyz for point in points3d.values()], dtype=np.float32)
    colmap_colors = np.asarray([point.rgb for point in points3d.values()], dtype=np.uint8)
    gs_points, gs_colors = load_gs_points(args.gs_ply)

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    if not args.no_reorient:
        average_up = (
            vtf.SO3(np.asarray([image.qvec for image in images.values()])).inverse()
            @ np.asarray([0.0, -1.0, 0.0])
        ).mean(axis=0)
        norm = np.linalg.norm(average_up)
        if norm > 1e-8:
            server.scene.set_up_direction(tuple((average_up / norm).tolist()))

    reset_up = server.gui.add_button("Reset up direction")

    @reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        if event.client is not None:
            event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.asarray(
                [0.0, -1.0, 0.0]
            )

    server.gui.add_markdown("## COLMAP sparse reconstruction")
    colmap_limit = server.gui.add_slider(
        "Maximum points",
        min=1,
        max=len(colmap_points),
        step=1,
        initial_value=min(len(colmap_points), args.max_colmap_points),
    )
    frame_limit = server.gui.add_slider(
        "Maximum camera frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), args.max_frames),
    )
    colmap_size = server.gui.add_slider(
        "COLMAP point size", min=0.001, max=0.1, step=0.001, initial_value=0.02
    )

    server.gui.add_markdown("## 3D Gaussian Splatting PLY")
    show_gs = server.gui.add_checkbox("Show 3DGS points", initial_value=True)
    gs_size = server.gui.add_slider(
        "3DGS point size", min=0.001, max=0.05, step=0.001, initial_value=0.01
    )
    gs_limit = server.gui.add_slider(
        "Maximum 3DGS points",
        min=1,
        max=len(gs_points),
        step=max(1, min(1000, len(gs_points))),
        initial_value=min(len(gs_points), args.max_gs_points),
    )

    colmap_idx = sample_indices(len(colmap_points), colmap_limit.value, rng)
    colmap_handle = server.scene.add_point_cloud(
        "/colmap/points",
        points=colmap_points[colmap_idx],
        colors=colmap_colors[colmap_idx],
        point_size=colmap_size.value,
    )
    gs_idx = sample_indices(len(gs_points), gs_limit.value, rng)
    gs_handle = server.scene.add_point_cloud(
        "/3dgs/points",
        points=gs_points[gs_idx],
        colors=gs_colors[gs_idx],
        point_size=gs_size.value,
    )

    frame_handles: list[viser.FrameHandle] = []

    def rebuild_frames() -> None:
        for handle in frame_handles:
            handle.remove()
        frame_handles.clear()

        image_ids = list(images)
        random.shuffle(image_ids)
        for image_id in tqdm(sorted(image_ids[: frame_limit.value]), desc="Adding camera frames"):
            image = images[image_id]
            camera = cameras[image.camera_id]
            world_from_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(image.qvec), image.tvec
            ).inverse()
            frame = server.scene.add_frame(
                f"/colmap/cameras/{image_id}",
                wxyz=world_from_camera.rotation().wxyz,
                position=world_from_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frame_handles.append(frame)

            height, width = int(camera.height), int(camera.width)
            focal_y = camera_focal_y(camera, height, width)
            texture: Optional[np.ndarray] = None
            if args.images is not None:
                image_path = args.images / image.name
                if image_path.is_file():
                    try:
                        texture = iio.imread(image_path)[
                            :: args.image_downsample, :: args.image_downsample
                        ]
                    except Exception as exc:
                        print(f"[WARN] Could not read {image_path}: {exc}")

            frustum = server.scene.add_camera_frustum(
                f"/colmap/cameras/{image_id}/frustum",
                fov=float(2 * np.arctan2(height / 2.0, focal_y)),
                aspect=float(width / height) if height else 1.0,
                scale=0.15,
                image=texture,
            )

            @frustum.on_click
            def _(_, selected_frame=frame) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = selected_frame.wxyz
                    client.camera.position = selected_frame.position

    rebuild_frames()

    @colmap_limit.on_update
    def _(_) -> None:
        idx = sample_indices(len(colmap_points), colmap_limit.value, rng)
        with server.atomic():
            colmap_handle.points = colmap_points[idx]
            colmap_handle.colors = colmap_colors[idx]

    @frame_limit.on_update
    def _(_) -> None:
        rebuild_frames()

    @colmap_size.on_update
    def _(_) -> None:
        colmap_handle.point_size = colmap_size.value

    @show_gs.on_update
    def _(_) -> None:
        gs_handle.visible = bool(show_gs.value)

    @gs_size.on_update
    def _(_) -> None:
        gs_handle.point_size = gs_size.value

    @gs_limit.on_update
    def _(_) -> None:
        idx = sample_indices(len(gs_points), gs_limit.value, rng)
        with server.atomic():
            gs_handle.points = gs_points[idx]
            gs_handle.colors = gs_colors[idx]

    print(f"Viewer ready: http://127.0.0.1:{args.port}")
    try:
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Viewer stopped")


if __name__ == "__main__":
    main()
