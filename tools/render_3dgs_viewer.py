"""Render a standard 3DGS PLY interactively with gsplat, Nerfview, and Viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nerfview
import numpy as np
from plyfile import PlyData
import torch
from gsplat.rendering import rasterization
import viser

SH_C0 = 0.28209479177387814


def load_3dgs_ply(
    path: Path,
    device: torch.device,
    scales_are_log: bool = True,
    opacity_is_logit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the core parameters used by the original 3DGS PLY format."""
    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing 3DGS fields: {sorted(missing)}")

    def column(name: str) -> torch.Tensor:
        array = np.array(vertices[name], dtype=np.float32, copy=True)
        return torch.from_numpy(array).to(device=device)

    means = torch.stack([column("x"), column("y"), column("z")], dim=-1)
    scales = torch.stack([column("scale_0"), column("scale_1"), column("scale_2")], dim=-1)
    if scales_are_log:
        scales = torch.exp(scales)

    quaternions = torch.stack(
        [column("rot_0"), column("rot_1"), column("rot_2"), column("rot_3")], dim=-1
    )
    quaternions = quaternions / torch.linalg.norm(
        quaternions, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    opacities = column("opacity")
    if opacity_is_logit:
        opacities = torch.sigmoid(opacities)

    dc = torch.stack([column("f_dc_0"), column("f_dc_1"), column("f_dc_2")], dim=-1)
    colors = torch.clamp(0.5 + SH_C0 * dc, 0.0, 1.0)
    return means, quaternions, scales, opacities, colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--scales-are-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exponentiate scale_* fields, as used by the original 3DGS format",
    )
    parser.add_argument(
        "--opacity-is-logit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply sigmoid to the opacity field",
    )
    args = parser.parse_args()

    if not args.ply.is_file():
        raise FileNotFoundError(f"3DGS PLY not found: {args.ply}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rasterization. Install a CUDA-enabled PyTorch build.")

    device = torch.device("cuda")
    means, quaternions, scales, opacities, colors = load_3dgs_ply(
        args.ply,
        device,
        scales_are_log=args.scales_are_log,
        opacity_is_logit=args.opacity_is_logit,
    )
    print(f"Loaded {len(means):,} Gaussians from {args.ply}")

    server = viser.ViserServer(port=args.port, verbose=False)

    def render(camera: nerfview.CameraState, tab: nerfview.RenderTabState) -> np.ndarray:
        if tab.preview_render:
            width, height = tab.render_width, tab.render_height
        else:
            width, height = tab.viewer_width, tab.viewer_height

        camera_to_world = torch.from_numpy(camera.c2w).to(device=device, dtype=torch.float32)
        world_to_camera = torch.linalg.inv(camera_to_world)[None]
        intrinsics = torch.from_numpy(camera.get_K([width, height])).to(
            device=device, dtype=torch.float32
        )[None]

        rgb, _alpha, _metadata = rasterization(
            means,
            quaternions,
            scales,
            opacities,
            colors,
            world_to_camera,
            intrinsics,
            width,
            height,
            render_mode="RGB",
            packed=True,
            rasterize_mode="classic",
        )
        return (
            torch.clamp(rgb[0], 0.0, 1.0)
            .mul(255)
            .to(torch.uint8)
            .detach()
            .cpu()
            .numpy()
        )

    _viewer = nerfview.Viewer(server=server, render_fn=render, mode="rendering")
    print(f"Viewer ready: http://127.0.0.1:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Viewer stopped")


if __name__ == "__main__":
    main()
