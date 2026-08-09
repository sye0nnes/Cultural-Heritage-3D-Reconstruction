# 3D Reconstruction Utilities

These public utilities support inspection and rendering of Gaussian Splatting outputs. They use command-line paths and do not contain private datasets, checkpoints, or machine-specific directories.

## 1. COLMAP and 3DGS inspection

`visualize_colmap_and_3dgs.py` displays the following in one interactive Viser scene:

- COLMAP sparse points
- registered camera poses and frustums
- optional source-image thumbnails
- a colorized 3DGS `point_cloud.ply`

```bash
python tools/visualize_colmap_and_3dgs.py \
  --colmap path/to/sparse/0 \
  --images path/to/images \
  --gs-ply path/to/point_cloud.ply \
  --port 8080
```

The `--images` argument is optional. Point and camera limits can be adjusted from the web GUI or command line.

## 2. CUDA 3DGS renderer

`render_3dgs_viewer.py` loads the standard 3DGS PLY fields and rasterizes the Gaussians using `gsplat`.

```bash
python tools/render_3dgs_viewer.py \
  --ply path/to/point_cloud.ply \
  --port 8080
```

This renderer requires:

- an NVIDIA GPU
- a CUDA-enabled PyTorch installation
- a compatible `gsplat` installation
- `nerfview` and `viser`

By default, the script interprets `scale_*` as log-scales and `opacity` as logits, matching the original 3DGS PLY convention. Use `--no-scales-are-log` or `--no-opacity-is-logit` for a different export convention.

## Installation

Install the lightweight viewer dependencies:

```bash
pip install -r requirements-viewers.txt
```

Install CUDA-enabled PyTorch and `gsplat` separately using their official instructions so the versions match the local CUDA toolkit.

## Scope

The utilities have been syntax-checked and cleaned for public release. Full end-to-end execution requires user-supplied COLMAP and 3DGS files; those private project assets are not included in this repository.
