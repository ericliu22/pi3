#!/usr/bin/env python3
import os
import math
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor

# --- gsplat imports (adjust to your installed version) ---
from gsplat import rasterization as gs_rast   # <-- if your version uses a different module, update here
from gsplat.io import save_ply                # <-- adjust signature if your version differs

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- Pi3 runner -------------------------

def run_pi3(input_dir: str, interval: int = 10) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Runs Pi3 over an image folder; returns (results, imgs). imgs is (N,3,H,W) in [0,1]."""
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    imgs = load_images_as_tensor(input_dir, interval=interval).to(device)
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"Expected imgs (N,3,H,W); got {tuple(imgs.shape)}")

    # Mixed precision when available
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                results = model(imgs[None])    # Pi3 expects (1, N, 3, H, W) or similar
        else:
            results = model(imgs[None])

    return results, imgs


# ------------------------- Camera helpers -------------------------

def parse_intrinsics(intr: Any, N: int, W: int, H: int) -> torch.Tensor:
    """
    Returns intrinsics as (N,4) tensor (fx, fy, cx, cy).
    Accepts Pi3 dict/array/tensor; falls back to 60° FOV if missing.
    """
    if intr is None:
        # Fallback from FOV 60°
        f = 0.5 * W / math.tan(0.5 * math.radians(60.0))
        fx = fy = f
        cx, cy = W / 2.0, H / 2.0
        arr = np.tile([fx, fy, cx, cy], (N, 1)).astype(np.float32)
        return torch.tensor(arr, device=device)

    if isinstance(intr, torch.Tensor):
        intr = intr.detach().to(device)
        if intr.ndim == 2 and intr.shape[1] >= 4:
            return intr[:, :4].contiguous().float()
        raise ValueError("Unrecognized intrinsics tensor shape from Pi3.")
    elif isinstance(intr, dict):
        fx = torch.as_tensor(intr["fx"], dtype=torch.float32, device=device)
        fy = torch.as_tensor(intr["fy"], dtype=torch.float32, device=device)
        cx = torch.as_tensor(intr["cx"], dtype=torch.float32, device=device)
        cy = torch.as_tensor(intr["cy"], dtype=torch.float32, device=device)
        return torch.stack([fx, fy, cx, cy], dim=-1)
    else:
        arr = np.asarray(intr)
        if arr.ndim == 2 and arr.shape[1] >= 4:
            return torch.tensor(arr[:, :4], dtype=torch.float32, device=device)
        raise ValueError("Unrecognized intrinsics format from Pi3.")


def parse_poses(poses: Any, N: int) -> torch.Tensor:
    """
    Returns (N,4,4) c2w poses as float32 tensor.
    """
    if isinstance(poses, torch.Tensor):
        P = poses.squeeze(0) if poses.ndim == 4 else poses
        P = P.to(device).float()
    else:
        P = torch.tensor(np.asarray(poses), dtype=torch.float32, device=device)
    if P.shape != (N, 4, 4):
        raise ValueError(f"Unexpected pose shape: {tuple(P.shape)}; expected ({N},4,4)")
    return P


# ------------------------- Depth -> World point cloud -------------------------

def backproject_depth_to_world(
    depth: torch.Tensor,        # (N,H,W) or (N,1,H,W), depth in meters (or scaled)
    intr44: torch.Tensor,       # (N,4) [fx,fy,cx,cy]
    T_c2w: torch.Tensor,        # (N,4,4)
    depth_scale: float = 1.0,
    valid_min: float = 1e-6,
    valid_max: float = 1e6,
    stride: int = 4,
    max_points: int = 200_000,
    mask: Optional[torch.Tensor] = None,   # (N,H,W) boolean mask (optional)
) -> torch.Tensor:
    """
    Backprojects per-pixel depths into world space and returns (M,3) fused point cloud.
    """
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]  # (N,H,W)

    N, H, W = depth.shape
    fx, fy, cx, cy = intr44[:, 0], intr44[:, 1], intr44[:, 2], intr44[:, 3]

    # make pixel grid (subsample by stride for speed)
    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, W, stride, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (h', w')

    # accumulate points across frames
    all_points = []

    for i in range(N):
        d = depth[i][grid_y, grid_x] * depth_scale  # (h', w')
        if mask is not None:
            m = mask[i][grid_y, grid_x]
        else:
            m = torch.isfinite(d) & (d > valid_min) & (d < valid_max)

        if not torch.any(m):
            continue

        u = grid_x[m].float()
        v = grid_y[m].float()
        di = d[m].float()

        x = (u - cx[i]) / fx[i] * di
        y = (v - cy[i]) / fy[i] * di
        z = di

        # camera -> world
        ones = torch.ones_like(z)
        Pc = torch.stack([x, y, z, ones], dim=-1)  # (K,4)
        Pw_h = (T_c2w[i] @ Pc.t()).t()             # (K,4)
        Pw = Pw_h[:, :3] / Pw_h[:, 3:4]           # (K,3)
        all_points.append(Pw)

    if not all_points:
        raise RuntimeError("No valid depth pixels to backproject (check masks / depth scale).")

    fused = torch.cat(all_points, dim=0)  # (M,3)

    # optional random downsample to cap memory
    if fused.shape[0] > max_points:
        idx = torch.randperm(fused.shape[0], device=device)[:max_points]
        fused = fused[idx]

    return fused


# ------------------------- Gaussian model -------------------------

def init_gaussians_from_points(xyz_world: torch.Tensor):
    """
    xyz_world: (M,3) on device
    Returns learnable params: means, log_scales, quats, colors, logit_opacity
    """
    M = xyz_world.shape[0]
    means = nn.Parameter(xyz_world.detach().clone())                      # (M,3)
    log_scales = nn.Parameter(torch.full((M, 3), -4.0, device=device))    # ~1.8cm if units in meters
    quats = nn.Parameter(torch.tensor([[1, 0, 0, 0]], device=device).repeat(M, 1).float())
    colors = nn.Parameter(torch.full((M, 3), 0.5, device=device))         # start gray
    logit_opacity = nn.Parameter(torch.full((M, 1), -1.5, device=device)) # ~0.18
    return means, log_scales, quats, colors, logit_opacity


def render_views(means, log_scales, quats, colors, logit_opacity, cams, bg=1.0):
    """
    cams: list of dicts with T_w2c (4x4), fx,fy,cx,cy, h,w
    Returns (N,3,H,W)
    """
    scales = torch.exp(log_scales)
    opac   = torch.sigmoid(logit_opacity)
    renders = []
    for c in cams:
        img = gs_rast.render(   # <-- adjust if your gsplat uses a different function/signature
            means=means, scales=scales, quats=quats,
            colors=colors, opacities=opac,
            T_w2c=c["T_w2c"], fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
            height=c["h"], width=c["w"], background=bg,
        )  # (3,H,W), float in [0,1]
        renders.append(img)
    return torch.stack(renders, dim=0)


# ------------------------- Training loop -------------------------

def train_gsplat_from_pi3(
    input_dir: str,
    interval: int = 10,
    iters: int = 4000,
    lr: float = 1e-2,
    out_ply: str = "gaussians.ply",
    depth_scale: float = 1.0,
    depth_stride: int = 4,
    max_init_points: int = 200_000,
    valid_min: float = 1e-6,
    valid_max: float = 1e6,
):
    results, imgs = run_pi3(input_dir, interval=interval)  # imgs: (N,3,H,W) in [0,1]
    N, _, H, W = imgs.shape

    # --- Poses (c2w) & intrinsics ---
    poses_c2w = parse_poses(results["camera_poses"], N)           # (N,4,4)
    intr = parse_intrinsics(results.get("intrinsics", None), N, W, H)  # (N,4)
    fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]

    # Build cam structs with T_w2c for rendering
    cams = []
    for i in range(N):
        T_c2w = poses_c2w[i]
        T_w2c = torch.linalg.inv(T_c2w)
        cams.append({
            "T_w2c": T_w2c,
            "fx": fx[i].item(), "fy": fy[i].item(),
            "cx": cx[i].item(), "cy": cy[i].item(),
            "h": H, "w": W
        })

    # --- Seed points: prefer depth, else points, else uniform box ---
    init_xyz: Optional[torch.Tensor] = None

    # 1) Depth maps
    if "depth" in results and results["depth"] is not None:
        depth = results["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.to(device)
        else:
            depth = torch.tensor(np.asarray(depth), dtype=torch.float32, device=device)
        # Expect (N,H,W) or (N,1,H,W). If Pi3 returns (H,W) per-frame, add batch dim yourself.
        # Optional masks (e.g., results["mask"] with 1 for valid)
        mask = None
        if "mask" in results and results["mask"] is not None:
            m = results["mask"]
            mask = m.to(device) if isinstance(m, torch.Tensor) else torch.tensor(np.asarray(m), device=device)
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask[:, 0]
            if mask.ndim == 3 and mask.dtype != torch.bool:
                mask = mask > 0.5

        init_xyz = backproject_depth_to_world(
            depth=depth, intr44=intr, T_c2w=poses_c2w,
            depth_scale=depth_scale, valid_min=valid_min, valid_max=valid_max,
            stride=depth_stride, max_points=max_init_points, mask=mask
        )

    # 2) Per-pixel 3D points in camera frame (rare but supported by some models)
    if init_xyz is None and "points" in results and results["points"] is not None:
        pts = results["points"]
        if isinstance(pts, torch.Tensor):
            pts = pts.to(device)
        else:
            pts = torch.tensor(np.asarray(pts), dtype=torch.float32, device=device)
        # Accept (N,H,W,3) in camera space
        if pts.ndim == 4 and pts.shape[-1] == 3:
            all_pw = []
            for i in range(N):
                Pw_h = (poses_c2w[i] @ torch.cat([pts[i].reshape(-1, 3), torch.ones(pts[i].shape[0]*pts[i].shape[1], 1, device=device)], dim=1).t()).t()
                Pw = Pw_h[:, :3] / Pw_h[:, 3:4]
                all_pw.append(Pw)
            init_xyz = torch.cat(all_pw, dim=0)
        else:
            raise ValueError("Unsupported results['points'] shape; expected (N,H,W,3) in camera frame.")

        if init_xyz.shape[0] > max_init_points:
            idx = torch.randperm(init_xyz.shape[0], device=device)[:max_init_points]
            init_xyz = init_xyz[idx]

    # 3) Fallback: uniform cube around camera centers
    if init_xyz is None:
        centers = poses_c2w[:, :3, 3]
        lo = centers.min(dim=0).values - 0.5
        hi = centers.max(dim=0).values + 0.5
        M = max_init_points
        init_xyz = torch.rand((M, 3), device=device) * (hi - lo) + lo

    # --- Build Gaussian params ---
    means, log_scales, quats, colors, logit_opacity = init_gaussians_from_points(init_xyz)

    # --- Optimize ---
    params = [means, log_scales, quats, colors, logit_opacity]
    optim = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.L1Loss()

    print(f"Init gaussians: {means.shape[0]:,} points; training for {iters} iters...")
    for step in range(1, iters + 1):
        # You can subsample views for speed; here we use all
        pred = render_views(means, log_scales, quats, colors, logit_opacity, cams, bg=1.0)
        loss = loss_fn(pred, imgs)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 100 == 0:
            print(f"[{step}/{iters}] L1: {loss.item():.4f}")

    # --- Save .ply ---
    with torch.no_grad():
        xyz = means.detach().cpu()
        sc  = torch.exp(log_scales).detach().cpu()
        qt  = quats.detach().cpu()
        col = (colors.clamp(0, 1) * 255).to(torch.uint8).detach().cpu()
        al  = torch.sigmoid(logit_opacity).detach().cpu()

    os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
    save_ply(out_ply, xyz, sc, qt, col, al)  # adjust to your gsplat.io.save_ply signature if needed
    print(f"✅ wrote {out_ply}")


# ------------------------- CLI -------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="End-to-end: Pi3 -> depth backprojection -> gsplat training (no file exports)")
    ap.add_argument("--input-dir", required=True, help="Folder with images for Pi3.")
    ap.add_argument("--interval", type=int, default=10, help="Sample every k-th frame for Pi3.")
    ap.add_argument("--iters", type=int, default=4000, help="Optimization iterations.")
    ap.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    ap.add_argument("--out-ply", default="gaussians.ply", help="Output 3DGS .ply path.")
    ap.add_argument("--depth-scale", type=float, default=1.0, help="Multiply depths by this (if Pi3 depth units differ).")
    ap.add_argument("--depth-stride", type=int, default=4, help="Subsample factor for depth backprojection.")
    ap.add_argument("--max-init-points", type=int, default=200000, help="Cap on initial point cloud.")
    ap.add_argument("--valid-min", type=float, default=1e-6, help="Min valid depth (after scaling).")
    ap.add_argument("--valid-max", type=float, default=1e6, help="Max valid depth (after scaling).")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_gsplat_from_pi3(
        input_dir=args.input_dir,
        interval=args.interval,
        iters=args.iters,
        lr=args.lr,
        out_ply=args.out_ply,
        depth_scale=args.depth_scale,
        depth_stride=args.depth_stride,
        max_init_points=args.max_init_points,
        valid_min=args.valid_min,
        valid_max=args.valid_max,
    )

