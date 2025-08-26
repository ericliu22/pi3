#!/usr/bin/env python3
import os
import json
import math
import glob
import argparse
import shutil
from pathlib import Path
from typing import Optional, Sequence, List, Literal

import numpy as np
import torch

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor  # your helper that returns (N,3,H,W) in [0,1]


# --------------------------- Helpers ---------------------------

def list_image_files(img_dir: str) -> Sequence[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
    files: List[str] = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(img_dir, e))))
    if not files:
        raise FileNotFoundError(f"No images found under {img_dir}")
    return files


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz).
    COLMAP expects world->camera rotation as a quaternion with positive scalar part.
    """
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[3] < 0:
        q = -q
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def estimate_intrinsics(width: int, height: int, fov_deg: float = 60.0):
    """Fallback intrinsics if Pi3 intrinsics are unavailable."""
    f = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
    cx, cy = width / 2.0, height / 2.0
    return f, f, cx, cy


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    out = np.eye(4)
    out[:3, :3] = Rinv
    out[:3, 3] = tinv
    return out


def write_colmap_text(
    out_dir: str,
    image_files: Sequence[str],
    poses_c2w: np.ndarray,
    width: int,
    height: int,
    fx: np.ndarray,
    fy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    poses_are_c2w: bool,
    use_shared_camera: bool,
    points_xyz: Optional[np.ndarray] = None,
):
    """
    Write COLMAP text model: cameras.txt, images.txt, points3D.txt
    COLMAP stores world->camera in images.txt; we invert if we start from c2w.
    """
    ensure_dir(out_dir)
    cameras_txt = os.path.join(out_dir, "cameras.txt")
    images_txt  = os.path.join(out_dir, "images.txt")
    points3d_txt = os.path.join(out_dir, "points3D.txt")

    N = len(image_files)

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        if use_shared_camera:
            f.write("# Number of cameras: 1\n")
            f.write(f"1 PINHOLE {width} {height} {fx[0]} {fy[0]} {cx[0]} {cy[0]}\n")
        else:
            f.write(f"# Number of cameras: {N}\n")
            for i in range(N):
                f.write(f"{i+1} PINHOLE {width} {height} {fx[i]} {fy[i]} {cx[i]} {cy[i]}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img_path in enumerate(image_files):
            T = poses_c2w[i]
            T_cw = inverse_se3(T) if poses_are_c2w else T  # world->camera
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            q = rotation_to_quaternion(R_cw)
            cam_id = 1 if use_shared_camera else (i + 1)
            name = os.path.basename(img_path)
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t_cw[0]} {t_cw[1]} {t_cw[2]} {cam_id} {name}\n")
            f.write("\n")

    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        if points_xyz is not None and len(points_xyz) > 0:
            for j, p in enumerate(points_xyz, start=1):
                f.write(f"{j} {p[0]} {p[1]} {p[2]} 128 128 128 0\n")


def materialize_images(
    src_files: Sequence[str],
    dest_dir: str,
    link_mode: Literal["copy", "hardlink", "symlink"] = "copy",
) -> Sequence[str]:
    """
    Ensure the sampled frames exist under dest_dir and return the *destination paths*.
    """
    ensure_dir(dest_dir)
    dst_files: List[str] = []
    for i, src in enumerate(src_files):
        base = os.path.basename(src)
        dst = os.path.join(dest_dir, base)
        if not os.path.exists(dst):
            try:
                if link_mode == "copy":
                    shutil.copy2(src, dst)
                elif link_mode == "hardlink":
                    os.link(src, dst)
                elif link_mode == "symlink":
                    # Use relative symlink for portability inside the scene folder
                    rel_src = os.path.relpath(src, start=dest_dir)
                    os.symlink(rel_src, dst)
                else:
                    raise ValueError(f"Unknown link_mode: {link_mode}")
            except Exception as e:
                # Fallback to copy on any failure (e.g., cross-device hardlink)
                shutil.copy2(src, dst)
        dst_files.append(dst)
    return dst_files


def write_nerfstudio_transforms(
    out_dir: str,
    image_files: Sequence[str],
    poses_c2w: np.ndarray,
    width: int,
    height: int,
    fx: np.ndarray,
    fy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    use_shared_camera: bool,
    pack_images: bool = False,
    link_mode: Literal["copy", "hardlink", "symlink"] = "copy",
):
    """
    Write Nerfstudio transforms.json with camera-to-world 4x4 per frame.
    If pack_images=True, copy/link sampled frames into out_dir/images/ and point frames there.
    """
    ensure_dir(out_dir)
    if pack_images:
        images_dir = os.path.join(out_dir, "images")
        packed_paths = materialize_images(image_files, images_dir, link_mode=link_mode)
        rel_paths = [os.path.relpath(p, start=out_dir) for p in packed_paths]  # e.g., images/xxx.jpg
    else:
        # Keep original locations, but still make them relative to transforms.json parent if possible
        rel_paths = [os.path.relpath(p, start=out_dir) for p in image_files]

    meta = {
        "w": int(width),
        "h": int(height),
        "camera_model": "PINHOLE",
        "frames": []
    }
    if use_shared_camera:
        meta.update(dict(fl_x=float(fx[0]), fl_y=float(fy[0]), cx=float(cx[0]), cy=float(cy[0])))

    frames = []
    for i, rel in enumerate(rel_paths):
        T = poses_c2w[i].astype(np.float64)
        entry = {
            "file_path": rel.replace("\\", "/"),  # sanitize for cross-platform use
            "transform_matrix": T.tolist(),
        }
        if not use_shared_camera:
            entry.update({
                "fl_x": float(fx[i]),
                "fl_y": float(fy[i]),
                "cx": float(cx[i]),
                "cy": float(cy[i]),
            })
        frames.append(entry)

    meta["frames"] = frames
    with open(os.path.join(out_dir, "transforms.json"), "w") as f:
        json.dump(meta, f, indent=2)

def get_image_size(path: str):
    """Return (width, height) without adding heavy deps."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        try:
            import imageio.v3 as iio
            im = iio.imread(path)
            return int(im.shape[1]), int(im.shape[0])
        except Exception:
            import cv2
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError(f"Failed to read image size for {path}")
            return int(im.shape[1]), int(im.shape[0])


# --------------------------- Core pipeline ---------------------------

def run_pi3_and_export(
    input_dir: str,
    output_dir: str,
    poses_are_c2w: bool = True,
    use_shared_camera: bool = True,
    interval: int = 10,
    fov_deg: float = 60.0,
    pack_nerfstudio: bool = False,
    link_mode: Literal["copy", "hardlink", "symlink"] = "copy",
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # Load images (N,3,H,W) in [0,1]
    imgs = load_images_as_tensor(input_dir, interval=interval).to(device)
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"Expected imgs of shape (N,3,H,W); got {tuple(imgs.shape)}")

    N, _, H, W = imgs.shape
    if N < 2:
        raise ValueError("At least 2 images are recommended for pose estimation.")

    # Inference
    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    with torch.no_grad():
        if device == 'cuda':
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                results = model(imgs[None])
        else:
            results = model(imgs[None])

    # Extract outputs
    poses = results["camera_poses"]
    if isinstance(poses, torch.Tensor):
        poses = poses.squeeze(0) if poses.ndim == 4 else poses
        poses = poses.detach().cpu().numpy()
    if poses.shape != (N, 4, 4):
        raise ValueError(f"Unexpected pose shape: {poses.shape}; expected ({N},4,4)")

    points_xyz = None
    if "points" in results and results["points"] is not None:
        pts = results["points"]
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        points_xyz = np.asarray(pts)

    # Intrinsics (prefer Pi3 if present; else estimate from FOV)
    intr = results.get("intrinsics", None)
    if intr is not None:
        if isinstance(intr, torch.Tensor):
            intr = intr.detach().cpu().numpy()
        intr = np.asarray(intr)
        if intr.ndim == 2 and intr.shape[1] >= 4:
            fx = intr[:, 0].astype(np.float64)
            fy = intr[:, 1].astype(np.float64)
            cx = intr[:, 2].astype(np.float64)
            cy = intr[:, 3].astype(np.float64)
        elif isinstance(intr, dict):
            fx = np.asarray(intr["fx"], dtype=np.float64)
            fy = np.asarray(intr["fy"], dtype=np.float64)
            cx = np.asarray(intr["cx"], dtype=np.float64)
            cy = np.asarray(intr["cy"], dtype=np.float64)
        else:
            raise ValueError("Unrecognized intrinsics format from Pi3.")
    else:
        fxe, fye, cxe, cye = estimate_intrinsics(W, H, fov_deg=fov_deg)
        fx = np.full((N,), fxe, dtype=np.float64)
        fy = np.full((N,), fye, dtype=np.float64)
        cx = np.full((N,), cxe, dtype=np.float64)
        cy = np.full((N,), cye, dtype=np.float64)

    # Output layout
    colmap_dir = os.path.join(output_dir, "colmap", "sparse_text")
    ns_dir = os.path.join(output_dir, "nerfstudio")
    ensure_dir(colmap_dir)
    ensure_dir(ns_dir)

    # List original files for naming
    image_files = list_image_files(input_dir)
    # Keep only the sampled files if load_images_as_tensor applied interval
    if interval > 1:
        image_files = image_files[::interval]
    if len(image_files) != N:
        image_files = image_files[:N]

    # --- Align intrinsics to the actual file sizes we will feed Nerfstudio ---
    # Pi3 intrinsics are for the tensor size (W,H). If we pack originals, scale intrinsics.
    # Assume all sampled frames have the same original size; otherwise scale per-frame.
    orig_sizes = [get_image_size(p) for p in image_files]  # list of (origW, origH)
    all_same = all(s == orig_sizes[0] for s in orig_sizes)
    if not all_same:
        # Per-frame scaling (rare for video, but handle it).
        fx_scaled = fx.copy()
        fy_scaled = fy.copy()
        cx_scaled = cx.copy()
        cy_scaled = cy.copy()
        widths = np.zeros(len(image_files), dtype=np.int64)
        heights = np.zeros(len(image_files), dtype=np.int64)
        for i, (ow, oh) in enumerate(orig_sizes):
            sx = ow / float(W)
            sy = oh / float(H)
            fx_scaled[i] *= sx
            fy_scaled[i] *= sy
            cx_scaled[i] *= sx
            cy_scaled[i] *= sy
            widths[i] = ow
            heights[i] = oh
    else:
        ow, oh = orig_sizes[0]
        sx = ow / float(W)
        sy = oh / float(H)
        fx_scaled = fx * sx
        fy_scaled = fy * sy
        cx_scaled = cx * sx
        cy_scaled = cy * sy
        widths = np.full((len(image_files),), ow, dtype=np.int64)
        heights = np.full((len(image_files),), oh, dtype=np.int64)
        # COLMAP export (COLMAP stores world->camera; Nerfstudio needs c2w)
    # For COLMAP text, WIDTH/HEIGHT are per-camera constants. If you use_shared_camera,
    # pass the common original size; otherwise, COLMAP’s text format still expects a single size per camera.
    common_w = int(widths[0]) if len(widths) else W
    common_h = int(heights[0]) if len(heights) else H

    write_colmap_text(
        out_dir=colmap_dir,
        image_files=image_files,
        poses_c2w=poses,
        width=common_w,
        height=common_h,
        fx=fx_scaled, fy=fy_scaled, cx=cx_scaled, cy=cy_scaled,
        poses_are_c2w=poses_are_c2w,
        use_shared_camera=use_shared_camera,
        points_xyz=points_xyz,
    )

    # Nerfstudio export (write w/h as the actual image size and use scaled intrinsics)
    # If frames have mixed sizes, we’ll store per-frame intrinsics (already handled by not using shared camera,
    # or by still emitting per-frame fl_x,.. when use_shared_camera=False).
    # If you prefer shared camera AND mixed sizes, consider forcing a resize or disabling shared camera.
    ns_w = int(widths[0]) if len(widths) else W
    ns_h = int(heights[0]) if len(heights) else H

    write_nerfstudio_transforms(
        out_dir=ns_dir,
        image_files=image_files,
        poses_c2w=poses,
        width=ns_w,
        height=ns_h,
        fx=fx_scaled, fy=fy_scaled, cx=cx_scaled, cy=cy_scaled,
        use_shared_camera=use_shared_camera,
        pack_images=pack_nerfstudio,
        link_mode=link_mode,
    )


    print(f"\n✅ Export complete.")
    print(f"  COLMAP text model: {colmap_dir}")
    print(f"  Nerfstudio scene:  {ns_dir}")
    if pack_nerfstudio:
        print(f"  Images packed to:  {os.path.join(ns_dir, 'images')}")


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Pi3 on an image sequence and export COLMAP text + Nerfstudio transforms.json"
    )
    p.add_argument("--input-dir", required=True, help="Directory containing images (png/jpg).")
    p.add_argument("--output-dir", required=True, help="Directory to write exports into.")
    p.add_argument("--poses-are-c2w", action="store_true",
                   help="Set if Pi3 poses are camera-to-world (default). If unset, poses are treated as world-to-camera.")
    p.add_argument("--use-shared-camera", action="store_true",
                   help="Export a single shared PINHOLE camera (fx,fy,cx,cy from first frame or FOV fallback).")
    p.add_argument("--interval", type=int, default=10,
                   help="Sample every k-th image from the folder when loading (default: 10).")
    p.add_argument("--fov-deg", type=float, default=60.0,
                   help="Fallback FOV (degrees) if intrinsics are not provided by Pi3 (default: 60).")
    p.add_argument("--pack-nerfstudio", action="store_true",
                   help="Copy/link sampled frames into nerfstudio/images/ and reference them from transforms.json.")
    p.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy",
                   help="How to materialize frames into the Nerfstudio folder (default: copy).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pi3_and_export(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poses_are_c2w=bool(args.poses_are_c2w),
        use_shared_camera=bool(args.use_shared_camera),
        interval=int(args.interval),
        fov_deg=float(args.fov_deg),
        pack_nerfstudio=bool(args.pack_nerfstudio),
        link_mode=args.link_mode,  # type: ignore
    )

