#!/usr/bin/env python3
# tools/compute_habitat_stats.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Use the Advanced dataset so samples are dicts {"videos","conds","nonterminal"}.
from datasets.video.habitat import HabitatAdvancedVideoDataset as HabitatDataset


def _rescale_crop_params(H0: int, W0: int, res: int) -> Tuple[float, int, int]:
    s = res / float(min(H0, W0))
    Hs = int(round(H0 * s))
    Ws = int(round(W0 * s))
    ty = max(0, (Hs - res) // 2)
    tx = max(0, (Ws - res) // 2)
    return s, tx, ty

def _adjust_K(K: np.ndarray, s: float, tx: int, ty: int) -> np.ndarray:
    K = K.copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = K[0, 2] * s - tx
    K[1, 2] = K[1, 2] * s - ty
    return K

def _load_intrinsics_for_seq(seq_dir: Path) -> np.ndarray:
    intr_path = seq_dir.parent / "intrinsics.npy"
    arr = np.load(intr_path, allow_pickle=False)
    if arr.shape == (3, 3):     K = arr.astype(np.float64)
    elif arr.shape == (4, 4):   K = arr[:3, :3].astype(np.float64)
    elif arr.shape == (4,):     fx,fy,cx,cy = map(float, arr); K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
    elif arr.shape == (1, 4):   fx,fy,cx,cy = map(float, arr[0]); K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
    else: raise ValueError(f"Bad intrinsics shape {arr.shape} at {intr_path}")
    return K

def _first_frame_size(seq_dir: Path) -> Tuple[int, int]:
    rgb = seq_dir / "rgb"
    files = sorted(rgb.glob("*.npy"), key=lambda p: (len(p.stem), p.stem))
    if not files: raise FileNotFoundError(f"No rgb/*.npy in {rgb}")
    img = np.load(files[0])
    return int(img.shape[0]), int(img.shape[1])  # H0, W0

def _check_rotation(R: torch.Tensor, tol: float = 5e-3) -> Tuple[bool, float, float]:
    I = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    ortho = (R.transpose(-1, -2) @ R) - I
    orth_err = ortho.abs().max().item()
    det = torch.det(R)
    det_err = (det - 1.0).abs().max().item()
    ok = (orth_err < tol) and (det_err < 5e-2)
    return ok, orth_err, det_err

def _flip_indices_for_cond16() -> List[int]:
    # For processed cond = [fx,fy,cx,cy, r00,r01,r02,t0, r10,r11,r12,t1, r20,r21,r22,t2]
    # RE10K-style horizontal flip of extrinsics (K unchanged):
    return [5, 6, 7, 8, 12]

def _as_yaml_3x1x1(x: torch.Tensor) -> List[List[List[float]]]:
    x = x.detach().cpu().float()
    return [[[float(x[0])]], [[float(x[1])]], [[float(x[2])]]]

@torch.no_grad()
def _validate_sample(sample: Dict[str, torch.Tensor], res: int, training_split: bool) -> None:
    vids = sample["videos"]  # (T,C,H,W)
    cond = sample["conds"]  # (T,16)
    nonterm = sample["nonterminal"]  # (T,)

    assert vids.ndim == 4 and vids.shape[1] == 3, f"videos shape bad: {vids.shape}"
    T, C, H, W = vids.shape
    assert (H, W) == (res, res), f"video not {res}x{res}: {(H,W)}"
    assert vids.dtype in (torch.float32, torch.float64), "videos must be float"
    assert torch.isfinite(vids).all(), "video has NaN/Inf"
    mn, mx = vids.min().item(), vids.max().item()
    assert -1e-6 <= mn <= 1.0 + 1e-6 and -1e-6 <= mx <= 1.0 + 1e-6, f"video not in [0,1]: ({mn},{mx})"

    assert cond.shape[0] == T and cond.shape[1] == 16, f"cond shape bad: {cond.shape}"
    assert torch.isfinite(cond).all(), "cond has NaN/Inf"

    assert nonterm.shape[0] == T
    if training_split:
        assert nonterm.sum().item() == T, "training clip padded unexpectedly"

    fx, fy, cx, cy = [cond[0, i].item() for i in range(4)]
    assert fx > 1 and fy > 1 and 0 <= cx <= res and 0 <= cy <= res, f"unusual K: {(fx,fy,cx,cy)}"

    E = cond[:, 4:].reshape(T, 3, 4)
    R = E[:, :, :3]
    ok, orth_err, det_err = _check_rotation(R)
    assert ok, f"R not orthonormal enough (|R^T R - I|∞={orth_err:.3e}, |det-1|={det_err:.3e})"
    t = E[:, :, 3]
    assert torch.isfinite(t).all(), "translation has NaN/Inf"

@torch.no_grad()
def _validate_K_consistency(ds: HabitatDataset, idx: int, atol=1e-3, rtol=5e-3) -> None:
    meta = ds.metadata[idx]
    seq_dir = Path(meta["video_paths"])
    H0, W0 = _first_frame_size(seq_dir)
    s, tx, ty = _rescale_crop_params(H0, W0, int(ds.resolution))
    K0 = _load_intrinsics_for_seq(seq_dir)
    K = _adjust_K(K0, s, tx, ty)  # target fx,fy,cx,cy
    sample = ds[idx]
    cond = sample["conds"]
    fx, fy, cx, cy = [cond[0, i].item() for i in range(4)]
    target = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
    got = np.array([fx, fy, cx, cy], dtype=np.float64)
    if not np.allclose(got, target, atol=atol, rtol=rtol):
        raise AssertionError(f"K mismatch:\n  expected={target}\n  got     ={got}\n  tol atol={atol} rtol={rtol}")

@torch.no_grad()
def _validate_horizontal_flip_deterministic(ds: HabitatDataset, idx: int) -> None:
    """
    Deterministic flip test:
    - Force SAME (video_idx, clip_idx)
    - Reproduce frame-skip + cond processing
    - Apply flip manually to one copy
    - Check pixels and conds match expected behavior
    """
    # Resolve clip location deterministically
    video_idx, clip_idx = ds.get_clip_location(idx)
    meta = ds.metadata[video_idx]
    V = ds.video_length(meta)
    cfg = ds.cfg

    # Compute frame skip like __getitem__ (training branch)
    frame_skip = (V - clip_idx - 1) // (cfg.max_frames - 1)
    # Use the scheduled cap if defined on the dataset (subclass property)
    if hasattr(ds, "_training_frame_skip"):
        frame_skip = min(frame_skip, ds._training_frame_skip)
    assert frame_skip > 0, f"frame_skip={frame_skip} must be > 0"

    end_frame = clip_idx + (cfg.max_frames - 1) * frame_skip + 1

    # Load raw (pre-augment) clip, then apply frame skip + cond processing
    video_raw, cond_raw = ds.load_video_and_cond(meta, clip_idx, end_frame)
    video = video_raw[::frame_skip]                              # (T,C,H,W)
    cond  = ds._process_external_cond(cond_raw, frame_skip)      # (T,16)

    # Create a flipped copy (deterministic)
    video_flip = video.flip(-1).contiguous()
    cond_flip = cond.clone()
    cond_flip[:, _flip_indices_for_cond16()] *= -1

    # Compare: flip(video) equals video_flip
    diff = (video.flip(-1) - video_flip).abs().max().item()
    assert diff < 2/255.0 + 1e-6, f"pixels not flipped as expected (max abs diff {diff})"

    # Compare cond flip indices
    same = cond.clone()
    same[:, _flip_indices_for_cond16()] *= -1
    err = (same - cond_flip).abs().max().item()
    assert err < 1e-5, f"cond sign flips differ (max abs diff {err})"

@torch.no_grad()
def compute_mean_std_streaming(
    ds: HabitatDataset,
    subset_clips: Optional[int],
    batch_size: int,
    num_workers: int,
    validate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-channel mean/std using a streaming pass.
    Masks out padded frames via 'nonterminal' to avoid bias.
    """
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    sum_c = torch.zeros(3, dtype=torch.float64)
    sumsq_c = torch.zeros(3, dtype=torch.float64)
    pix_count = torch.tensor(0.0, dtype=torch.float64)

    seen_clips = 0
    training_split = (ds.split == "training")

    for batch in dl:
        vids = batch["videos"]      # (B,T,C,H,W)
        conds = batch["conds"]      # (B,T,16)
        nonterm = batch["nonterminal"]  # (B,T) bool

        if vids.ndim == 4:  # (T,C,H,W) – guard
            vids   = vids.unsqueeze(0)
            conds  = conds.unsqueeze(0)
            nonterm= nonterm.unsqueeze(0)

        B, T, C, H, W = vids.shape

        if validate:
            for b in range(B):
                _validate_sample({"videos": vids[b], "conds": conds[b], "nonterminal": nonterm[b]},
                                 ds.resolution, training_split)

        # mask out padded frames
        mask = nonterm.to(torch.float64).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1,1)
        vids64 = vids.to(torch.float64)
        sum_c   += (vids64 * mask).sum(dim=(0,1,3,4))              # (C,)
        sumsq_c += (vids64.pow(2) * mask).sum(dim=(0,1,3,4))       # (C,)
        pix_count += (nonterm.to(torch.float64).sum() * H * W)

        seen_clips += B
        if subset_clips is not None and seen_clips >= subset_clips:
            break

    mean = (sum_c / pix_count).to(torch.float32)
    var  = (sumsq_c / pix_count).to(torch.float32) - mean**2
    std  = torch.sqrt(torch.clamp(var, min=0.0) + 1e-12)
    return mean, std


# ------------------------------ CLI & main ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--habitat_dir", type=str, default="/scratch/tshu2/yyin34/projects/3d_belief/partnr-planner/data/trajectories/habelief")
    p.add_argument("--save_dir", type=str, default="/scratch/tshu2/zwen19/diffusion-forcing-transformer/data/habitat")  # ABS path recommended
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--split", type=str, default="training", choices=["training", "validation", "test"])
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--frame_skip", type=int, default=10)
    p.add_argument("--context_length", type=int, default=4)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--subset", type=int, default=None, help="Number of clips to include in stats pass")

    p.add_argument("--check_batches", type=int, default=8, help="#batches to scan for basic validation (shapes/ranges/etc.)")
    p.add_argument("--check_K_on", type=int, default=3, help="#indices to check K-consistency (0 to skip)")
    p.add_argument("--check_flip", action="store_true", help="Run deterministic horizontal-flip test")
    p.add_argument("--anchor_to_first", action="store_true", help="Anchor poses to first frame (consistent scenes)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def build_cfg(args) -> OmegaConf:
    # Fill required keys for BaseVideoDataset / BaseAdvancedVideoDataset
    cfg = OmegaConf.create({
        "name": "habitat",
        "habitat_dir": args.habitat_dir,
        "save_dir": args.save_dir,
        "metadata_dir": str(Path(args.save_dir) / "metadata"),
        "external_cond_dim": 16,
        "external_cond_stack": False,          # required
        "max_frames": args.max_frames,
        "frame_skip": args.frame_skip,
        "context_length": args.context_length,
        "resolution": args.resolution,
        "latent": {                            # required
            "downsampling_factor": [1, 8],
            "suffix": "",
            "enable": False,
            "type": None,
        },
        "filter_min_len": None,
        "num_eval_videos": 1024,
        "subdataset_size": None,
        "maximize_training_data": False,
        "augmentation": {
            "frame_skip_increase": 0,
            "horizontal_flip_prob": 0.0,       # keep OFF for stats/validation pass
            "reverse_prob": 0.0,
            "back_and_forth_prob": 0.0
        },
        "anchor_to_first": bool(args.anchor_to_first),
    })
    return cfg

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = build_cfg(args)
    ds = HabitatDataset(cfg, split=args.split)

    # (A) Quick sanity on a few batches
    print(f"[sanity] scanning {args.check_batches} batches for basic validity...")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False, drop_last=False)
    scanned = 0
    for batch in dl:
        vids = batch["videos"]      # (B,T,C,H,W)
        conds = batch["conds"]      # (B,T,16)
        nonterms = batch["nonterminal"]  # (B,T)
        if vids.ndim == 4:  # (T,C,H,W)
            vids = vids.unsqueeze(0); conds = conds.unsqueeze(0); nonterms = nonterms.unsqueeze(0)
        B = vids.shape[0]
        for b in range(B):
            _validate_sample({"videos": vids[b], "conds": conds[b], "nonterminal": nonterms[b]},
                             res=args.resolution,
                             training_split=(args.split=="training"))
        scanned += 1
        if scanned >= args.check_batches:
            break
    print("[ok] basic batch validation passed ✓")

    # (B) Intrinsics K-consistency on a few indices
    if args.check_K_on > 0:
        n = len(ds.metadata)
        idxs = np.linspace(0, max(0, n-1), num=min(args.check_K_on, n), dtype=int)
        for i in idxs:
            _validate_K_consistency(ds, int(i))
        print(f"[ok] K-consistency passed on indices: {idxs.tolist()}")

    # (C) Deterministic horizontal flip behavior (clip-locked, manual flip)
    if args.check_flip and args.split == "training":
        _validate_horizontal_flip_deterministic(ds, idx=0)
        print("[ok] deterministic horizontal flip test passed")

    # (D) Streaming mean/std (masked by nonterminal)
    print("[stats] computing per-channel mean/std (masked, post-preprocess)...")
    mean, std = compute_mean_std_streaming(
        ds,
        subset_clips=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validate=False,  # already validated above
    )

    print("\n===== Dataset statistics (post-preprocessing) =====")
    print(f"Per-channel mean (R,G,B): {mean.tolist()}")
    print(f"Per-channel std  (R,G,B): {std.tolist()}")
    print("\nYAML snippet:")
    print("data_mean:", _as_yaml_3x1x1(mean))
    print("data_std: ", _as_yaml_3x1x1(std))
    print("\n[ok] validation + stats complete ✓")

if __name__ == "__main__":
    main()