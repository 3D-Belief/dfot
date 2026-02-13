#!/usr/bin/env python3
# tools/compute_spoc_stats.py
from __future__ import annotations
import argparse, os, math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Use the Advanced dataset so samples are dicts {"videos","conds","nonterminal"}.
from datasets.video.spoc import SpocAdvancedVideoDataset as SpocDataset

# ---------- geometry helpers ----------
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

def _spoc_normalized_intrinsics() -> Tuple[float, float, float, float]:
    # SPOC normalized intrinsics (per your VGGT loader)
    return 0.390, 0.385, 0.5, 0.5  # fx_n, fy_n, cx_n, cy_n

def _pixel_intrinsics_from_normalized(H0: int, W0: int) -> np.ndarray:
    fx_n, fy_n, cx_n, cy_n = _spoc_normalized_intrinsics()
    K = np.array([
        [fx_n * W0, 0.0,       cx_n * W0],
        [0.0,       fy_n * H0, cy_n * H0],
        [0.0,       0.0,       1.0]
    ], dtype=np.float64)
    return K

def _first_frame_size(seq_dir: Path) -> Tuple[int, int]:
    """
    SPOC now stores RGB as videos/rgb_trajectory.mp4.
    Return (H0, W0) from the mp4 metadata.
    """
    mp4 = seq_dir / "videos" / "rgb_trajectory.mp4"
    if not mp4.is_file():
        raise FileNotFoundError(f"Missing rgb video: {mp4}")
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {mp4}")
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return H0, W0

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
    return [5, 6, 7, 8, 12]

def _as_yaml_3x1x1(x: torch.Tensor) -> List[List[List[float]]]:
    x = x.detach().cpu().float()
    return [[[float(x[0])]], [[float(x[1])]], [[float(x[2])]]]

# ---------- validations ----------
@torch.no_grad()
def _validate_sample(sample: Dict[str, torch.Tensor], res: int, training_split: bool) -> None:
    vids = sample["videos"]      # (T,C,H,W)
    cond = sample["conds"]       # (T,16)
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
def _validate_K_consistency(ds: SpocDataset, idx: int, atol=1e-3, rtol=5e-3) -> None:
    """
    Rebuild K from SPOC normalized intrinsics and original (H0,W0), then
    apply the same resize+center-crop adjustment used by the loader. Compare
    to cond[:4] from the processed sample.
    """
    meta = ds.metadata[idx]
    seq_dir = Path(meta["video_paths"])
    H0, W0 = _first_frame_size(seq_dir)
    s, tx, ty = _rescale_crop_params(H0, W0, int(ds.resolution))
    K0 = _pixel_intrinsics_from_normalized(H0, W0)  # normalized → pixel
    K = _adjust_K(K0, s, tx, ty)                    # target fx,fy,cx,cy

    sample = ds[idx]
    cond = sample["conds"]
    fx, fy, cx, cy = [cond[0, i].item() for i in range(4)]
    target = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
    got = np.array([fx, fy, cx, cy], dtype=np.float64)
    if not np.allclose(got, target, atol=atol, rtol=rtol):
        raise AssertionError(f"K mismatch:\n  expected={target}\n  got     ={got}\n  tol atol={atol} rtol={rtol}")

@torch.no_grad()
def _validate_horizontal_flip_deterministic(ds: SpocDataset, idx: int) -> None:
    """
    Deterministic flip test mirroring your Habitat checker.
    """
    video_idx, clip_idx = ds.get_clip_location(idx)
    meta = ds.metadata[video_idx]
    V = ds.video_length(meta)
    cfg = ds.cfg

    frame_skip = (V - clip_idx - 1) // (cfg.max_frames - 1)
    if hasattr(ds, "_training_frame_skip"):
        frame_skip = min(frame_skip, ds._training_frame_skip)
    assert frame_skip > 0, f"frame_skip={frame_skip} must be > 0"
    end_frame = clip_idx + (cfg.max_frames - 1) * frame_skip + 1

    if hasattr(ds, "load_video_and_cond"):
        video_raw, cond_raw = ds.load_video_and_cond(meta, clip_idx, end_frame)
    else:
        video_raw = ds.load_video(meta, clip_idx, end_frame)
        cond_raw  = ds.load_cond(meta,  clip_idx, end_frame)

    video = video_raw[::frame_skip]                         # (T,C,H,W)
    cond  = ds._process_external_cond(cond_raw, frame_skip) # (T,16)

    video_flip = video.flip(-1).contiguous()
    cond_flip = cond.clone()
    cond_flip[:, _flip_indices_for_cond16()] *= -1

    diff = (video.flip(-1) - video_flip).abs().max().item()
    assert diff < 2/255.0 + 1e-6, f"pixels not flipped as expected (max abs diff {diff})"

    same = cond.clone(); same[:, _flip_indices_for_cond16()] *= -1
    err = (same - cond_flip).abs().max().item()
    assert err < 1e-5, f"cond sign flips differ (max abs diff {err})"

# ---------- streaming stats ----------
@torch.no_grad()
def compute_mean_std_streaming(
    ds: SpocDataset,
    subset_clips: Optional[int],
    batch_size: int,
    num_workers: int,
    validate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        vids = batch["videos"]      # (B,T,C,H,W) or (T,C,H,W)
        conds = batch["conds"]
        nonterm = batch["nonterminal"]

        if vids.ndim == 4:
            vids   = vids.unsqueeze(0)
            conds  = conds.unsqueeze(0)
            nonterm= nonterm.unsqueeze(0)

        B, T, C, H, W = vids.shape

        if validate:
            for b in range(B):
                _validate_sample({"videos": vids[b], "conds": conds[b], "nonterminal": nonterm[b]},
                                 ds.resolution, training_split)

        mask = nonterm.to(torch.float64).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1,1)
        vids64 = vids.to(torch.float64)
        sum_c   += (vids64 * mask).sum(dim=(0,1,3,4))
        sumsq_c += (vids64.pow(2) * mask).sum(dim=(0,1,3,4))
        pix_count += (nonterm.to(torch.float64).sum() * H * W)

        seen_clips += B
        if subset_clips is not None and seen_clips >= subset_clips:
            break

    mean = (sum_c / pix_count).to(torch.float32)
    var  = (sumsq_c / pix_count).to(torch.float32) - mean**2
    std  = torch.sqrt(torch.clamp(var, min=0.0) + 1e-12)
    return mean, std

# ---------- dataloader summary ---------
def _env_world_and_rank(default_nodes: int = 1) -> Tuple[int, int]:
    # Try torch.distributed first
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
    except Exception:
        pass
    # Fallback to env or single-process
    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank  = int(os.environ.get("RANK", "0"))
    if world <= 0:
        # assume single node, all visible GPUs
        world = max(1, torch.cuda.device_count()) * max(1, default_nodes)
    return world, rank

def _infer_n_frames(ds: SpocDataset) -> Optional[int]:
    # Prefer dataset-provided value
    n = getattr(ds, "n_frames", None)
    if isinstance(n, int) and n > 0:
        return n
    # Else infer from cfg if possible
    try:
        max_frames = int(ds.cfg.max_frames)
        frame_skip = int(ds.frame_skip)
        return 1 + (max_frames - 1) * frame_skip
    except Exception:
        return None

@torch.no_grad()
def print_dataloader_summary(
    ds: SpocDataset,
    batch_size: int,
    accumulate_grad_batches: int,
    max_epochs: int,
    num_workers: int,
    header: str = "Run summary",
) -> None:
    # world/devices
    world_size, rank = _env_world_and_rank(default_nodes=int(getattr(ds.cfg, "num_nodes", 1)))
    num_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices_hint = f"{num_cuda} CUDA (visible) x {int(getattr(ds.cfg, 'num_nodes', 1))} node(s)"

    # per-rank dataloader length
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    len_train_dl = len(dl)  # batches per epoch (this rank)
    accum = max(1, int(accumulate_grad_batches))
    steps_per_epoch = math.ceil(len_train_dl / accum)

    # effective batch across all ranks
    eff_batch = batch_size * max(1, world_size) * accum

    # dataset-specific extras
    n_frames   = _infer_n_frames(ds)
    frame_skip = getattr(ds, "frame_skip", None)
    max_frames = getattr(ds.cfg, "max_frames", None)
    epoch_clips = len(ds)  # clips yielded by this split (after any filtering)
    clips_available = None
    try:
        if hasattr(ds, "metadata") and hasattr(ds, "video_length") and n_frames:
            clips_available = sum(max(ds.video_length(m) - int(n_frames) + 1, 0) for m in ds.metadata)
    except Exception:
        pass

    # aug & stats
    resolution = int(getattr(ds, "resolution", getattr(ds.cfg, "resolution", 0)))
    aug = getattr(ds.cfg, "augmentation", {})
    data_mean = getattr(ds.cfg, "data_mean", None)
    data_std  = getattr(ds.cfg, "data_std",  None)

    # optional current subdataset_size from cfg
    cfg_subsz = getattr(ds.cfg, "subdataset_size", None)
    implied_steps = None
    if isinstance(cfg_subsz, int) and cfg_subsz > 0:
        implied_steps = math.ceil(cfg_subsz / max(1, eff_batch))

    # pretty print
    print("────────────────────────────────────────────────────────────────────────")
    print(header)
    print(f"• World: rank={rank} | world_size={world_size} | devices (visible)={devices_hint}")
    print(f"• Per-GPU batch={batch_size} | Accum={accum} ⇒ Effective batch={eff_batch}")
    print(f"• len(dataloader)={len_train_dl} | steps/epoch≈ceil({len_train_dl}/{accum})={steps_per_epoch}")
    print(f"• Max epochs={int(max_epochs)} | Max steps≈{steps_per_epoch * int(max_epochs)}")
    print(f"• Train epoch clips=len(dataset)={epoch_clips}")
    if clips_available is not None:
        print(f"• Total clips available in metadata={clips_available}")
    if n_frames is not None:
        print(f"• Clip geometry: n_frames={n_frames} (=1 + (max_frames-1)*frame_skip) | frame_skip={frame_skip} | max_frames={max_frames}")
    print(f"• Resolution={resolution} | data_mean={data_mean} | data_std={data_std}")
    print(f"• Augmentation: {aug}")
    print(f"• Dataloader workers: {num_workers}")
    if hasattr(ds, "metadata"):
        print(f"• Sequences (videos_kept)={len(ds.metadata)}")

    # planning helpers for subdataset_size (no CLI changes_
    # Choose a few common target steps/epoch
    targets = [1000, 2000, 5000]
    cap = clips_available if isinstance(clips_available, int) and clips_available > 0 else epoch_clips
    if isinstance(cap, int) and cap > 0:
        print("• Suggested subdataset_size for target steps/epoch:")
        for S in targets:
            sugg = min(S * eff_batch, cap)
            print(f"    - S={S:>5} → subdataset_size≈{sugg}  (cap={cap})")
    if implied_steps is not None:
        print(f"• cfg.subdataset_size={cfg_subsz} ⇒ implied steps/epoch≈ceil({cfg_subsz}/{eff_batch})={implied_steps}")
    print("────────────────────────────────────────────────────────────────────────")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spoc_dir", type=str, default="/scratch/tshu2/yyin34/projects/3d_belief/DFM/data/spoc_video_only")
    p.add_argument("--save_dir", type=str, default="data/spoc")  # ABS recommended
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--split", type=str, default="training", choices=["training", "validation", "test", "unit"])
    p.add_argument("--max_frames", type=int, default=8)
    p.add_argument("--frame_skip", type=int, default=4)
    p.add_argument("--context_length", type=int, default=4)

    # dataloader / “trainer-like” knobs (for the summary only)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=32)

    p.add_argument("--subset", type=int, default=None, help="Number of clips to include in stats pass")

    p.add_argument("--check_batches", type=int, default=8, help="#batches to scan for basic validation (shapes/ranges/etc.)")
    p.add_argument("--check_K_on", type=int, default=3, help="#indices to check K-consistency (0 to skip)")
    p.add_argument("--check_flip", action="store_true", help="Run deterministic horizontal-flip test")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def build_cfg(args) -> OmegaConf:
    cfg = OmegaConf.create({
        "name": "spoc",
        "spoc_dir": args.spoc_dir,
        "save_dir": args.save_dir,
        "metadata_dir": str(Path(args.save_dir) / "metadata"),
        "external_cond_dim": 16,
        "external_cond_stack": False,
        "max_frames": args.max_frames,
        "frame_skip": args.frame_skip,
        "context_length": args.context_length,
        "resolution": args.resolution,
        "latent": {
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
            "horizontal_flip_prob": 0.0,  # keep OFF for stats/validation pass
            "reverse_prob": 0.0,
            "back_and_forth_prob": 0.0
        },
        "anchor_to_first": False,
        "num_nodes": 1,
    })
    return cfg

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = build_cfg(args)
    ds = SpocDataset(cfg, split=args.split)

    # (0) Trainer-style dataloader summary
    print_dataloader_summary(
        ds,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        header="Dataloader summary",
    )

    # (A) Quick sanity on a few batches
    print(f"[sanity] scanning {args.check_batches} batches for basic validity...")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=False, drop_last=False)
    scanned = 0
    for batch in dl:
        vids = batch["videos"]      # (B,T,C,H,W) or (T,C,H,W)
        conds = batch["conds"]
        nonterms = batch["nonterminal"]
        if vids.ndim == 4:
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

    # (B) K-consistency on a few indices
    if args.check_K_on > 0:
        n = len(ds.metadata)
        idxs = np.linspace(0, max(0, n-1), num=min(args.check_K_on, n), dtype=int)
        for i in idxs:
            _validate_K_consistency(ds, int(i))
        print(f"[ok] K-consistency passed on indices: {idxs.tolist()}")

    # (C) Deterministic horizontal flip behavior
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
        validate=False,
    )

    print("\n===== SPOC dataset statistics (post-preprocessing) =====")
    print(f"Per-channel mean (R,G,B): {mean.tolist()}")
    print(f"Per-channel std  (R,G,B): {std.tolist()}")
    print("\nYAML snippet:")
    print("data_mean:", _as_yaml_3x1x1(mean))
    print("data_std: ", _as_yaml_3x1x1(std))
    print("\n[ok] validation + stats complete ✓")

if __name__ == "__main__":
    main()
