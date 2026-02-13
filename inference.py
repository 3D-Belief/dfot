# inference.py
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import Trainer

from omegaconf import OmegaConf

# Adjust import to your tree:
from algorithms.dfot.dfot_video_pose import DFoTVideoPose as ModuleClass


# ----------------- small utils -----------------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def _is_dirlike(pattern: str) -> bool:
    p = Path(pattern)
    return p.exists() and p.is_dir()

def _glob_all(pattern: str) -> List[str]:
    from glob import glob
    pat = pattern
    if _is_dirlike(pattern):
        pat = str(Path(pattern) / "*")
    return glob(pat)

# ----------------- I/O helpers -----------------
def _read_pose_any(p: Path) -> np.ndarray:
    """Load one OpenGL c2w pose: .npy (3x4 or 4x4) or text with 12/16 floats."""
    if p.suffix.lower() == ".npy":
        m = np.load(p, allow_pickle=False)
    else:
        vals = np.fromstring(p.read_text().strip(), sep=" ")
        if vals.size == 12:
            m = vals.reshape(3, 4)
        elif vals.size == 16:
            m = vals.reshape(4, 4)
        else:
            raise ValueError(f"Unsupported pose text format in {p} (need 12 or 16 floats).")
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (3, 4): return m
    if m.shape == (4, 4): return m
    raise ValueError(f"Pose shape {m.shape} in {p}: expected (3,4) or (4,4).")

def _read_image_any(p: Path) -> np.ndarray:
    """Read RGB image as uint8 HxWx3 from .npy or standard image formats."""
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.5:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    from PIL import Image
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

# ----------------- camera/geo helpers -----------------
def _to_4x4(arr: np.ndarray) -> np.ndarray:
    m = np.asarray(arr, dtype=np.float64)
    if m.shape == (4, 4): return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64); out[:3, :4] = m
        return out
    raise ValueError(f"Pose shape {m.shape}: expected (3,4) or (4,4).")

def _opengl_c2w_to_opencv_w2c_3x4(c2w_gl_4x4: np.ndarray,
                                  anchor: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    # GL c2w -> CV w2c
    R_gl = c2w_gl_4x4[:3, :3]; t_gl = c2w_gl_4x4[:3, 3]
    if anchor is not None:
        R0, t0 = anchor
        R_gl = R0.T @ R_gl
        t_gl = R0.T @ (t_gl - t0)
    F = np.diag([1.0, -1.0, -1.0])
    R_cv_c2w = F @ R_gl @ F
    t_cv_c2w = F @ t_gl
    R = R_cv_c2w.T
    t = -R @ t_cv_c2w
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return np.hstack([R, t[:, None]]).astype(np.float32)

def _resize_center_crop_thwc_uint8(video_thwc_uint8: torch.Tensor, res: int) -> np.ndarray:
    """Short-side resize → center crop to (res,res). Input THWC uint8 torch, output THWC uint8 numpy."""
    T, H0, W0, C = video_thwc_uint8.shape
    s = res / float(min(H0, W0))
    Hs = int(round(H0 * s)); Ws = int(round(W0 * s))
    vid = video_thwc_uint8.permute(0, 3, 1, 2).float()  # T,C,H,W
    vid = torch.nn.functional.interpolate(vid, size=(Hs, Ws), mode="bilinear", align_corners=False)
    ty = max(0, (Hs - res)//2); tx = max(0, (Ws - res)//2)
    vid = vid[:, :, ty:ty+res, tx:tx+res]  # T,C,res,res
    return vid.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # T,H,W,C

def _build_cond_from_c2w_gl(
    c2w_list: List[np.ndarray],
    K_in: np.ndarray,     # (3,3) or (4,) or (1,4)
    H: int, W: int,
    anchor_to_first: bool = False
) -> torch.Tensor:
    """Return (T,16): [fx,fy,cx,cy] + 12*E with (H,W) dropped to mirror training cond processing."""
    K = np.asarray(K_in, dtype=np.float64)
    if K.shape == (3, 3):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    elif K.shape == (4,):
        fx, fy, cx, cy = [float(x) for x in K]
    elif K.shape == (1, 4):
        fx, fy, cx, cy = map(float, K[0])
    else:
        raise ValueError(f"K must be (3,3) or (4,) or (1,4); got {K.shape}")

    first = _to_4x4(c2w_list[0])
    anchor = (first[:3, :3], first[:3, 3]) if anchor_to_first else None

    rows = []
    for c2w in c2w_list:
        E = _opengl_c2w_to_opencv_w2c_3x4(_to_4x4(c2w), anchor)  # (3,4)
        e_flat = E.reshape(3, 4).flatten()
        # match DFoT cond semantics: [fx,fy,cx,cy, H, W, e(12...)] then drop H,W
        row18 = np.concatenate([[fx, fy, cx, cy, float(H), float(W)], e_flat], axis=0)
        rows.append(row18)
    cams18 = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)  # (T,18)
    cams16 = torch.cat([cams18[:, :4], cams18[:, 6:]], dim=-1)          # drop H,W -> (T,16)
    return cams16

# ----------------- minimal config recovery -----------------
def _minimal_algo_cfg_for_infer():
    return OmegaConf.create({
        "_name": "dfot_video_pose",
        "tasks": {
            "prediction": {
                "history_guidance": {"name": "vanilla", "guidance_scale": 0.0},
                "sampler": {"name": "ddim", "steps": 50, "eta": 0.0},
            }
        },
        "diffusion": {
            "is_continuous": True,
            "precond_scale": 0.125,
            "beta_schedule": "cosine_simple_diffusion",
            "schedule_fn_kwargs": {"shifted": 0.125, "interpolated": False},
            "training_schedule": {"name": "cosine", "shift": 0.125},
            "loss_weighting": {"strategy": "sigmoid", "sigmoid_bias": -1.0},
        },
        "backbone": {"use_fourier_noise_embedding": True},
    })

def build_algo_cfg_from_ckpt(ckpt_path: str, overrides: dict | None = None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    algo_cfg = None

    if isinstance(hp.get("algorithm"), (dict, OmegaConf)):
        algo_cfg = hp["algorithm"]
    elif isinstance(hp.get("cfg"), (dict, OmegaConf)):
        maybe = hp["cfg"]
        if isinstance(maybe, dict) and "algorithm" in maybe:
            algo_cfg = maybe["algorithm"]
        else:
            algo_cfg = maybe

    if algo_cfg is None:
        algo_cfg = _minimal_algo_cfg_for_infer()
    if not isinstance(algo_cfg, OmegaConf.__class__) and not hasattr(algo_cfg, "get"):
        algo_cfg = OmegaConf.create(algo_cfg)

    if overrides:
        algo_cfg = OmegaConf.merge(algo_cfg, OmegaConf.create(overrides))
    return algo_cfg

def load_dfot_for_infer(ckpt_path: str, cfg_overrides: dict | None = None, device: str = "cuda"):
    algo_cfg = build_algo_cfg_from_ckpt(ckpt_path, overrides=cfg_overrides)
    model = ModuleClass(algo_cfg)
    ck = torch.load(ckpt_path, map_location="cpu")
    # prefer EMA if present
    state = ck.get("state_dict_ema") or ck.get("state_dict") or ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)} while loading weights")
    model.eval()
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    return model.to(dev)

# ----------------- one-shot dataset -----------------
class OneBatchImagesPoses(Dataset):
    """
    Minimal dataset:
      emits exactly one batch with keys: 'videos' (T,C,H,W), 'conds' (T,16), 'nonterminal' (T)
    """
    def __init__(
        self,
        images: List[np.ndarray] | List[str | Path],
        c2w_gl_list: List[np.ndarray],
        K: np.ndarray,
        context: int = 4,
        predict: int = 12,
        frame_skip: int = 4,
        resolution: int = 256,
        pure_generation: bool = False,
        anchor_to_first: bool = False,
    ):
        total_needed = (context + predict) * frame_skip
        assert len(images) >= total_needed and len(c2w_gl_list) >= total_needed, \
            f"Need >= {(context + predict)} frames @ frame_skip={frame_skip}; got {len(images)}"

        take = list(range(0, total_needed, frame_skip))

        if isinstance(images[0], (str, Path)):
            imgs = [_read_image_any(Path(images[i])) for i in take]
        else:
            imgs = [np.asarray(images[i], dtype=np.uint8) for i in take]

        # THWC uint8 -> resize/crop -> TCHW float
        vid_thwc = torch.from_numpy(np.stack(imgs, axis=0))  # T,H,W,C uint8
        vid_thwc_np = _resize_center_crop_thwc_uint8(vid_thwc, resolution)
        video = torch.from_numpy(vid_thwc_np).permute(0, 3, 1, 2).float() / 255.0

        if pure_generation:
            video[context:] = 0.0

        c2w_subset = [c2w_gl_list[i] for i in take]
        conds = _build_cond_from_c2w_gl(c2w_subset, np.asarray(K), resolution, resolution, anchor_to_first)

        self.video = video              # (T,C,H,W)
        self.cond  = conds              # (T,16)
        self.nonterminal = torch.ones(video.shape[0], dtype=torch.bool)

    def __len__(self): return 1
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"videos": self.video, "conds": self.cond, "nonterminal": self.nonterminal}

# ----------------- prediction runners -----------------
@torch.no_grad()
def predict_with_trainer(ckpt: str, ds: Dataset, devices: int = 1, precision: str = "16-mixed", cfg_overrides: dict | None = None):
    module = load_dfot_for_infer(ckpt, cfg_overrides=cfg_overrides, device="cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        precision=precision
    )
    preds = trainer.predict(module, dl)
    return preds[0] if preds else None

@torch.no_grad()
def predict_direct(ckpt: str, ds: Dataset, cfg_overrides: dict | None = None, device: str = "cuda"):
    module = load_dfot_for_infer(ckpt, cfg_overrides=cfg_overrides, device=device)
    batch = ds[0]
    batch = {k: v.unsqueeze(0).to(module.device) for k, v in batch.items()}  # add batch dim
    # Works if your LightningModule implements predict_step; otherwise call your sampler directly
    out = module.predict_step(batch, batch_idx=0)
    return out

# ----------------- save gif -----------------
def save_tensor_video_as_gif(x: torch.Tensor, path: str, fps: int = 8):
    try:
        import imageio
    except Exception:
        print("Install imageio to save GIFs: pip install imageio")
        return
    if x.ndim == 5:  # B,T,C,H,W
        x = x[0]
    x = (x.clamp(0,1).detach().cpu().permute(0,2,3,1).numpy() * 255).astype(np.uint8)
    imageio.mimsave(path, list(x), fps=fps)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images_glob", required=True, help="dir or glob for images (.png/.jpg/.npy)")
    ap.add_argument("--poses_glob",  required=True, help="dir or glob for poses (.npy or text 12/16 floats)")
    ap.add_argument("--K", required=True, help="npy of (3,3) or (4,) = (fx,fy,cx,cy)")
    ap.add_argument("--out_gif", default="dfot_pred.gif")
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--predict", type=int, default=12)
    ap.add_argument("--frame_skip", type=int, default=4)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--pure_gen", action="store_true")
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--direct", action="store_true", help="use predict_step without Trainer")
    # optional runtime overrides for the checkpoint algo cfg
    ap.add_argument("--guidance", type=str, default="vanilla")
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--anchor_to_first", action="store_true")
    args = ap.parse_args()

    img_paths = sorted([Path(p) for p in _glob_all(args.images_glob)], key=_natural_key)
    pose_paths = sorted([Path(p) for p in _glob_all(args.poses_glob)], key=_natural_key)
    assert img_paths, f"No images found: {args.images_glob}"
    assert pose_paths, f"No poses found: {args.poses_glob}"

    # Match by stem
    poses_by_stem = {pf.stem: pf for pf in pose_paths}
    images = [p for p in img_paths if p.stem in poses_by_stem]
    if not images:
        raise FileNotFoundError("No (image, pose) pairs with matching stems.")
    missing = [p for p in img_paths if p.stem not in poses_by_stem]
    if missing:
        print(f"[warn] {len(missing)} images have no matching pose (skipped).")

    c2w_list = [_read_pose_any(poses_by_stem[p.stem]) for p in images]
    K = np.load(args.K)

    ds = OneBatchImagesPoses(
        images=images,
        c2w_gl_list=c2w_list,
        K=K,
        context=args.context,
        predict=args.predict,
        frame_skip=args.frame_skip,
        resolution=args.res,
        pure_generation=args.pure_gen,
        anchor_to_first=args.anchor_to_first,
    )

    cfg_overrides = {
        "tasks": {"prediction": {"history_guidance": {"name": args.guidance, "guidance_scale": float(args.guidance_scale)}}},
        "diffusion": {"is_continuous": True, "precond_scale": 0.125},
        "backbone": {"use_fourier_noise_embedding": True},
    }

    if args.direct:
        out = predict_direct(args.ckpt, ds, cfg_overrides=cfg_overrides)
    else:
        out = predict_with_trainer(args.ckpt, ds, devices=args.devices, precision=args.precision, cfg_overrides=cfg_overrides)
    print("[OK] prediction finished.")

    # Extract a tensor video and save
    vid = None
    if torch.is_tensor(out):
        vid = out
    elif isinstance(out, dict):
        for k in ["pred_videos", "videos", "samples", "outputs", "generated"]:
            if k in out and torch.is_tensor(out[k]):
                vid = out[k]
                break
    if vid is None:
        print("Prediction returned but no tensor video found; keys:", (list(out.keys()) if isinstance(out, dict) else type(out)))
        return

    fps = max(1, 8 // max(1, args.frame_skip))
    save_tensor_video_as_gif(vid, args.out_gif, fps=fps)
    print(f"[OK] wrote {args.out_gif}")

if __name__ == "__main__":
    main()
