# datasets/video/spoc_video_dataset.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Literal, Optional
from pathlib import Path
import re
import numpy as np
import torch
import cv2

from omegaconf import DictConfig

from dfot_utils.print_utils import cyan
from dfot_utils.storage_utils import safe_torch_save
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import rescale_and_crop, random_bool


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def _list_sorted(folder: Path, suffix: str) -> List[Path]:
    files = sorted(folder.glob(f"*{suffix}"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No {suffix} files in {folder}")
    return files

def _to_4x4(arr: np.ndarray) -> np.ndarray:
    m = np.asarray(arr, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :4] = m
        return m4
    raise ValueError(f"Unsupported pose shape {m.shape}; expected (3,4) or (4,4).")

def _opengl_c2w_to_opencv_w2c_3x4(c2w_gl_4x4: np.ndarray,
                                  anchor: Optional[tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    R_gl = c2w_gl_4x4[:3, :3]
    t_gl = c2w_gl_4x4[:3, 3]
    if anchor is not None:
        R0_gl, t0_gl = anchor
        R_gl = R0_gl.T @ R_gl
        t_gl = R0_gl.T @ (t_gl - t0_gl)
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

def _compute_scale_crop_params(H0: int, W0: int, res: int) -> tuple[float, int, int]:
    s = res / float(min(H0, W0))
    Hs = int(round(H0 * s))
    Ws = int(round(W0 * s))
    ty = max(0, (Hs - res) // 2)
    tx = max(0, (Ws - res) // 2)
    return s, tx, ty

def _adjust_K_for_scale_crop(K: np.ndarray, s: float, tx: int, ty: int) -> np.ndarray:
    K = K.copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = K[0, 2] * s - tx
    K[1, 2] = K[1, 2] * s - ty
    return K

def _read_png_rgb_uint8(p: Path) -> np.ndarray:
    try:
        import imageio.v2 as iio
        arr = iio.imread(p)
    except Exception:
        from PIL import Image
        arr = np.array(Image.open(p).convert("RGB"))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr

def _read_mp4_info(mp4_path: Path) -> tuple[int, int, int]:
    """Return (H, W, Nframes) for an mp4."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h, w, n

def _read_mp4_rgb_uint8(mp4_path: Path, start: int, end: int) -> List[np.ndarray]:
    """
    Read frames [start, end) from mp4, return list of HWC uint8 RGB arrays.
    Uses OpenCV (BGR->RGB conversion).
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {mp4_path}")

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frames: List[np.ndarray] = []
    for _ in range(max(0, end - start)):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.uint8, copy=False))
    cap.release()

    if len(frames) != (end - start):
        raise RuntimeError(
            f"Short read from {mp4_path}: requested [{start},{end}) "
            f"but got {len(frames)} frames"
        )
    return frames

# SPOC uses normalized intrinsics; convert to pixels then adjust for resize/crop
def _load_spoc_intrinsics_as_pixels(img_hw: tuple[int, int]) -> np.ndarray:
    # normalized SPOC intrinsics
    fx_n, fy_n, cx_n, cy_n = 0.390, 0.385, 0.5, 0.5
    H0, W0 = img_hw
    K = np.array([
        [fx_n * W0, 0.0,       cx_n * W0],
        [0.0,       fy_n * H0, cy_n * H0],
        [0.0,       0.0,       1.0]
    ], dtype=np.float64)
    return K

def _pose_rgb_basename(png: Path) -> str:
    # rgb:  frame_0001.png  -> "frame_0001"
    # pose: frame_0001_pose.npy -> "frame_0001"
    stem = png.stem
    return stem

def _pose_stem_to_rgb_basename(npy: Path) -> str:
    # remove trailing "_pose" if present
    stem = npy.stem
    if stem.endswith("_pose"):
        stem = stem[:-5]
    return stem

class SpocBaseVideoDataset(BaseVideoDataset):
    """
    SPOC dataset:
      - reads RGB video from <spoc_dir>/{train|test}/<sequence>/videos/rgb_trajectory.mp4
      - reads poses  .npy  from <spoc_dir>/{train|test}/<sequence>/pose/frame_*_pose.npy
        (OpenGL c2w) → converts to OpenCV w2c, caches as (T,18): [fx,fy,cx,cy,H,W] + vec(E)
      - metadata list saved to <save_dir>/metadata/{train|test}.pt
      - returns video (T,C,H,W) in [0,1] and cond processed to (T,16) in Advanced class
    """

    _ALL_SPLITS = ["training", "validation", "test"]

    @staticmethod
    def _alias_split(split: SPLIT) -> str:
        if split in ("train", "training"):
            return "train"
        if split in ("validation", "test"):
            return "test"
        raise ValueError(f"Unknown split: {split}")

    def __init__(self, cfg: DictConfig, split: SPLIT = "training", *args, **kwargs):
        # ensure save_dir
        if not cfg.get("save_dir", None):
            cfg.save_dir = str(Path("data/spoc").resolve())

        self.spoc_dir = Path(cfg.get("spoc_dir", cfg.get("SPOC_DIR", "")))
        if not self.spoc_dir.exists():
            raise FileNotFoundError(f"spoc_dir not found: {self.spoc_dir}")

        self.anchor_to_first: bool = bool(cfg.get("anchor_to_first", False))
        self.resolution = int(cfg.get("resolution", 256))

        split_alias = self._alias_split(self.split)
        meta_path = self.metadata_dir / f"{split_alias}.pt"
        if not meta_path.exists():
            self.build_metadata(split_alias)
          
        super().__init__(cfg, split, *args, **kwargs)

        self.metadata = self.load_metadata()

    # no downloader
    def _should_download(self) -> bool: return False
    def download_dataset(self) -> None:
        raise RuntimeError("SPOC loader does not download. Provide local spoc_dir.")

    def setup(self) -> None:
        self.transform = lambda x: x

    # ---- metadata/build ----
    def _list_sequences(self, split_alias: str) -> List[Path]:
        root = (self.spoc_dir / split_alias).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Missing split folder: {root}")
        # train typically: obj_nav_type_* ; test: house_*
        seqs = [d for d in root.iterdir() if d.is_dir()]
        return sorted(seqs, key=lambda p: _natural_key(p.name))

    def build_metadata(self, split: SPLIT) -> None:
        split_alias = self._alias_split(split)
        root = (self.spoc_dir / split_alias).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Missing split folder: {root}")

        print(cyan(f"[SPOC] Scanning sequences under {root} ..."))
        pose_out_dir = (self.save_dir / f"{split_alias}_poses").resolve()
        pose_out_dir.mkdir(parents=True, exist_ok=True)
        print(cyan(f"[SPOC] Pose tensors will be saved under {pose_out_dir}"))

        metadata: List[Dict[str, Any]] = []

        for seq_dir in self._list_sequences(split_alias):
            videos_dir = seq_dir / "videos"
            pose_dir = seq_dir / "pose"
            if not (videos_dir.is_dir() and pose_dir.is_dir()):
                continue

            mp4_path = videos_dir / "rgb_trajectory.mp4"
            if not mp4_path.is_file():
                continue
            pose_files = sorted(pose_dir.glob("*.npy"), key=_natural_key)
            if not pose_files:
                continue

            # video length + image size from mp4
            H0, W0, n_video = _read_mp4_info(mp4_path)
            kept = min(n_video, len(pose_files))
            if kept <= 0:
                continue

            # SPOC normalized intrinsics → pixels
            K0 = _load_spoc_intrinsics_as_pixels((H0, W0))
            # scale+center-crop to (res,res) and adjust K
            s, tx, ty = _compute_scale_crop_params(H0, W0, self.resolution)
            K = _adjust_K_for_scale_crop(K0, s, tx, ty)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            H = W = float(self.resolution)

            # anchor to first (OpenGL)
            first_pose_path = pose_files[0]
            first_pose = _to_4x4(np.load(first_pose_path, allow_pickle=False))
            anchor = (first_pose[:3, :3], first_pose[:3, 3]) if self.anchor_to_first else None

            rows = []
            for i in range(kept):
                c2w = _to_4x4(np.load(pose_files[i], allow_pickle=False))
                E = _opengl_c2w_to_opencv_w2c_3x4(c2w, anchor)
                e_flat = E.reshape(3, 4).flatten()
                rows.append(np.concatenate([[fx, fy, cx, cy, H, W], e_flat], axis=0))
            cams = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)  # (T,18)

            key = seq_dir.name  # one "video" per sequence
            out_pose_path = (pose_out_dir / f"{key}.pt").resolve()
            safe_torch_save(cams, out_pose_path)

            metadata.append({
                "video_paths": seq_dir.resolve(),  # absolute path to sequence root
                "length": kept,
                "key": key,
                "n_video_frames": int(n_video),
            })

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        out_meta_path = (self.metadata_dir / f"{split_alias}.pt").resolve()
        safe_torch_save(metadata, out_meta_path)
        print(cyan(f"[SPOC] Wrote metadata: {out_meta_path}  (n_seqs={len(metadata)})"))

    def load_metadata(self) -> List[Dict[str, Any]]:
        split_alias = self._alias_split(self.split)
        path = self.metadata_dir / f"{split_alias}.pt"
        print(cyan(f"[SPOC] Loading metadata from {path}"))
        meta = torch.load(path, weights_only=False)
        if isinstance(meta, list):
            return meta
        raise RuntimeError(f"Unexpected metadata format at {path}: {type(meta)}")

    # ---- dataset api required by BaseVideoDataset ----
    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return int(video_metadata["length"])

    def build_transform(self):
        return (lambda x: x)

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load RGB frames from videos/rgb_trajectory.mp4, resize+center-crop to (res,res),
        return float (T,C,H,W) in [0,1].
        """
        seq_dir: Path = Path(video_metadata["video_paths"])
        mp4_path = seq_dir / "videos" / "rgb_trajectory.mp4"
        if not mp4_path.is_file():
            raise FileNotFoundError(f"Missing video file: {mp4_path}")
 
        n_frames = int(video_metadata.get("n_video_frames", -1))
        if n_frames <= 0:
            _, _, n_frames = _read_mp4_info(mp4_path)

        if end_frame is None:
            end_frame = n_frames
        assert 0 <= start_frame < end_frame <= n_frames, "Invalid frame slice"

        imgs_np = _read_mp4_rgb_uint8(mp4_path, start_frame, end_frame)
        video_thwc_torch = torch.from_numpy(np.stack(imgs_np, axis=0))         # (T,H,W,C), uint8
        video_thwc_np = rescale_and_crop(video_thwc_torch, int(self.resolution))  # THWC uint8 (numpy)
        video = torch.from_numpy(video_thwc_np).permute(0, 3, 1, 2).float() / 255.0
        return video

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load cached (T,18) camera tensors for this sequence from <save_dir>/{train|test}_poses/<key>.pt
        """
        split_alias = self._alias_split(self.split)
        key = video_metadata["key"]
        pose_path = (self.save_dir / f"{split_alias}_poses" / f"{key}.pt").resolve()
        cams: torch.Tensor = torch.load(pose_path, weights_only=False)
        if end_frame is None:
            end_frame = cams.shape[0]
        assert 0 <= start_frame < end_frame <= cams.shape[0], "Invalid cond slice"
        return cams[start_frame:end_frame]

    # ---- latent path helpers ----
    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        split_alias = self._alias_split(self.split)
        out_dir = (self.latent_dir / split_alias)
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{video_metadata['key']}").with_suffix(".pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        split_alias = self._alias_split(split)
        base = (self.latent_dir / split_alias)
        base.mkdir(parents=True, exist_ok=True)
        return sorted(base.glob("*.pt"), key=str)

class SpocSimpleVideoDataset(SpocBaseVideoDataset, BaseSimpleVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "train"):
        super().__init__(cfg, split)
        self.setup()

    def setup(self) -> None:
        return

class SpocAdvancedVideoDataset(SpocBaseVideoDataset, BaseAdvancedVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "training", current_epoch: Optional[int] = None):
        if split == "validation":
            split = "test"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        self.spoc_dir = Path(cfg.get("spoc_dir", cfg.get("SPOC_DIR", "")))
        self.anchor_to_first = bool(cfg.get("anchor_to_first", False))
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert self.current_subepoch is not None, "Subepoch required for frame skip schedule"
        return self.frame_skip + int(self.current_subepoch * self.augmentation.frame_skip_increase)

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def _process_external_cond(self, external_cond: torch.Tensor, frame_skip: Optional[int] = None) -> torch.Tensor:
        # external_cond: (T,18) -> keep [fx,fy,cx,cy] + 12*E -> (T,16)
        poses = external_cond[:: frame_skip or self.frame_skip]
        return torch.cat([poses[:, :4], poses[:, 6:]], dim=-1).to(torch.float32)

    def _augment(self, video: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # cond columns: [fx,fy,cx,cy, r00,r01,r02,t0, r10,r11,r12,t1, r20,r21,r22,t2]
            cond[:, [5, 6, 7, 8, 12]] *= -1
        if random_bool(self.augmentation.back_and_forth_prob):
            video, cond = map(lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(), (video, cond))
        if random_bool(self.augmentation.reverse_prob):
            video, cond = map(lambda x: x.flip(0).contiguous(), (video, cond))
        return video, cond

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split != "training":
            return super().__getitem__(idx)

        video_idx, start_frame = self.get_clip_location(idx)
        meta = self.metadata[video_idx]
        V = self.video_length(meta)

        frame_skip = (V - start_frame - 1) // (self.cfg.max_frames - 1)
        frame_skip = min(frame_skip, self._training_frame_skip) if self.split == "training" else np.random.randint(self.frame_skip, frame_skip + 1)
        assert frame_skip > 0, f"Frame skip {frame_skip} must be > 0"

        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1
        video = self.load_video(meta, start_frame, end_frame)
        cond = self.load_cond(meta, start_frame, end_frame)
        assert len(video) == len(cond), "Video and cond length mismatch"

        video, cond = video[::frame_skip], self._process_external_cond(cond, frame_skip)

        video, cond = self._augment(video, cond)

        return {
            "videos": self.transform(video),                    # (T,C,H,W) float [0,1]
            "conds": cond,                                      # (T,16)
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }
