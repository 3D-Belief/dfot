# datasets/video/habitat_video_dataset.py
from __future__ import annotations
from typing import Tuple, List, Dict, Any, Literal, Optional
from pathlib import Path
import re
import numpy as np
import torch

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

# ----------------- helpers -----------------
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

def _load_habitat_intrinsics(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.shape == (3, 3):
        K = arr.astype(np.float64)
    elif arr.shape == (4, 4):
        K = arr[:3, :3].astype(np.float64)
    elif arr.shape == (4,):
        fx, fy, cx, cy = map(float, arr)
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    elif arr.shape == (1, 4):
        fx, fy, cx, cy = map(float, arr[0])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unrecognized intrinsics shape {arr.shape}")
    return K

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

def _read_npy_rgb_as_uint8(p: Path) -> np.ndarray:
    arr = np.load(p)
    if arr.dtype == np.uint8:
        return arr
    arr = np.asarray(arr)
    if arr.max() <= 1.5:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    return arr

# ----------------- dataset -----------------
VideoPreprocessingType = Literal["npz", "mp4"]

class HabitatBaseVideoDataset(BaseVideoDataset):
    """
    Habitat dataset that:
      - reads RGB frames from <habitat_dir>/{train|test}/<scene>/agent_1/<traj>/{rgb,pose}/*.npy
      - writes pose tensors to <save_dir>/{train|test}_poses/<key>.pt
      - writes metadata list to <save_dir>/metadata/{train|test}.pt
      - returns (T,C,H,W) float in [0,1] and cond (T,18) or processed shape in subclasses
    It cooperates with BaseVideoDataset: we set cfg.save_dir (if missing), override load/build metadata,
    and provide identity transform (we already resize+crop in load_video).
    """

    # Accept all names BaseVideoDataset might call us with
    _ALL_SPLITS = ["training", "validation", "test"]

    @staticmethod
    def _alias_split(split: SPLIT) -> str:
        if split in ("train", "training"):
            return "train"
        if split in ("validation", "test"):
            return "test"
        raise ValueError(f"Unknown split: {split}")

    # ---- lifecycle ---------------------------------------------------------
    def __init__(self, cfg: DictConfig, split: SPLIT = "training", *args, **kwargs):
        # Ensure save_dir exists on cfg before BaseVideoDataset reads it
        if not cfg.get("save_dir", None):
            cfg.save_dir = str(Path("data/habitat").resolve())

        # Validate habitat_dir early (kept for clearer error messages)
        self.habitat_dir = Path(cfg.get("habitat_dir", ""))
        if not self.habitat_dir.exists():
            raise FileNotFoundError(f"habitat_dir not found: {self.habitat_dir}")

        self.anchor_to_first: bool = bool(cfg.get("anchor_to_first", False))
        self.resolution = int(cfg.get("resolution", 256))

        # Let BaseVideoDataset set save_dir / metadata_dir / latent_dir consistently
        super().__init__(cfg, split, *args, **kwargs)

        # If our normalized split metadata doesn't exist yet, build it once
        split_alias = self._alias_split(self.split)
        meta_path = self.metadata_dir / f"{split_alias}.pt"
        if not meta_path.exists():
            self.build_metadata(split_alias)

        # Reload using our loader (list-of-dicts format)
        self.metadata = self.load_metadata()

    # Disable downloading
    def _should_download(self) -> bool: return False
    def download_dataset(self) -> None:
        raise RuntimeError("Habitat loader does not download. Provide local habitat_dir.")

    def setup(self) -> None:
        self.transform = lambda x: x

    # ---- metadata i/o ------------------------------------------------------

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build a list[dict]:
          {"video_paths": Path, "length": int, "key": str}
        and write it to <save_dir>/metadata/{train|test}.pt.
        Also cache per-sequence pose tensors under <save_dir>/{train|test}_poses.
        """
        split_alias = self._alias_split(split)             # 'train' or 'test'
        split_folder = split_alias                         # on-disk name

        habi_dir = Path("/scratch/tshu2/yyin34/projects/3d_belief/partnr-planner/data/trajectories/habelief")
        root = (habi_dir / split_folder).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Missing split folder: {root}")

        print(cyan(f"[Habitat] Scanning sequences under {root} ..."))
        pose_out_dir = (self.save_dir / f"{split_alias}_poses").resolve()
        pose_out_dir.mkdir(parents=True, exist_ok=True)
        print(cyan(f"[Habitat] Pose tensors will be saved under {pose_out_dir}"))

        metadata: List[Dict[str, Any]] = []
        # Each sequence = <scene>/agent_1/<traj> with rgb/ and pose/ subfolders
        for scene_dir in sorted([d for d in root.iterdir() if d.is_dir()], key=_natural_key):
            agent_dir = scene_dir / "agent_1"
            if not agent_dir.is_dir():
                continue
            for traj in sorted([d for d in agent_dir.iterdir() if d.is_dir()], key=_natural_key):
                rgb_dir = traj / "rgb"
                pose_dir = traj / "pose"
                if not (rgb_dir.is_dir() and pose_dir.is_dir()):
                    continue

                # Intersect rgb/pose frames by basename
                rgb_files = _list_sorted(rgb_dir, ".npy")
                pose_files = _list_sorted(pose_dir, ".npy")
                pose_basenames = {p.stem for p in pose_files}
                frames = [f for f in rgb_files if f.stem in pose_basenames]
                if not frames:
                    continue

                # Intrinsics at <scene>/agent_1/intrinsics.npy
                intr_path = agent_dir / "intrinsics.npy"
                if not intr_path.exists():
                    raise FileNotFoundError(f"Missing intrinsics: {intr_path}")

                # Compute resize/crop params to adjust K to (res,res)
                H0, W0 = np.load(frames[0]).shape[:2]
                s, tx, ty = _compute_scale_crop_params(H0, W0, self.resolution)
                K0 = _load_habitat_intrinsics(intr_path)
                K = _adjust_K_for_scale_crop(K0, s, tx, ty)
                fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
                H = W = float(self.resolution)

                # Anchor-to-first (optional) in OpenGL space before GL->CV and c2w->w2c
                first_pose = _to_4x4(np.load((pose_dir / f"{frames[0].stem}.npy"), allow_pickle=False))
                anchor = (first_pose[:3, :3], first_pose[:3, 3]) if self.anchor_to_first else None

                rows = []
                for f in frames:
                    c2w = _to_4x4(np.load(pose_dir / f"{f.stem}.npy", allow_pickle=False))
                    E = _opengl_c2w_to_opencv_w2c_3x4(c2w, anchor)    # (3,4)
                    e_flat = E.reshape(3, 4).flatten()
                    rows.append(np.concatenate([[fx, fy, cx, cy, H, W], e_flat], axis=0))
                cams = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)  # (T,18)

                # Unique key and outputs
                scene_name = scene_dir.name
                key = f"{scene_name}__{traj.name}"

                out_pose_path = (pose_out_dir / f"{key}.pt").resolve()
                safe_torch_save(cams, out_pose_path)

                metadata.append({
                    "video_paths": traj.resolve(),  # absolute path to the sequence root
                    "length": len(frames),
                    "key": key,
                })

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        out_meta_path = (self.metadata_dir / f"{split_alias}.pt").resolve()
        safe_torch_save(metadata, out_meta_path)
        print(cyan(f"[Habitat] Wrote metadata: {out_meta_path}  (n_seqs={len(metadata)})"))

    def load_metadata(self) -> List[Dict[str, Any]]:
        split_alias = self._alias_split(self.split)
        path = self.metadata_dir / f"{split_alias}.pt"
        print(cyan(f"[Habitat] Loading metadata from {path}"))
        meta = torch.load(path, weights_only=False)
        if isinstance(meta, list):
            return meta
        raise RuntimeError(f"Unexpected metadata format at {path}: {type(meta)}")

    # ---- dataset api required by BaseVideoDataset -------------------------

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return int(video_metadata["length"])

    def build_transform(self):
        # We already resize/crop inside load_video; keep identity to avoid double interpolation
        return (lambda x: x)

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load npy RGB frames (uint8), do DFoT-style resize+center-crop to square, return float (T,C,H,W) in [0,1].
        """
        seq_dir: Path = Path(video_metadata["video_paths"])
        rgb_dir = seq_dir / "rgb"
        frames = _list_sorted(rgb_dir, ".npy")

        if end_frame is None:
            end_frame = len(frames)
        assert 0 <= start_frame < end_frame <= len(frames), "Invalid frame slice"

        # THWC uint8 (numpy) -> torch uint8 (THWC) -> rescale/crop (returns numpy THWC) -> torch float (TCHW)
        imgs_np = [_read_npy_rgb_as_uint8(frames[i]) for i in range(start_frame, end_frame)]
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

    # ---- latent path helpers (because raw videos live outside save_dir) ----

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        BaseVideoDataset assumes raw video under save_dir and builds latents relative to it.
        Here raw videos are outside save_dir; instead, store latents as:
            <latent_dir>/<alias>/<key>.pt
        """
        split_alias = self._alias_split(self.split)
        out_dir = (self.latent_dir / split_alias)
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{video_metadata['key']}").with_suffix(".pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        split_alias = self._alias_split(split)
        base = (self.latent_dir / split_alias)
        base.mkdir(parents=True, exist_ok=True)
        return sorted(base.glob("*.pt"), key=str)

class HabitatSimpleVideoDataset(HabitatBaseVideoDataset, BaseSimpleVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "train"):
        super().__init__(cfg, split)   # <- DO NOT call BaseSimpleVideoDataset.__init__ directly
        self.setup()

    def setup(self) -> None:
        return

class HabitatAdvancedVideoDataset(HabitatBaseVideoDataset, BaseAdvancedVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "train", current_epoch: Optional[int] = None):
        if split == "validation":
            split = "test"
        if split == "training":
            split = "train"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        super().__init__(cfg, split, current_epoch=current_epoch)

    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert self.current_subepoch is not None
        return self.frame_skip + int(self.current_subepoch * self.augmentation.frame_skip_increase)

    def on_before_prepare_clips(self) -> None:
        return

    def _process_external_cond(self, external_cond: torch.Tensor, frame_skip: Optional[int] = None) -> torch.Tensor:
        poses = external_cond[:: frame_skip or self.frame_skip]
        return torch.cat([poses[:, :4], poses[:, 6:]], dim=-1).to(torch.float32)

class HabitatAdvancedVideoDataset(HabitatBaseVideoDataset, BaseAdvancedVideoDataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "training", current_epoch: Optional[int] = None):
        if split == "validation":
            split = "test"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
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
        # Mirror RealEstate10K: flip pixels, keep K, tweak E via fixed sign flips
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