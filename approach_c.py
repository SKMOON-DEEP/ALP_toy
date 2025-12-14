"""approach_c.py

Approach C: Few-shot Domain Shift Adaptation with Metadata-Based Auxiliary Tasks
------------------------------------------------------------------------------

This module implements a toy-yet-rigorous version of the method described in:

  - Bingqing Chen, Luca Bondi, Samarjit Das,
    "Learning to Adapt to Domain Shifts with Few-shot Samples in Anomalous Sound Detection",
    Proc. ICPR 2022 / arXiv:2204.01905.

Core algorithmic elements implemented:
  1) Classification-based anomaly detection using auxiliary tasks defined on metadata
     (e.g., section ID, speed level, noise ID, microphone position).
  2) Episodic training (ProtoNet-style) to match few-shot inference.
  3) Optional gradient-based meta-learning with Reptile by alternating across tasks.
  4) Inference-time adaptation using few-shot normal samples from the target domain.
  5) Anomaly scoring from prototype distances in the embedding space.

This file is designed to interoperate with `toy_dummy_data.py` and mirrors the
API/CLI style of approach_a.py and approach_b.py.

CLI example
-----------
  python approach_c.py --data_root /path/to/toy_dataset --output_dir ./out_c

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

# Dataset/metrics contract (toy)
try:
    from toy_dummy_data import (
        ClipMeta,
        ToyASDDataset,
        partial_roc_auc_score_binary,
        roc_auc_score_binary,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "approach_c.py expects toy_dummy_data.py to be importable in PYTHONPATH. "
        "Place approach_c.py next to toy_dummy_data.py or install it as a module."
    ) from e


# -----------------------------
# Torch imports (required)
# -----------------------------

def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for approach_c.py. Install it via https://pytorch.org/get-started/locally/."
        ) from e


def _import_torch_nn() -> Any:
    try:
        import torch.nn as nn  # type: ignore
        return nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import torch.nn.") from e


def _import_torch_f() -> Any:
    try:
        import torch.nn.functional as F  # type: ignore
        return F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import torch.nn.functional.") from e


def _has_cuda(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    d = (device or "").strip().lower()
    if d in ("", "auto"):
        torch = _import_torch()
        return "cuda" if _has_cuda(torch) else "cpu"
    if d in ("cpu", "cuda"):
        return d
    if d.startswith("cuda:"):
        return d
    raise ValueError(f"Unsupported device spec: {device!r}")


# -----------------------------
# Logging
# -----------------------------

def _setup_logger(verbosity: int) -> logging.Logger:
    logger = logging.getLogger("approach_c")
    if logger.handlers:
        return logger

    level = logging.INFO
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity >= 2:
        level = logging.DEBUG

    logger.setLevel(level)
    h = logging.StreamHandler(stream=sys.stdout)
    h.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


# -----------------------------
# Determinism
# -----------------------------

def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed python, numpy, torch. Optionally enforce deterministic torch ops."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _import_torch()
    torch.manual_seed(seed)
    if _has_cuda(torch):
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass


# -----------------------------
# Dataset compatibility helpers
# -----------------------------

def _get_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _dataset_metas(ds: Any) -> List[Any]:
    metas = _get_attr(ds, ["metas", "clips"], None)
    if metas is None:
        raise AttributeError("Dataset object has no 'metas' or 'clips' attribute.")
    return list(metas)


def _dataset_filter(
    ds: Any,
    *,
    machine_types: Optional[Sequence[str]] = None,
    section_ids: Optional[Sequence[int]] = None,
    split: Optional[str] = None,
    domain: Optional[str] = None,
    is_anomaly: Optional[bool] = None,
) -> List[Any]:
    if hasattr(ds, "filter"):
        kwargs: Dict[str, Any] = {}
        if machine_types is not None:
            kwargs["machine_types"] = list(machine_types)
        if section_ids is not None:
            kwargs["section_ids"] = list(section_ids)
        if split is not None:
            kwargs["split"] = split
        if domain is not None:
            kwargs["domain"] = domain
        if is_anomaly is not None:
            kwargs["is_anomaly"] = is_anomaly
        return list(ds.filter(**kwargs))

    metas = _dataset_metas(ds)
    out: List[Any] = []
    for m in metas:
        if machine_types is not None and str(_get_attr(m, ["machine_type", "sound_domain"], "")) not in set(machine_types):
            continue
        if section_ids is not None and int(_get_attr(m, ["section_id", "asset_id"], -1)) not in set(int(s) for s in section_ids):
            continue
        if split is not None and str(_get_attr(m, ["split"], "")) != str(split):
            continue
        if domain is not None and str(_get_attr(m, ["domain"], "")) != str(domain):
            continue
        if is_anomaly is not None and bool(_get_attr(m, ["is_anomaly"], False)) != bool(is_anomaly):
            continue
        out.append(m)
    return out


def _dataset_load_audio(ds: Any, meta: Any) -> Tuple[np.ndarray, int]:
    if hasattr(ds, "load_audio"):
        y, sr = ds.load_audio(meta)
        return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)
    if hasattr(ds, "load_wave"):
        y = ds.load_wave(meta)
        sr = int(_get_attr(meta, ["sr", "sample_rate"], _get_attr(ds, ["sample_rate", "sr"], 16000)))
        return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)
    raise AttributeError("Dataset object has neither load_audio(meta) nor load_wave(meta).")


def _clip_id(meta: Any) -> str:
    return str(_get_attr(meta, ["clip_id", "id"], ""))


def _rel_path(meta: Any) -> str:
    return str(_get_attr(meta, ["relative_path", "filepath", "path"], ""))


def _machine_type(meta: Any) -> str:
    return str(_get_attr(meta, ["machine_type", "sound_domain"], ""))


def _section_id(meta: Any) -> int:
    return int(_get_attr(meta, ["section_id", "asset_id"], -1))


def _domain(meta: Any) -> Optional[str]:
    d = _get_attr(meta, ["domain"], None)
    return None if d is None else str(d)


def _split(meta: Any) -> str:
    return str(_get_attr(meta, ["split"], ""))


def _is_anomaly(meta: Any) -> bool:
    return bool(_get_attr(meta, ["is_anomaly"], False))


def _label(meta: Any) -> Optional[str]:
    v = _get_attr(meta, ["label"], None)
    return None if v is None else str(v)


def _unique_sorted(values: Iterable[Any]) -> List[Any]:
    uniq = list(set(values))
    if not uniq:
        return []
    if all(isinstance(v, (int, np.integer)) for v in uniq):
        return sorted(int(v) for v in uniq)
    if all(isinstance(v, str) for v in uniq):
        return sorted(uniq)
    return sorted(uniq, key=lambda v: (type(v).__name__, str(v)))


# -----------------------------
# Metrics helpers
# -----------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score_binary(y_true.astype(np.float64), y_score.astype(np.float64)))


def _safe_pauc(y_true: np.ndarray, y_score: np.ndarray, *, max_fpr: float = 0.1) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return float(
        partial_roc_auc_score_binary(
            y_true.astype(np.float64),
            y_score.astype(np.float64),
            max_fpr=float(max_fpr),
            standardized=True,
        )
    )


# -----------------------------
# Log-mel feature extraction (numpy-only)
# -----------------------------

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if not (0.0 <= fmin < fmax <= sample_rate / 2.0 + 1e-6):
        raise ValueError(f"Invalid fmin/fmax: fmin={fmin}, fmax={fmax}, sr/2={sample_rate/2.0}")

    n_freqs = n_fft // 2 + 1

    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / float(sample_rate)).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    # Ensure strictly increasing bins to avoid zero-width filters
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = min(bins[i - 1] + 1, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for i in range(n_mels):
        left = int(bins[i])
        center = int(bins[i + 1])
        right = int(bins[i + 2])

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        if right <= left:
            continue

        if center > left:
            fb[i, left:center] = (np.arange(left, center, dtype=np.float64) - left) / max(center - left, 1.0)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right, dtype=np.float64)) / max(right - center, 1.0)

    # Slaney-style area normalization
    enorm = 2.0 / np.maximum(hz_pts[2 : n_mels + 2] - hz_pts[:n_mels], 1e-12)
    fb *= enorm[:, None]
    return fb.astype(dtype, copy=False)


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((1, frame_length), dtype=np.float32)

    if n <= frame_length:
        pad = frame_length - n
        x_padded = np.pad(x, (0, pad), mode="constant")
        return x_padded[None, :]

    n_frames = 1 + int(math.ceil((n - frame_length) / float(hop_length)))
    pad = (n_frames - 1) * hop_length + frame_length - n
    x_padded = np.pad(x, (0, pad), mode="constant")

    stride = x_padded.strides[0]
    shape = (n_frames, frame_length)
    strides = (hop_length * stride, stride)
    frames = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    return frames.copy()


def _stft_power(
    x: np.ndarray,
    *,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    if win_length <= 0 or hop_length <= 0 or n_fft <= 0:
        raise ValueError("n_fft/win_length/hop_length must be positive")
    if n_fft < win_length:
        raise ValueError("n_fft must be >= win_length")

    frames = _frame_signal(x, frame_length=win_length, hop_length=hop_length)
    win = np.hanning(win_length).astype(np.float32)
    frames = frames * win[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32, copy=False)
    return power.T


@dataclass(frozen=True)
class LogMelConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    fmin: float = 0.0
    fmax: Optional[float] = None
    eps: float = 1e-10

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.win_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("win_ms/hop_ms must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.fmax is not None and self.fmax <= 0:
            raise ValueError("fmax must be positive if set")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


class LogMelExtractor:
    """Compute log-mel spectrograms with optional per-clip normalization."""

    def __init__(self, cfg: LogMelConfig, *, per_clip_norm: bool = True) -> None:
        cfg.validate()
        self.cfg = cfg
        self.per_clip_norm = bool(per_clip_norm)
        self._fb: Optional[np.ndarray] = None
        self._fb_nfft: Optional[int] = None

    def _ensure_fb(self, n_fft: int) -> np.ndarray:
        if self._fb is not None and self._fb_nfft == int(n_fft):
            return self._fb
        sr = int(self.cfg.sample_rate)
        fmax = float(self.cfg.fmax) if self.cfg.fmax is not None else float(sr) / 2.0
        fb = _mel_filterbank(
            sample_rate=sr,
            n_fft=int(n_fft),
            n_mels=int(self.cfg.n_mels),
            fmin=float(self.cfg.fmin),
            fmax=fmax,
            dtype=np.float32,
        )
        self._fb = fb
        self._fb_nfft = int(n_fft)
        return fb

    def __call__(self, wave: np.ndarray, sr: int) -> np.ndarray:
        sr = int(sr)
        if sr != int(self.cfg.sample_rate):
            raise ValueError(f"LogMelExtractor expects {self.cfg.sample_rate} Hz audio, got {sr} Hz.")

        x = np.asarray(wave, dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        win_length = int(round(self.cfg.win_ms * sr / 1000.0))
        hop_length = int(round(self.cfg.hop_ms * sr / 1000.0))
        n_fft = int(self.cfg.n_fft)
        if win_length > n_fft:
            n_fft = 1 << int(math.ceil(math.log2(win_length)))
            n_fft = int(max(n_fft, win_length))

        fb = self._ensure_fb(n_fft)
        power = _stft_power(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        mel = np.matmul(fb, power)
        mel = np.maximum(mel, float(self.cfg.eps))
        log_mel = np.log(mel).astype(np.float32, copy=False)

        if self.per_clip_norm:
            mu = float(np.mean(log_mel))
            sd = float(np.std(log_mel))
            log_mel = (log_mel - mu) / max(sd, 1e-6)

        if not np.isfinite(log_mel).all():
            log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        return log_mel  # (n_mels, n_frames)


class FeatureCache:
    """Caches features per clip_id for efficiency (important for episodic training)."""

    def __init__(self, extractor: LogMelExtractor, dataset: Any, *, max_items: Optional[int] = None) -> None:
        self.extractor = extractor
        self.dataset = dataset
        self.max_items = None if max_items is None else int(max_items)
        if self.max_items is not None and self.max_items <= 0:
            # Treat non-positive as "no eviction" to avoid edge-case errors.
            self.max_items = None
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []

    def get(self, meta: Any) -> np.ndarray:
        cid = _clip_id(meta)
        if cid in self._cache:
            return self._cache[cid]

        wave, sr = _dataset_load_audio(self.dataset, meta)
        feat = self.extractor(wave, sr)
        feat = np.asarray(feat, dtype=np.float32)

        if self.max_items is not None and len(self._cache) >= int(self.max_items):
            # FIFO eviction for simplicity and determinism
            if self._order:
                old = self._order.pop(0)
                self._cache.pop(old, None)

        self._cache[cid] = feat
        self._order.append(cid)
        return feat


def pad_mels(mels: Sequence[np.ndarray], *, pad_value: float = 0.0) -> np.ndarray:
    """Pad variable-length mel arrays to a batch tensor (B, n_mels, T_max)."""
    if len(mels) == 0:
        raise ValueError("pad_mels: empty input")
    n_mels = int(mels[0].shape[0])
    t_max = max(int(m.shape[1]) for m in mels)
    out = np.full((len(mels), n_mels, t_max), float(pad_value), dtype=np.float32)
    for i, m in enumerate(mels):
        mm = np.asarray(m, dtype=np.float32)
        if mm.ndim != 2 or int(mm.shape[0]) != n_mels:
            raise ValueError(f"Mel shape mismatch at i={i}: expected (n_mels, T), got {mm.shape}")
        t = int(mm.shape[1])
        out[i, :, :t] = mm
    return out


# -----------------------------
# Auxiliary tasks
# -----------------------------

AuxTaskName = Literal[
    "speed_level",
    "load_level",
    "noise_id",
    "mic_position_id",
    "section_id",
]


def _get_operating_condition(meta: Any) -> Mapping[str, Any]:
    oc = _get_attr(meta, ["operating_condition"], None)
    if oc is None:
        return {}
    if isinstance(oc, dict):
        return oc
    if isinstance(oc, str):
        try:
            val = json.loads(oc)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    return {}


def get_task_label(meta: Any, task: AuxTaskName) -> int:
    if task == "speed_level":
        oc = _get_operating_condition(meta)
        v = oc.get("speed_level", _get_attr(meta, ["speed_level"], 0))
        return int(v)
    if task == "load_level":
        oc = _get_operating_condition(meta)
        v = oc.get("load_level", _get_attr(meta, ["load_level"], 0))
        return int(v)
    if task == "noise_id":
        return int(_get_attr(meta, ["noise_id"], 0))
    if task == "mic_position_id":
        return int(_get_attr(meta, ["mic_position_id"], 0))
    if task == "section_id":
        return int(_section_id(meta))
    raise ValueError(f"Unknown task: {task!r}")


def group_by_task_label(metas: Sequence[Any], task: AuxTaskName) -> Dict[int, List[Any]]:
    groups: Dict[int, List[Any]] = {}
    for m in metas:
        lab = get_task_label(m, task)
        groups.setdefault(int(lab), []).append(m)
    return groups


# -----------------------------
# Episodic sampling
# -----------------------------

@dataclass(frozen=True)
class EpisodeSpec:
    task: AuxTaskName
    n_way: int
    k_shot: int
    q_query: int


@dataclass
class Episode:
    task: AuxTaskName
    support: List[Any]
    support_y: np.ndarray  # (Ns,)
    query: List[Any]
    query_y: np.ndarray  # (Nq,)


class EpisodeSampler:
    """Samples ProtoNet episodes from metas for a given auxiliary task."""

    def __init__(self, rng: random.Random, *, max_attempts: int = 50) -> None:
        self.rng = rng
        self.max_attempts = int(max_attempts)

    def sample(self, metas: Sequence[Any], spec: EpisodeSpec) -> Optional[Episode]:
        if spec.n_way <= 0 or spec.k_shot <= 0 or spec.q_query <= 0:
            raise ValueError("EpisodeSpec n_way/k_shot/q_query must be positive")

        groups = group_by_task_label(metas, spec.task)
        labels_all = list(groups.keys())
        if len(labels_all) < 2:
            return None

        for _ in range(self.max_attempts):
            # Try decreasing q_query if needed
            for q in range(spec.q_query, 0, -1):
                eligible = [lab for lab in labels_all if len(groups[lab]) >= spec.k_shot + q]
                if len(eligible) < 2:
                    continue
                n_way_eff = int(min(spec.n_way, len(eligible)))
                if n_way_eff < 2:
                    continue

                chosen = self.rng.sample(eligible, n_way_eff)
                support: List[Any] = []
                support_y: List[int] = []
                query: List[Any] = []
                query_y: List[int] = []

                for ci, lab in enumerate(chosen):
                    items = groups[lab]
                    idx = list(range(len(items)))
                    self.rng.shuffle(idx)
                    sup_idx = idx[: spec.k_shot]
                    qry_idx = idx[spec.k_shot : spec.k_shot + q]

                    support.extend([items[j] for j in sup_idx])
                    support_y.extend([ci] * len(sup_idx))
                    query.extend([items[j] for j in qry_idx])
                    query_y.extend([ci] * len(qry_idx))

                if support and query:
                    return Episode(
                        task=spec.task,
                        support=support,
                        support_y=np.asarray(support_y, dtype=np.int64),
                        query=query,
                        query_y=np.asarray(query_y, dtype=np.int64),
                    )

        return None


# -----------------------------
# Model: Conv encoder -> embedding
# -----------------------------

def build_conv_encoder(
    *,
    embedding_dim: int,
    channels: Tuple[int, int, int] = (32, 64, 128),
    dropout: float = 0.1,
) -> Any:
    """Return a torch.nn.Module mapping (B,1,n_mels,T) -> (B,embedding_dim)."""
    torch = _import_torch()
    nn = _import_torch_nn()

    c1, c2, c3 = (int(channels[0]), int(channels[1]), int(channels[2]))

    class ConvEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.drop = nn.Dropout(p=float(dropout))
            self.proj = nn.Linear(c3, int(embedding_dim), bias=True)
            self.ln = nn.LayerNorm(int(embedding_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                raise ValueError(f"Expected input (B,1,n_mels,T), got shape {tuple(x.shape)}")
            h = self.net(x)
            h = h.view(h.shape[0], -1)
            h = self.drop(h)
            e = self.proj(h)
            e = self.ln(e)
            return e

    return ConvEncoder()


# -----------------------------
# ProtoNet computations
# -----------------------------

def prototypical_logits(emb_query: Any, prototypes: Any, *, temperature: float = 1.0) -> Any:
    """logits = -||q - p||^2 / temperature."""
    torch = _import_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = emb_query[:, None, :]
    p = prototypes[None, :, :]
    dist2 = torch.sum((q - p) ** 2, dim=-1)
    return -dist2 / float(temperature)


def compute_prototypes(emb_support: Any, y_support: Any, n_way: int) -> Any:
    """Compute prototypes as class means."""
    torch = _import_torch()
    D = int(emb_support.shape[1])
    protos = []
    for c in range(int(n_way)):
        mask = (y_support == c)
        if torch.sum(mask).item() <= 0:
            proto = torch.zeros((D,), device=emb_support.device, dtype=emb_support.dtype)
        else:
            proto = emb_support[mask].mean(dim=0)
        protos.append(proto)
    return torch.stack(protos, dim=0)


def episode_loss(
    model: Any,
    x_support: Any,
    y_support: Any,
    x_query: Any,
    y_query: Any,
    *,
    temperature: float,
) -> Any:
    torch = _import_torch()
    F = _import_torch_f()

    emb_s = model(x_support)
    emb_q = model(x_query)

    n_way = int(torch.unique(y_support).numel())
    protos = compute_prototypes(emb_s, y_support, n_way=n_way)
    logits = prototypical_logits(emb_q, protos, temperature=float(temperature))
    return F.cross_entropy(logits, y_query)


def score_from_embeddings(
    emb: Any,
    prototypes: Any,
    *,
    score_mode: Literal["min_dist", "nll"],
    temperature: float,
    eps: float = 1e-12,
) -> Any:
    """Compute anomaly score given embeddings and prototypes (no extra forward pass)."""
    torch = _import_torch()
    F = _import_torch_f()

    q = emb[:, None, :]
    p = prototypes[None, :, :]
    dist2 = torch.sum((q - p) ** 2, dim=-1)  # (B,C)
    min_dist2 = torch.min(dist2, dim=1).values

    C = int(prototypes.shape[0])
    if score_mode == "min_dist" or C < 2:
        return min_dist2

    logits = -dist2 / float(temperature)
    probs = F.softmax(logits, dim=1)
    maxp = torch.max(probs, dim=1).values
    return -torch.log(maxp.clamp(min=float(eps)))


# -----------------------------
# Training and inference pipeline
# -----------------------------

TrainMode = Literal["protonet", "reptile"]
AdaptMode = Literal["prototypes", "finetune", "none"]
ScoreMode = Literal["min_dist", "nll"]
TaskAgg = Literal["mean", "max", "sum"]


@dataclass(frozen=True)
class MetaProtoConfig:
    # Feature extraction
    sample_rate: int = 16000
    n_mels: int = 128
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    per_clip_norm: bool = True

    # Model
    embedding_dim: int = 256
    dropout: float = 0.1

    # Tasks (include section_id by default for robustness)
    tasks: Tuple[AuxTaskName, ...] = ("section_id", "speed_level", "noise_id", "mic_position_id")

    # Episode spec (train)
    n_way: int = 4
    k_shot: int = 3
    q_query: int = 5

    # Training
    train_mode: TrainMode = "reptile"
    train_steps: int = 800  # protonet steps or reptile outer steps
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0

    # Reptile
    inner_steps: int = 5
    inner_lr: float = 1e-3
    meta_lr: float = 1e-2

    # Adaptation
    adapt_mode: AdaptMode = "prototypes"
    adapt_steps: int = 10
    adapt_lr: float = 5e-4
    adapt_k_shot: int = 1
    adapt_q_query: int = 1

    # Scoring
    score_mode: ScoreMode = "min_dist"
    temperature: float = 1.0
    task_agg: TaskAgg = "mean"

    # Performance
    batch_size: int = 16
    cache_features: bool = True
    cache_max_items: Optional[int] = None

    # Repro
    seed: int = 0
    deterministic: bool = False

    # Device
    device: str = "auto"

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.win_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("win_ms/hop_ms must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.n_way <= 0 or self.k_shot <= 0 or self.q_query <= 0:
            raise ValueError("n_way/k_shot/q_query must be positive")
        if self.train_steps <= 0:
            raise ValueError("train_steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.train_mode not in ("protonet", "reptile"):
            raise ValueError("train_mode must be protonet|reptile")
        if self.train_mode == "reptile":
            if self.inner_steps <= 0:
                raise ValueError("inner_steps must be positive")
            if self.inner_lr <= 0:
                raise ValueError("inner_lr must be positive")
            if self.meta_lr <= 0:
                raise ValueError("meta_lr must be positive")
        if self.adapt_mode not in ("prototypes", "finetune", "none"):
            raise ValueError("adapt_mode must be prototypes|finetune|none")
        if self.adapt_mode == "finetune":
            if self.adapt_steps <= 0:
                raise ValueError("adapt_steps must be positive")
            if self.adapt_lr <= 0:
                raise ValueError("adapt_lr must be positive")
            if self.adapt_k_shot <= 0 or self.adapt_q_query <= 0:
                raise ValueError("adapt_k_shot/adapt_q_query must be positive")
        if self.score_mode not in ("min_dist", "nll"):
            raise ValueError("score_mode must be min_dist|nll")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.task_agg not in ("mean", "max", "sum"):
            raise ValueError("task_agg must be mean|max|sum")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class MetaProtoASD:
    """Approach C engine: train -> adapt -> score."""

    def __init__(self, cfg: MetaProtoConfig, logger: Optional[logging.Logger] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.logger = logger or logging.getLogger("approach_c")

        set_global_seed(cfg.seed, deterministic=cfg.deterministic)
        self.device = _resolve_device(cfg.device)

        self.feature_extractor = LogMelExtractor(
            LogMelConfig(
                sample_rate=int(cfg.sample_rate),
                n_mels=int(cfg.n_mels),
                win_ms=float(cfg.win_ms),
                hop_ms=float(cfg.hop_ms),
                n_fft=int(cfg.n_fft),
            ),
            per_clip_norm=bool(cfg.per_clip_norm),
        )

        self.model = build_conv_encoder(
            embedding_dim=int(cfg.embedding_dim),
            dropout=float(cfg.dropout),
        ).to(self.device)

        self.tasks: Tuple[AuxTaskName, ...] = tuple(cfg.tasks)

        self._cache: Optional[FeatureCache] = None
        self._cache_ds: Optional[Any] = None

    def _ensure_cache(self, dataset: Any) -> FeatureCache:
        if not bool(self.cfg.cache_features):
            raise RuntimeError("Feature caching is disabled, but _ensure_cache was called.")
        if self._cache is None or self._cache_ds is not dataset:
            self._cache = FeatureCache(
                extractor=self.feature_extractor,
                dataset=dataset,
                max_items=self.cfg.cache_max_items,
            )
            self._cache_ds = dataset
        return self._cache

    def _extract_feats(self, metas: Sequence[Any], dataset: Any) -> List[np.ndarray]:
        feats: List[np.ndarray] = []
        if bool(self.cfg.cache_features):
            cache = self._ensure_cache(dataset)
            for m in metas:
                feats.append(cache.get(m))
            return feats

        for m in metas:
            wave, sr = _dataset_load_audio(dataset, m)
            feats.append(self.feature_extractor(wave, sr))
        return feats

    def _feats_to_tensor(self, feats: Sequence[np.ndarray]) -> Any:
        torch = _import_torch()
        x = pad_mels(feats)  # (B, n_mels, T)
        xt = torch.from_numpy(x).to(self.device)
        return xt.unsqueeze(1)  # (B,1,n_mels,T)

    def train_on_source(self, source_train_metas: Sequence[Any], dataset: Any) -> None:
        """Train representation on source normal data."""
        torch = _import_torch()

        if len(source_train_metas) == 0:
            raise ValueError("source_train_metas is empty")

        rng = random.Random(int(self.cfg.seed))
        sampler = EpisodeSampler(rng, max_attempts=200)

        if self.cfg.train_mode == "protonet":
            opt = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.lr), weight_decay=float(self.cfg.weight_decay))
            self.model.train()
            steps = int(self.cfg.train_steps)

            updates = 0
            for step in range(steps):
                task = rng.choice(self.tasks)
                ep = sampler.sample(
                    source_train_metas,
                    EpisodeSpec(task=task, n_way=int(self.cfg.n_way), k_shot=int(self.cfg.k_shot), q_query=int(self.cfg.q_query)),
                )
                if ep is None:
                    continue

                feats_s = self._extract_feats(ep.support, dataset)
                feats_q = self._extract_feats(ep.query, dataset)

                x_s = self._feats_to_tensor(feats_s)
                x_q = self._feats_to_tensor(feats_q)

                y_s = torch.from_numpy(ep.support_y).to(self.device)
                y_q = torch.from_numpy(ep.query_y).to(self.device)

                opt.zero_grad(set_to_none=True)
                loss = episode_loss(self.model, x_s, y_s, x_q, y_q, temperature=float(self.cfg.temperature))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.grad_clip_norm))
                opt.step()
                updates += 1

                if self.logger and (step % 50 == 0 or step == steps - 1):
                    self.logger.info("ProtoNet step %d/%d task=%s loss=%.6f", step + 1, steps, task, float(loss.item()))

            if updates == 0 and self.logger:
                self.logger.warning("ProtoNet training performed 0 updates (insufficient class diversity for tasks?)")

        else:
            # Reptile meta-learning
            outer_steps = int(self.cfg.train_steps)
            inner_steps = int(self.cfg.inner_steps)
            meta_lr = float(self.cfg.meta_lr)
            inner_lr = float(self.cfg.inner_lr)

            self.model.train()

            outer_updates = 0
            for outer in range(outer_steps):
                task = rng.choice(self.tasks)

                inner_model = build_conv_encoder(
                    embedding_dim=int(self.cfg.embedding_dim),
                    dropout=float(self.cfg.dropout),
                ).to(self.device)
                inner_model.load_state_dict(self.model.state_dict(), strict=True)
                inner_model.train()

                inner_opt = torch.optim.Adam(inner_model.parameters(), lr=float(inner_lr), weight_decay=float(self.cfg.weight_decay))

                loss_acc = 0.0
                inner_done = 0

                for _ in range(inner_steps):
                    ep = sampler.sample(
                        source_train_metas,
                        EpisodeSpec(task=task, n_way=int(self.cfg.n_way), k_shot=int(self.cfg.k_shot), q_query=int(self.cfg.q_query)),
                    )
                    if ep is None:
                        continue

                    feats_s = self._extract_feats(ep.support, dataset)
                    feats_q = self._extract_feats(ep.query, dataset)
                    x_s = self._feats_to_tensor(feats_s)
                    x_q = self._feats_to_tensor(feats_q)
                    y_s = torch.from_numpy(ep.support_y).to(self.device)
                    y_q = torch.from_numpy(ep.query_y).to(self.device)

                    inner_opt.zero_grad(set_to_none=True)
                    loss = episode_loss(inner_model, x_s, y_s, x_q, y_q, temperature=float(self.cfg.temperature))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(inner_model.parameters(), max_norm=float(self.cfg.grad_clip_norm))
                    inner_opt.step()

                    loss_acc += float(loss.item())
                    inner_done += 1

                if inner_done == 0:
                    continue

                with torch.no_grad():
                    for p, p_i in zip(self.model.parameters(), inner_model.parameters()):
                        p.add_(p_i - p, alpha=float(meta_lr))

                outer_updates += 1
                if self.logger and (outer % 50 == 0 or outer == outer_steps - 1):
                    self.logger.info(
                        "Reptile outer %d/%d task=%s inner_updates=%d avg_loss=%.6f",
                        outer + 1,
                        outer_steps,
                        task,
                        inner_done,
                        loss_acc / max(inner_done, 1),
                    )

            if outer_updates == 0 and self.logger:
                self.logger.warning("Reptile training performed 0 outer updates (insufficient class diversity for tasks?)")

    def adapt_to_target(self, support_metas: Sequence[Any], dataset: Any) -> Any:
        """Return an adapted model according to cfg.adapt_mode."""
        torch = _import_torch()
        if self.cfg.adapt_mode in ("none", "prototypes"):
            return self.model
        if len(support_metas) == 0:
            return self.model

        adapted = build_conv_encoder(
            embedding_dim=int(self.cfg.embedding_dim),
            dropout=float(self.cfg.dropout),
        ).to(self.device)
        adapted.load_state_dict(self.model.state_dict(), strict=True)
        adapted.train()

        opt = torch.optim.Adam(adapted.parameters(), lr=float(self.cfg.adapt_lr), weight_decay=float(self.cfg.weight_decay))

        rng = random.Random(int(self.cfg.seed) + 12345)
        sampler = EpisodeSampler(rng, max_attempts=200)

        steps = int(self.cfg.adapt_steps)
        used = 0

        # Pre-extract all support features once (used in fallback mode)
        support_feats_all = self._extract_feats(support_metas, dataset)
        x_support_all = self._feats_to_tensor(support_feats_all)

        for _ in range(steps):
            task = rng.choice(self.tasks)

            # Try episodic adaptation if possible (support has enough samples)
            ep = sampler.sample(
                support_metas,
                EpisodeSpec(task=task, n_way=int(self.cfg.n_way), k_shot=int(self.cfg.adapt_k_shot), q_query=int(self.cfg.adapt_q_query)),
            )

            opt.zero_grad(set_to_none=True)

            if ep is not None:
                feats_s = self._extract_feats(ep.support, dataset)
                feats_q = self._extract_feats(ep.query, dataset)
                x_s = self._feats_to_tensor(feats_s)
                x_q = self._feats_to_tensor(feats_q)
                y_s = torch.from_numpy(ep.support_y).to(self.device)
                y_q = torch.from_numpy(ep.query_y).to(self.device)
                loss = episode_loss(adapted, x_s, y_s, x_q, y_q, temperature=float(self.cfg.temperature))
            else:
                # Fallback: full-batch ProtoNet classification on support itself.
                labels = np.array([get_task_label(m, task) for m in support_metas], dtype=np.int64)
                uniq = sorted(set(int(v) for v in labels.tolist()))
                if len(uniq) < 2:
                    continue
                label_to_ci = {lab: i for i, lab in enumerate(uniq)}
                y = torch.tensor([label_to_ci[int(v)] for v in labels], dtype=torch.long, device=self.device)

                emb = adapted(x_support_all)
                protos = compute_prototypes(emb, y, n_way=int(len(uniq)))
                logits = prototypical_logits(emb, protos, temperature=float(self.cfg.temperature))
                F = _import_torch_f()
                loss = F.cross_entropy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), max_norm=float(self.cfg.grad_clip_norm))
            opt.step()
            used += 1

        if self.logger:
            self.logger.info("Target adaptation: requested_steps=%d performed_steps=%d", steps, used)

        adapted.eval()
        return adapted

    def build_task_prototypes(self, model: Any, support_metas: Sequence[Any], dataset: Any) -> Dict[AuxTaskName, Any]:
        """Compute prototypes per configured task from support metas."""
        torch = _import_torch()
        if len(support_metas) == 0:
            raise ValueError("support_metas is empty")

        model.eval()
        feats = self._extract_feats(support_metas, dataset)
        x = self._feats_to_tensor(feats)

        with torch.no_grad():
            emb = model(x)  # (N,D)

        protos_by_task: Dict[AuxTaskName, Any] = {}
        for task in self.tasks:
            labels = np.array([get_task_label(m, task) for m in support_metas], dtype=np.int64)
            uniq = sorted(set(int(v) for v in labels.tolist()))
            label_to_ci = {lab: i for i, lab in enumerate(uniq)}
            y = torch.tensor([label_to_ci[int(v)] for v in labels], dtype=torch.long, device=self.device)

            C = int(len(uniq))
            protos = compute_prototypes(emb, y, n_way=C)
            protos_by_task[task] = protos

        return protos_by_task

    def score_clips(
        self,
        model: Any,
        protos_by_task: Mapping[AuxTaskName, Any],
        metas: Sequence[Any],
        dataset: Any,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Score metas and return (final_scores, per_task_scores)."""
        torch = _import_torch()
        if len(metas) == 0:
            return np.zeros((0,), dtype=np.float32), {}

        model.eval()
        bs = int(self.cfg.batch_size)

        final_scores: List[np.ndarray] = []
        per_task_scores: Dict[str, List[np.ndarray]] = {str(t): [] for t in protos_by_task.keys()}

        for i in range(0, len(metas), bs):
            batch = metas[i : i + bs]
            feats = self._extract_feats(batch, dataset)
            x = self._feats_to_tensor(feats)

            with torch.no_grad():
                emb = model(x)  # (B,D)

                task_scores_t: List[Any] = []
                for task, protos in protos_by_task.items():
                    sc = score_from_embeddings(
                        emb,
                        protos,
                        score_mode=str(self.cfg.score_mode),
                        temperature=float(self.cfg.temperature),
                    )
                    task_scores_t.append(sc)
                    per_task_scores[str(task)].append(sc.detach().cpu().float().numpy())

                if not task_scores_t:
                    raise RuntimeError("No tasks to score. Check tasks/prototypes.")

                if self.cfg.task_agg == "mean":
                    agg = torch.stack(task_scores_t, dim=0).mean(dim=0)
                elif self.cfg.task_agg == "max":
                    agg = torch.stack(task_scores_t, dim=0).max(dim=0).values
                else:
                    agg = torch.stack(task_scores_t, dim=0).sum(dim=0)

            final_scores.append(agg.detach().cpu().float().numpy())

        final = np.concatenate(final_scores, axis=0).astype(np.float32, copy=False)
        per_task = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in per_task_scores.items()}
        return final, per_task


# -----------------------------
# Output writers
# -----------------------------

def _write_scores_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# CLI
# -----------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Approach C: metadata auxiliary tasks + ProtoNet/Reptile few-shot adaptation ASD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Device / reproducibility
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--verbosity", type=int, default=1, help="0=warnings,1=info,2=debug")

    # Feature
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--win_ms", type=float, default=25.0)
    p.add_argument("--hop_ms", type=float, default=10.0)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--per_clip_norm", action=argparse.BooleanOptionalAction, default=True)

    # Model
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Tasks
    p.add_argument(
        "--tasks",
        type=str,
        default="section_id,speed_level,noise_id,mic_position_id",
        help="Comma-separated tasks: section_id,speed_level,load_level,noise_id,mic_position_id",
    )

    # Episode
    p.add_argument("--n_way", type=int, default=4)
    p.add_argument("--k_shot", type=int, default=3)
    p.add_argument("--q_query", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)

    # Training
    p.add_argument("--train_mode", type=str, default="reptile", choices=["protonet", "reptile"])
    p.add_argument("--train_steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)

    # Reptile
    p.add_argument("--inner_steps", type=int, default=5)
    p.add_argument("--inner_lr", type=float, default=1e-3)
    p.add_argument("--meta_lr", type=float, default=1e-2)

    # Adaptation
    p.add_argument("--adapt_mode", type=str, default="prototypes", choices=["prototypes", "finetune", "none"])
    p.add_argument("--adapt_steps", type=int, default=10)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_k_shot", type=int, default=1)
    p.add_argument("--adapt_q_query", type=int, default=1)

    # Scoring
    p.add_argument("--score_mode", type=str, default="min_dist", choices=["min_dist", "nll"])
    p.add_argument("--task_agg", type=str, default="mean", choices=["mean", "max", "sum"])
    p.add_argument("--batch_size", type=int, default=16)

    # Support selection per section
    p.add_argument("--support_domain", type=str, default="target", choices=["source", "target"])
    p.add_argument("--support_n", type=int, default=3, help="Few-shot support size per section. -1 for all.")
    p.add_argument("--support_sampling", type=str, default="random", choices=["first", "random"])
    p.add_argument("--support_seed", type=int, default=0)

    # Test selection
    p.add_argument("--test_domain", type=str, default="all", choices=["source", "target", "all"])

    # Metrics
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--no_eval", action="store_true")

    # Filters
    p.add_argument("--machine_types", type=str, default="", help="Comma-separated subset of machine types. Default: all.")
    p.add_argument("--section_ids", type=str, default="", help="Comma-separated subset of section ids. Default: all.")

    # Caching
    p.add_argument("--cache_features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cache_max_items", type=int, default=-1, help="-1 = unlimited (no eviction)")

    # Model checkpointing
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--load_models", action="store_true")

    return p.parse_args(argv)


def _select_support(
    metas: Sequence[Any],
    *,
    n: Optional[int],
    mode: Literal["first", "random"],
    seed: int,
) -> List[Any]:
    metas_sorted = sorted(metas, key=_clip_id)
    if n is None or n >= len(metas_sorted):
        return metas_sorted
    if n <= 0:
        raise ValueError("support_n must be positive when provided")
    if mode == "first":
        return metas_sorted[:n]
    rng = random.Random(int(seed))
    idx = list(range(len(metas_sorted)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [metas_sorted[i] for i in idx]


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logger = _setup_logger(int(args.verbosity))

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ToyASDDataset.load(data_root)
    metas_all = _dataset_metas(ds)

    # Machine types
    if str(args.machine_types).strip():
        mts = [m.strip() for m in str(args.machine_types).split(",") if m.strip()]
    else:
        mts = _unique_sorted([_machine_type(m) for m in metas_all])

    # Section ids filter
    user_sids_set: Optional[set[int]] = None
    if str(args.section_ids).strip():
        user_sids = [int(s.strip()) for s in str(args.section_ids).split(",") if s.strip()]
        user_sids_set = set(user_sids)

    sections_by_mt: Dict[str, List[int]] = {}
    for mt in mts:
        sids = _unique_sorted([_section_id(m) for m in metas_all if _machine_type(m) == mt])
        if user_sids_set is not None:
            sids = [sid for sid in sids if sid in user_sids_set]
        sections_by_mt[mt] = sids

    logger.info("Loaded dataset: root=%s machine_types=%s", str(data_root), mts)
    for mt in mts:
        logger.info("  - %s sections=%s", mt, sections_by_mt.get(mt, []))

    # Parse tasks
    raw_tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    allowed = {"section_id", "speed_level", "load_level", "noise_id", "mic_position_id"}
    for t in raw_tasks:
        if t not in allowed:
            raise ValueError(f"Unknown task name: {t!r}. Allowed: {sorted(allowed)}")
    tasks = tuple(raw_tasks)  # type: ignore[assignment]

    support_n = None if int(args.support_n) < 0 else int(args.support_n)

    cfg = MetaProtoConfig(
        sample_rate=int(args.sample_rate),
        n_mels=int(args.n_mels),
        win_ms=float(args.win_ms),
        hop_ms=float(args.hop_ms),
        n_fft=int(args.n_fft),
        per_clip_norm=bool(args.per_clip_norm),
        embedding_dim=int(args.embedding_dim),
        dropout=float(args.dropout),
        tasks=tasks,  # type: ignore[arg-type]
        n_way=int(args.n_way),
        k_shot=int(args.k_shot),
        q_query=int(args.q_query),
        train_mode=str(args.train_mode),  # type: ignore[arg-type]
        train_steps=int(args.train_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        inner_steps=int(args.inner_steps),
        inner_lr=float(args.inner_lr),
        meta_lr=float(args.meta_lr),
        adapt_mode=str(args.adapt_mode),  # type: ignore[arg-type]
        adapt_steps=int(args.adapt_steps),
        adapt_lr=float(args.adapt_lr),
        adapt_k_shot=int(args.adapt_k_shot),
        adapt_q_query=int(args.adapt_q_query),
        score_mode=str(args.score_mode),  # type: ignore[arg-type]
        temperature=float(args.temperature),
        task_agg=str(args.task_agg),  # type: ignore[arg-type]
        batch_size=int(args.batch_size),
        cache_features=bool(args.cache_features),
        cache_max_items=(None if int(args.cache_max_items) < 0 else int(args.cache_max_items)),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        device=str(args.device),
    )
    cfg.validate()

    _write_json(out_dir / "config.json", dataclasses.asdict(cfg))

    score_rows: List[Dict[str, Any]] = []
    section_metrics: List[Dict[str, Any]] = []

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for mt in mts:
        engine = MetaProtoASD(cfg, logger=logger)
        model_path = models_dir / f"model_{mt}.pt"

        if bool(args.load_models) and model_path.exists():
            torch = _import_torch()
            state = torch.load(str(model_path), map_location=_resolve_device(cfg.device))
            state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
            engine.model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded model for machine_type=%s from %s", mt, str(model_path))
        else:
            src_train = _dataset_filter(ds, machine_types=[mt], split="train", domain="source", is_anomaly=False)
            if len(src_train) == 0:
                logger.warning("No source train normals for machine_type=%s; skipping.", mt)
                continue
            logger.info("Training model for machine_type=%s on %d source normal clips", mt, len(src_train))
            engine.train_on_source(src_train, ds)

            if bool(args.save_models):
                torch = _import_torch()
                torch.save({"state_dict": engine.model.state_dict(), "config": dataclasses.asdict(cfg)}, str(model_path))
                logger.info("Saved model for machine_type=%s to %s", mt, str(model_path))

        # Evaluate per section
        for sid in sections_by_mt.get(mt, []):
            support_all = _dataset_filter(ds, machine_types=[mt], section_ids=[sid], split="train", domain=str(args.support_domain), is_anomaly=False)
            if len(support_all) == 0:
                logger.warning("No support clips for machine=%s section=%s domain=%s; skipping.", mt, sid, args.support_domain)
                continue
            support = _select_support(support_all, n=support_n, mode=str(args.support_sampling), seed=int(args.support_seed))

            if str(args.test_domain) == "all":
                test_domains = ["source", "target"]
            else:
                test_domains = [str(args.test_domain)]

            test_metas: List[Any] = []
            for d in test_domains:
                test_metas.extend(_dataset_filter(ds, machine_types=[mt], section_ids=[sid], split="test", domain=d))
            if len(test_metas) == 0:
                logger.warning("No test clips for machine=%s section=%s; skipping.", mt, sid)
                continue

            adapted_model = engine.adapt_to_target(support, ds)
            protos_by_task = engine.build_task_prototypes(adapted_model, support, ds)
            scores, per_task = engine.score_clips(adapted_model, protos_by_task, test_metas, ds)

            task_cols = sorted(per_task.keys())
            for i, m in enumerate(test_metas):
                row: Dict[str, Any] = {
                    "clip_id": _clip_id(m),
                    "relative_path": _rel_path(m),
                    "machine_type": _machine_type(m),
                    "section_id": int(_section_id(m)),
                    "split": _split(m),
                    "domain": "" if _domain(m) is None else str(_domain(m)),
                    "is_anomaly": int(_is_anomaly(m)),
                    "label": "" if _label(m) is None else str(_label(m)),
                    "score": float(scores[i]),
                    "train_mode": str(cfg.train_mode),
                    "adapt_mode": str(cfg.adapt_mode),
                    "score_mode": str(cfg.score_mode),
                    "task_agg": str(cfg.task_agg),
                    "tasks": ",".join(cfg.tasks),
                    "support_domain": str(args.support_domain),
                    "support_n": int(len(support)),
                }
                for t in task_cols:
                    row[f"score_{t}"] = float(per_task[t][i])
                score_rows.append(row)

            if not bool(args.no_eval):
                y_true_all = np.array([1 if _is_anomaly(m) else 0 for m in test_metas], dtype=np.int64)
                auc_all = _safe_auc(y_true_all, scores)
                pauc_all = _safe_pauc(y_true_all, scores, max_fpr=float(args.pauc_max_fpr))

                per_domain: Dict[str, Dict[str, Optional[float]]] = {}
                for d in test_domains:
                    idx = [i for i, m in enumerate(test_metas) if _domain(m) == d]
                    if not idx:
                        continue
                    y = y_true_all[idx]
                    sc = scores[idx]
                    per_domain[d] = {
                        "auc": _safe_auc(y, sc),
                        "pauc": _safe_pauc(y, sc, max_fpr=float(args.pauc_max_fpr)),
                        "n": int(len(idx)),
                        "n_anom": int(y.sum()),
                    }

                section_metrics.append(
                    {
                        "machine_type": mt,
                        "section_id": int(sid),
                        "support_domain": str(args.support_domain),
                        "support_n": int(len(support)),
                        "n_test": int(len(test_metas)),
                        "auc": auc_all,
                        "pauc": pauc_all,
                        "per_domain": per_domain,
                    }
                )

            logger.info(
                "Finished machine=%s section=%02d: support=%d test=%d",
                mt,
                int(sid),
                int(len(support)),
                int(len(test_metas)),
            )

    scores_csv = out_dir / "scores.csv"
    _write_scores_csv(scores_csv, score_rows)

    if not bool(args.no_eval):
        aucs = [m["auc"] for m in section_metrics if m.get("auc") is not None]
        paucs = [m["pauc"] for m in section_metrics if m.get("pauc") is not None]
        agg = {
            "n_sections": int(len(section_metrics)),
            "macro_auc_mean": float(np.mean(aucs)) if aucs else None,
            "macro_pauc_mean": float(np.mean(paucs)) if paucs else None,
            "pauc_max_fpr": float(args.pauc_max_fpr),
            "config": dataclasses.asdict(cfg),
        }
        _write_json(out_dir / "metrics.json", {"aggregate": agg, "sections": section_metrics})

    logger.info("Wrote: %s", str(scores_csv))
    if not bool(args.no_eval):
        logger.info("Wrote: %s", str(out_dir / "metrics.json"))

    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
