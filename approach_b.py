"""approach_b.py

Approach B: Domain-Generalized Memory Bank ASD (GenRep-style)
------------------------------------------------------------

This module implements the core algorithmic stack described in:

  - Phurich Saengthong, Takahiro Shinozaki,
    "Deep Generic Representations for Domain-Generalized Anomalous Sound Detection",
    arXiv:2409.05035 (2024).

GenRep-style pipeline (training-free on the ASD head):
  1) Extract clip embeddings using a frozen, generic audio representation model
     (original paper uses BEATs). This implementation provides:
       - A default, dependency-light encoder: LogMelStatsEncoder (numpy-only).
       - Optional SpeechBrain BEATs encoder if you have torch+speechbrain+checkpoint.
  2) Build a source normal memory bank M_s and a target normal memory bank M_t.
  3) (Optional) MemMixup: augment the target memory bank by interpolating each target
     embedding with its K nearest source embeddings using lambda=0.9.
  4) Score each query sample by its kNN mean squared distance to each bank:
       d_s(y), d_t(y).
  5) (Optional) Domain Normalization (DN): z-score normalize d_s and d_t using
     statistics computed on training normal samples and take the minimum:
       score(y) = min(z_s(y), z_t(y)).

The code is designed to work with `toy_dummy_data.py` (ToyASDDataset/ClipMeta) and
provides both a Python API and a CLI.

Usage (CLI):
  python approach_b.py --data_root /path/to/toy_dataset --output_dir ./out_b

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

# Dataset/metrics contract from toy_dummy_data.py
try:
    from toy_dummy_data import (
        ClipMeta,
        ToyASDDataset,
        partial_roc_auc_score_binary,
        roc_auc_score_binary,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "approach_b.py expects toy_dummy_data.py to be importable in PYTHONPATH. "
        "Place approach_b.py next to toy_dummy_data.py or install it as a module."
    ) from e


# -----------------------------
# Logging
# -----------------------------

def _setup_logger(verbosity: int) -> logging.Logger:
    logger = logging.getLogger("approach_b")
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
# Small utilities (dataset API compatibility)
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
        # We assume toy_dummy_data.ToyASDDataset.filter supports these keywords.
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

    # Fallback: manual filtering over ds.metas
    metas = _dataset_metas(ds)
    out: List[Any] = []
    for m in metas:
        if machine_types is not None and _get_attr(m, ["machine_type", "sound_domain"], None) not in set(machine_types):
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
        # Try to get sr from dataset or meta
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


def _harmonic_mean(vals: Sequence[float], eps: float = 1e-12) -> float:
    v = [float(x) for x in vals if x is not None and float(x) > 0.0]
    if not v:
        return float("nan")
    return float(len(v) / max(sum((1.0 / max(x, eps)) for x in v), eps))


def _unique_sorted(values: Iterable[Any]) -> List[Any]:
    uniq = list(set(values))
    if not uniq:
        return []
    if all(isinstance(v, (int, np.integer)) for v in uniq):
        return sorted(int(v) for v in uniq)
    if all(isinstance(v, str) for v in uniq):
        return sorted(uniq)
    return sorted(uniq, key=lambda v: (type(v).__name__, str(v)))


def _select_subset(
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
        raise ValueError("n must be positive when provided")
    if mode == "first":
        return metas_sorted[:n]
    rng = random.Random(int(seed))
    idx = list(range(len(metas_sorted)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [metas_sorted[i] for i in idx]


# -----------------------------
# Feature extraction: log-mel (numpy-only)
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
    """Create a mel filter bank matrix of shape (n_mels, n_fft//2+1)."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if not (0.0 <= fmin < fmax <= sample_rate / 2.0 + 1e-6):
        raise ValueError(f"Invalid fmin/fmax: fmin={fmin}, fmax={fmax}, sr/2={sample_rate/2.0}")

    n_freqs = n_fft // 2 + 1
    # Mel points
    m_min = _hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    m_pts = np.linspace(m_min, m_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / float(sample_rate)).astype(int)

    # Ensure strictly increasing bins to avoid zero-width triangles
    bins = np.clip(bins, 0, n_freqs - 1)
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

        # Rising edge
        if center > left:
            fb[i, left:center] = (np.arange(left, center, dtype=np.float64) - left) / max(center - left, 1.0)
        # Falling edge
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right, dtype=np.float64)) / max(right - center, 1.0)

    # Slaney-style area normalization: normalize each filter by its bandwidth.
    enorm = 2.0 / np.maximum(hz_pts[2 : n_mels + 2] - hz_pts[:n_mels], 1e-12)
    fb *= enorm[:, None]

    return fb.astype(dtype, copy=False)


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frame a 1D signal into shape (n_frames, frame_length) with end-padding."""
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
    window: Literal["hann"] = "hann",
) -> np.ndarray:
    """Compute power spectrogram (freq_bins, time_frames) using numpy."""
    if win_length <= 0 or hop_length <= 0 or n_fft <= 0:
        raise ValueError("n_fft/win_length/hop_length must be positive")
    if n_fft < win_length:
        raise ValueError("n_fft must be >= win_length")

    frames = _frame_signal(x, frame_length=win_length, hop_length=hop_length)
    if window == "hann":
        win = np.hanning(win_length).astype(np.float32)
    else:
        raise ValueError(f"Unsupported window: {window!r}")

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


def compute_log_mel(x: np.ndarray, cfg: LogMelConfig, *, mel_fb: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute log-mel spectrogram: shape (n_mels, n_frames) float32."""
    cfg.validate()
    sr = int(cfg.sample_rate)
    fmax = float(cfg.fmax) if cfg.fmax is not None else float(sr) / 2.0

    win_length = int(round(cfg.win_ms * sr / 1000.0))
    hop_length = int(round(cfg.hop_ms * sr / 1000.0))
    n_fft = int(cfg.n_fft)

    if win_length > n_fft:
        n_fft = 1 << (int(math.ceil(math.log2(win_length))))
        n_fft = int(max(n_fft, win_length))

    power = _stft_power(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window="hann")
    if mel_fb is None:
        mel_fb = _mel_filterbank(
            sample_rate=sr,
            n_fft=n_fft,
            n_mels=int(cfg.n_mels),
            fmin=float(cfg.fmin),
            fmax=fmax,
            dtype=np.float32,
        )
    if mel_fb.shape[1] != power.shape[0]:
        raise RuntimeError(f"Mel filterbank shape mismatch: fb {mel_fb.shape} vs power {power.shape}")

    mel = np.matmul(mel_fb, power)
    mel = np.maximum(mel, float(cfg.eps))
    log_mel = np.log(mel).astype(np.float32, copy=False)
    if not np.isfinite(log_mel).all():
        log_mel = np.nan_to_num(log_mel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return log_mel


# -----------------------------
# Encoders
# -----------------------------

class ClipEncoderBase:
    """Interface for clip-level encoders producing a single embedding per clip."""

    def encode(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> np.ndarray:
        """Return embeddings of shape (B, D) float32."""
        raise NotImplementedError

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def expected_sample_rate(self) -> int:
        return 16000


@dataclass(frozen=True)
class LogMelStatsEncoderConfig:
    mel: LogMelConfig = LogMelConfig()
    include_std: bool = True
    per_clip_norm: bool = True


class LogMelStatsEncoder(ClipEncoderBase):
    """Deterministic encoder based on log-mel statistics (numpy-only)."""

    def __init__(self, cfg: LogMelStatsEncoderConfig) -> None:
        cfg.mel.validate()
        self.cfg = cfg
        self._fb_cache: Optional[np.ndarray] = None
        self._fb_cache_nfft: Optional[int] = None

    @property
    def expected_sample_rate(self) -> int:
        return int(self.cfg.mel.sample_rate)

    @property
    def embedding_dim(self) -> int:
        return int(self.cfg.mel.n_mels) * (2 if self.cfg.include_std else 1)

    def _get_mel_fb(self, n_fft: int) -> np.ndarray:
        if self._fb_cache is not None and self._fb_cache_nfft == int(n_fft):
            return self._fb_cache
        fmax = float(self.cfg.mel.fmax) if self.cfg.mel.fmax is not None else float(self.cfg.mel.sample_rate) / 2.0
        fb = _mel_filterbank(
            sample_rate=int(self.cfg.mel.sample_rate),
            n_fft=int(n_fft),
            n_mels=int(self.cfg.mel.n_mels),
            fmin=float(self.cfg.mel.fmin),
            fmax=fmax,
            dtype=np.float32,
        )
        self._fb_cache = fb
        self._fb_cache_nfft = int(n_fft)
        return fb

    def encode(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> np.ndarray:
        sr = int(sampling_rate)
        if sr != int(self.cfg.mel.sample_rate):
            raise ValueError(f"LogMelStatsEncoder expects {self.cfg.mel.sample_rate} Hz audio, got {sr}.")

        B = len(waveforms)
        D = int(self.embedding_dim)
        out = np.zeros((B, D), dtype=np.float32)

        win_length = int(round(self.cfg.mel.win_ms * sr / 1000.0))
        n_fft = int(self.cfg.mel.n_fft)
        if win_length > n_fft:
            n_fft = 1 << (int(math.ceil(math.log2(win_length))))
            n_fft = int(max(n_fft, win_length))
        fb = self._get_mel_fb(n_fft)

        mel_cfg = dataclasses.replace(self.cfg.mel, n_fft=n_fft)

        for i, w in enumerate(waveforms):
            x = np.asarray(w, dtype=np.float32).reshape(-1)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            mel = compute_log_mel(x, mel_cfg, mel_fb=fb)
            if self.cfg.per_clip_norm:
                mu = float(np.mean(mel))
                sd = float(np.std(mel))
                mel = (mel - mu) / max(sd, 1e-6)

            mean = mel.mean(axis=1)
            if self.cfg.include_std:
                std = mel.std(axis=1)
                emb = np.concatenate([mean, std], axis=0)
            else:
                emb = mean

            out[i, :] = emb.astype(np.float32, copy=False)

        return out


# Optional: SpeechBrain BEATs encoder (torch+speechbrain)
def _import_torch() -> Any:
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for BEATs encoder. Install it via https://pytorch.org/get-started/locally/."
        ) from e


def _import_speechbrain_beats() -> Any:
    try:
        from speechbrain.lobes.models.beats import BEATs  # type: ignore
        return BEATs
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "speechbrain is required for the 'beats_speechbrain' encoder. "
            "Install it via: pip install speechbrain"
        ) from e


@dataclass(frozen=True)
class SpeechBrainBEATsEncoderConfig:
    checkpoint_path: str
    device: str = "auto"
    output_all_hiddens: bool = False
    layer: int = -1
    fbank_mean: float = 15.41663
    fbank_std: float = 6.55582
    freeze: bool = True

    def validate(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if self.fbank_std <= 0:
            raise ValueError("fbank_std must be positive")


def _resolve_device(device: str) -> str:
    device = (device or "").strip().lower()
    if device in ("", "auto"):
        torch = _import_torch()
        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    if device in ("cpu", "cuda"):
        return device
    if device.startswith("cuda:"):
        return device
    raise ValueError(f"Unsupported device: {device!r}")


class SpeechBrainBEATsEncoder(ClipEncoderBase):
    """BEATs encoder wrapper using SpeechBrain's BEATs implementation."""

    def __init__(self, cfg: SpeechBrainBEATsEncoderConfig, logger: Optional[logging.Logger] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.logger = logger or logging.getLogger("approach_b")

        torch = _import_torch()
        BEATs = _import_speechbrain_beats()

        self.device = _resolve_device(cfg.device)
        self.model = BEATs(
            ckp_path=str(cfg.checkpoint_path),
            freeze=bool(cfg.freeze),
            output_all_hiddens=bool(cfg.output_all_hiddens),
        )
        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim: int = 768

        if self.logger:
            self.logger.info(
                "Loaded SpeechBrain BEATs: ckpt=%s, device=%s, output_all_hiddens=%s, layer=%s",
                str(cfg.checkpoint_path),
                self.device,
                str(cfg.output_all_hiddens),
                str(cfg.layer),
            )

        try:
            wav = torch.zeros((1, 16000), dtype=torch.float32, device=self.device)
            wav_lens = torch.ones((1,), dtype=torch.float32, device=self.device)
            out = self.model.extract_features(
                wav,
                wav_lens,
                fbank_mean=float(cfg.fbank_mean),
                fbank_std=float(cfg.fbank_std),
            )
            feats = out[0] if isinstance(out, (tuple, list)) else out
            if feats.ndim == 4:
                feats = feats[int(cfg.layer)]
            if feats.ndim == 3:
                self._embedding_dim = int(feats.shape[-1])
        except Exception:
            pass

    @property
    def expected_sample_rate(self) -> int:
        return 16000

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_dim)

    def encode(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> np.ndarray:
        sr = int(sampling_rate)
        if sr != 16000:
            raise ValueError(f"SpeechBrainBEATsEncoder expects 16000 Hz audio, got {sr}.")

        torch = _import_torch()

        waves: List[np.ndarray] = []
        for w in waveforms:
            x = np.asarray(w, dtype=np.float32).reshape(-1)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            waves.append(x)

        max_len = max(int(x.shape[0]) for x in waves)
        B = len(waves)
        batch = np.zeros((B, max_len), dtype=np.float32)
        wav_lens = np.zeros((B,), dtype=np.float32)
        for i, x in enumerate(waves):
            n = int(x.shape[0])
            batch[i, :n] = x
            wav_lens[i] = float(n) / float(max_len)

        wav_t = torch.from_numpy(batch).to(self.device)
        lens_t = torch.from_numpy(wav_lens).to(self.device)

        with torch.no_grad():
            out = self.model.extract_features(
                wav_t,
                lens_t,
                fbank_mean=float(self.cfg.fbank_mean),
                fbank_std=float(self.cfg.fbank_std),
            )
        feats = out[0] if isinstance(out, (tuple, list)) else out

        if feats.ndim == 4:
            feats = feats[int(self.cfg.layer)]

        if feats.ndim != 3:
            raise RuntimeError(f"Unexpected BEATs feature shape: {tuple(feats.shape)}")

        emb = feats.mean(dim=1)
        emb_np = emb.detach().cpu().float().numpy().astype(np.float32, copy=False)
        return emb_np


# -----------------------------
# kNN utilities (numpy)
# -----------------------------

def _pairwise_sqdist_chunked(
    q: np.ndarray, x: np.ndarray, *, chunk_size: int
) -> Iterable[Tuple[int, int, np.ndarray]]:
    """Yield (start, end, dist2_chunk) where dist2_chunk has shape (B, C)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    q = np.asarray(q, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if q.ndim != 2 or x.ndim != 2:
        raise ValueError("q and x must be 2D arrays")
    if q.shape[1] != x.shape[1]:
        raise ValueError(f"Dim mismatch: q D={q.shape[1]} vs x D={x.shape[1]}")

    B = q.shape[0]
    N = x.shape[0]
    q_norm2 = np.sum(q * q, axis=1, keepdims=True)
    x_norm2_all = np.sum(x * x, axis=1)

    for start in range(0, N, int(chunk_size)):
        end = min(N, start + int(chunk_size))
        xc = x[start:end]
        dot = np.matmul(q, xc.T)
        dist2 = q_norm2 + x_norm2_all[start:end][None, :] - 2.0 * dot
        dist2 = np.maximum(dist2, 0.0).astype(np.float32, copy=False)
        yield start, end, dist2


def knn_topk_sqdist(
    q: np.ndarray,
    bank: np.ndarray,
    *,
    k: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Return the k smallest squared distances from each query to the bank."""
    q = np.asarray(q, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)

    if q.ndim != 2 or bank.ndim != 2:
        raise ValueError("q and bank must be 2D arrays")
    if bank.shape[0] <= 0:
        raise ValueError("bank is empty")
    if k <= 0:
        raise ValueError("k must be positive")

    k_eff = int(min(k, bank.shape[0]))
    B = q.shape[0]

    best = np.full((B, k_eff), np.inf, dtype=np.float32)

    for _, _, dist2 in _pairwise_sqdist_chunked(q, bank, chunk_size=int(chunk_size)):
        if dist2.shape[1] <= k_eff:
            cand = dist2
        else:
            idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            cand = np.take_along_axis(dist2, idx, axis=1)
        merged = np.concatenate([best, cand], axis=1)
        idx2 = np.argpartition(merged, kth=k_eff - 1, axis=1)[:, :k_eff]
        best = np.take_along_axis(merged, idx2, axis=1)

    return best


def knn_topk_indices(
    q: np.ndarray,
    bank: np.ndarray,
    *,
    k: int,
    chunk_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (topk_dist2, topk_indices) for each query."""
    q = np.asarray(q, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)

    if q.ndim != 2 or bank.ndim != 2:
        raise ValueError("q and bank must be 2D arrays")
    N = int(bank.shape[0])
    if N <= 0:
        raise ValueError("bank is empty")
    if k <= 0:
        raise ValueError("k must be positive")

    k_eff = int(min(k, N))
    B = int(q.shape[0])

    best_d2 = np.full((B, k_eff), np.inf, dtype=np.float32)
    best_idx = np.full((B, k_eff), -1, dtype=np.int64)

    def merge(
        best_d2: np.ndarray,
        best_idx: np.ndarray,
        cand_d2: np.ndarray,
        cand_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        merged_d2 = np.concatenate([best_d2, cand_d2], axis=1)
        merged_idx = np.concatenate([best_idx, cand_idx], axis=1)
        sel = np.argpartition(merged_d2, kth=k_eff - 1, axis=1)[:, :k_eff]
        out_d2 = np.take_along_axis(merged_d2, sel, axis=1)
        out_idx = np.take_along_axis(merged_idx, sel, axis=1)
        return out_d2, out_idx

    for start, end, dist2 in _pairwise_sqdist_chunked(q, bank, chunk_size=int(chunk_size)):
        C = int(end - start)
        if C <= k_eff:
            cand_d2 = dist2
            cand_idx = (np.arange(start, end, dtype=np.int64)[None, :]).repeat(B, axis=0)
        else:
            sel = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]
            cand_d2 = np.take_along_axis(dist2, sel, axis=1)
            cand_idx = (sel.astype(np.int64) + int(start))
        best_d2, best_idx = merge(best_d2, best_idx, cand_d2, cand_idx)

    return best_d2, best_idx


def knn_mean_sqdist(q: np.ndarray, bank: np.ndarray, *, k: int, chunk_size: int = 8192) -> np.ndarray:
    """Compute mean squared distance to k nearest neighbors: shape (B,)."""
    d2 = knn_topk_sqdist(q, bank, k=int(k), chunk_size=int(chunk_size))
    return d2.mean(axis=1).astype(np.float32, copy=False)


# -----------------------------
# GenRep-style config and detector
# -----------------------------

@dataclass(frozen=True)
class GenRepConfig:
    encoder: Literal["logmel_stats", "beats_speechbrain"] = "logmel_stats"

    knn_k: int = 1
    dist_chunk_size: int = 8192

    enable_memmixup: bool = True
    memmixup_k: int = 5
    memmixup_lambda: float = 0.9

    enable_dn: bool = True
    dn_eps_std: float = 1e-6
    dn_use_loocv: bool = True

    source_train_domain: str = "source"
    target_train_domain: str = "target"

    batch_size: int = 16

    def validate(self) -> None:
        if self.knn_k <= 0:
            raise ValueError("knn_k must be positive")
        if self.dist_chunk_size <= 0:
            raise ValueError("dist_chunk_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.enable_memmixup:
            if self.memmixup_k <= 0:
                raise ValueError("memmixup_k must be positive")
            if not (0.0 < self.memmixup_lambda <= 1.0):
                raise ValueError("memmixup_lambda must be in (0,1]")
        if self.enable_dn:
            if self.dn_eps_std <= 0:
                raise ValueError("dn_eps_std must be positive")


@dataclass
class DomainNormStats:
    mu_s: float
    std_s: float
    mu_t: float
    std_t: float


class GenRepDetector:
    """GenRep-style memory bank detector with optional MemMixup and Domain Normalization."""

    def __init__(self, cfg: GenRepConfig, encoder: ClipEncoderBase, logger: Optional[logging.Logger] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.encoder = encoder
        self.logger = logger or logging.getLogger("approach_b")

        self.source_bank: Optional[np.ndarray] = None
        self.target_bank: Optional[np.ndarray] = None
        self.dn_stats: Optional[DomainNormStats] = None

    def _encode_metas(self, metas: Sequence[Any], dataset: Any) -> np.ndarray:
        if len(metas) == 0:
            return np.zeros((0, self.encoder.embedding_dim), dtype=np.float32)

        bs = int(self.cfg.batch_size)
        sr_expected = int(self.encoder.expected_sample_rate)

        out: List[np.ndarray] = []
        for i in range(0, len(metas), bs):
            batch = metas[i : i + bs]
            waves: List[np.ndarray] = []
            for m in batch:
                y, sr = _dataset_load_audio(dataset, m)
                if int(sr) != sr_expected:
                    raise ValueError(f"Expected {sr_expected} Hz audio, got {sr} Hz for { _rel_path(m) }")
                waves.append(y)
            emb = self.encoder.encode(waves, sampling_rate=sr_expected)
            if emb.ndim != 2 or emb.shape[0] != len(batch):
                raise RuntimeError(f"Encoder returned shape {emb.shape}, expected (B,D) with B={len(batch)}")
            out.append(emb.astype(np.float32, copy=False))

        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    def _apply_memmixup(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Augment target bank with MemMixup (lambda * target + (1-lambda) * nearest source)."""
        if target.shape[0] == 0:
            return target
        if source.shape[0] == 0:
            return target

        K = int(min(self.cfg.memmixup_k, source.shape[0]))
        lam = float(self.cfg.memmixup_lambda)

        _, idx = knn_topk_indices(target, source, k=K, chunk_size=int(self.cfg.dist_chunk_size))
        src_nn = source[idx]
        tgt = target[:, None, :]
        aug = lam * tgt + (1.0 - lam) * src_nn
        aug = aug.reshape(-1, source.shape[1]).astype(np.float32, copy=False)

        enriched = np.concatenate([target, aug], axis=0).astype(np.float32, copy=False)
        return enriched

    def _compute_dn_stats(self, source_train: np.ndarray, target_train: np.ndarray, source_bank: np.ndarray, target_bank: np.ndarray) -> DomainNormStats:
        """Compute Z-score statistics for DN using training normal samples.

        Notes
        -----
        When K_n=1, naive "train-to-bank" distances can collapse to ~0 if the query appears
        in the bank; therefore, we default to a leave-one-out strategy (LOOCV) when
        estimating μ/σ, if there are enough samples.

        If a domain has no available samples/bank, its statistics fall back to (μ=0, σ=1),
        and its normalized distance will be ignored via min() because the distance is set
        to +inf at scoring time.
        """
        eps = float(self.cfg.dn_eps_std)

        # ---- Source stats ----
        ds: np.ndarray
        if source_bank.shape[0] == 0 or source_train.shape[0] == 0:
            ds = np.zeros((0,), dtype=np.float32)
        elif source_train.shape[0] >= 2 and self.cfg.dn_use_loocv:
            k_eff = int(min(self.cfg.knn_k, source_bank.shape[0] - 1))
            if k_eff <= 0:
                ds = knn_mean_sqdist(source_train, source_bank, k=1, chunk_size=int(self.cfg.dist_chunk_size))
            else:
                d2 = knn_topk_sqdist(source_train, source_bank, k=k_eff + 1, chunk_size=int(self.cfg.dist_chunk_size))
                ds = d2[:, 1:].mean(axis=1).astype(np.float32, copy=False)
        else:
            k_use = int(min(self.cfg.knn_k, max(1, source_bank.shape[0])))
            ds = knn_mean_sqdist(source_train, source_bank, k=k_use, chunk_size=int(self.cfg.dist_chunk_size))

        # ---- Target stats ----
        dt: np.ndarray
        if target_bank.shape[0] == 0 or target_train.shape[0] == 0:
            dt = np.zeros((0,), dtype=np.float32)
        elif target_train.shape[0] >= 2 and self.cfg.dn_use_loocv:
            k_eff = int(min(self.cfg.knn_k, target_bank.shape[0] - 1))
            if k_eff <= 0:
                dt = knn_mean_sqdist(target_train, target_bank, k=1, chunk_size=int(self.cfg.dist_chunk_size))
            else:
                d2 = knn_topk_sqdist(target_train, target_bank, k=k_eff + 1, chunk_size=int(self.cfg.dist_chunk_size))
                dt = d2[:, 1:].mean(axis=1).astype(np.float32, copy=False)
        else:
            k_use = int(min(self.cfg.knn_k, max(1, target_bank.shape[0])))
            dt = knn_mean_sqdist(target_train, target_bank, k=k_use, chunk_size=int(self.cfg.dist_chunk_size))

        mu_s = float(np.mean(ds)) if ds.size else 0.0
        std_s = float(np.std(ds)) if ds.size else 1.0
        mu_t = float(np.mean(dt)) if dt.size else 0.0
        std_t = float(np.std(dt)) if dt.size else 1.0

        std_s = max(std_s, eps)
        std_t = max(std_t, eps)

        return DomainNormStats(mu_s=mu_s, std_s=std_s, mu_t=mu_t, std_t=std_t)

    def fit(self, source_train_metas: Sequence[Any], target_train_metas: Sequence[Any], dataset: Any) -> None:
        """Build memory banks (and DN statistics if enabled)."""
        if len(source_train_metas) == 0 and len(target_train_metas) == 0:
            raise ValueError("Both source_train_metas and target_train_metas are empty.")

        source_train = self._encode_metas(source_train_metas, dataset)
        target_train = self._encode_metas(target_train_metas, dataset)

        source_bank = source_train.copy()
        target_bank = target_train.copy()

        if self.cfg.enable_memmixup:
            target_bank = self._apply_memmixup(source_bank, target_bank)

        self.source_bank = source_bank.astype(np.float32, copy=False)
        self.target_bank = target_bank.astype(np.float32, copy=False)

        if self.cfg.enable_dn:
            self.dn_stats = self._compute_dn_stats(source_train, target_train, self.source_bank, self.target_bank)
        else:
            self.dn_stats = None

        if self.logger:
            self.logger.debug(
                "Fitted detector: source_bank=%s, target_bank=%s, dn=%s",
                tuple(self.source_bank.shape) if self.source_bank is not None else None,
                tuple(self.target_bank.shape) if self.target_bank is not None else None,
                dataclasses.asdict(self.dn_stats) if self.dn_stats is not None else None,
            )

    def score_embeddings(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score a batch of embeddings.

        Returns:
          final_score: (B,)
          ds: (B,) distance to source bank (mean squared kNN distance)
          dt: (B,) distance to target bank (mean squared kNN distance)
        """
        if self.source_bank is None or self.target_bank is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"emb must be 2D (B,D), got shape {emb.shape}")

        bank_dim = None
        if self.source_bank.shape[0] > 0:
            bank_dim = int(self.source_bank.shape[1])
        elif self.target_bank.shape[0] > 0:
            bank_dim = int(self.target_bank.shape[1])
        else:
            raise RuntimeError("Both source and target banks are empty; cannot score.")

        if int(emb.shape[1]) != int(bank_dim):
            raise ValueError(f"Embedding dim mismatch: emb D={emb.shape[1]} vs bank D={bank_dim}")

        B = int(emb.shape[0])

        if self.source_bank.shape[0] == 0:
            ds = np.full((B,), np.inf, dtype=np.float32)
        else:
            k_s = int(min(self.cfg.knn_k, int(self.source_bank.shape[0])))
            ds = knn_mean_sqdist(emb, self.source_bank, k=k_s, chunk_size=int(self.cfg.dist_chunk_size))

        if self.target_bank.shape[0] == 0:
            dt = np.full((B,), np.inf, dtype=np.float32)
        else:
            k_t = int(min(self.cfg.knn_k, int(self.target_bank.shape[0])))
            dt = knn_mean_sqdist(emb, self.target_bank, k=k_t, chunk_size=int(self.cfg.dist_chunk_size))

        if self.cfg.enable_dn and self.dn_stats is not None:
            z_s = (ds - float(self.dn_stats.mu_s)) / float(self.dn_stats.std_s)
            z_t = (dt - float(self.dn_stats.mu_t)) / float(self.dn_stats.std_t)
            final = np.minimum(z_s, z_t).astype(np.float32, copy=False)
        else:
            final = np.minimum(ds, dt).astype(np.float32, copy=False)

        return final, ds, dt

    def score_clips(self, metas: Sequence[Any], dataset: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score dataset clips and return (final, ds, dt)."""
        emb = self._encode_metas(metas, dataset)
        return self.score_embeddings(emb)


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
        description="Approach B (GenRep-style): BEATs/log-mel embeddings + kNN + MemMixup + Domain Normalization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data_root", type=str, required=True, help="Dataset root containing metadata.csv (ToyASDDataset).")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write scores/metrics.")

    p.add_argument("--encoder", type=str, default="logmel_stats", choices=["logmel_stats", "beats_speechbrain"], help="Embedding encoder.")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding extraction.")

    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--win_ms", type=float, default=25.0)
    p.add_argument("--hop_ms", type=float, default=10.0)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--per_clip_norm", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--beats_ckpt", type=str, default="", help="Path to BEATs checkpoint (.pt) for SpeechBrain encoder.")
    p.add_argument("--beats_device", type=str, default="auto")
    p.add_argument("--beats_output_all_hiddens", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--beats_layer", type=int, default=-1)
    p.add_argument("--beats_fbank_mean", type=float, default=15.41663)
    p.add_argument("--beats_fbank_std", type=float, default=6.55582)

    p.add_argument("--knn_k", type=int, default=1)
    p.add_argument("--dist_chunk_size", type=int, default=8192)

    p.add_argument("--enable_memmixup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--memmixup_k", type=int, default=5)
    p.add_argument("--memmixup_lambda", type=float, default=0.9)

    p.add_argument("--enable_dn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dn_eps_std", type=float, default=1e-6)
    p.add_argument("--dn_use_loocv", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--source_n", type=int, default=-1, help="If >0, use only this many source train normal clips per section.")
    p.add_argument("--target_n", type=int, default=-1, help="If >0, use only this many target train normal clips per section.")
    p.add_argument("--subset_sampling", type=str, default="random", choices=["first", "random"])
    p.add_argument("--subset_seed", type=int, default=0)

    p.add_argument("--test_domain", type=str, default="all", choices=["source", "target", "all"])
    p.add_argument("--pauc_max_fpr", type=float, default=0.1)
    p.add_argument("--no_eval", action="store_true")

    p.add_argument("--machine_types", type=str, default="", help="Comma-separated subset of machine types. Default: all.")
    p.add_argument("--section_ids", type=str, default="", help="Comma-separated subset of section ids. Default: all.")

    p.add_argument("--verbosity", type=int, default=1, help="0=warnings,1=info,2=debug")

    return p.parse_args(argv)


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logger = _setup_logger(int(args.verbosity))

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ToyASDDataset.load(data_root)
    metas_all = _dataset_metas(ds)

    if str(args.machine_types).strip():
        mts = [m.strip() for m in str(args.machine_types).split(",") if m.strip()]
    else:
        mts = _unique_sorted([_machine_type(m) for m in metas_all])

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

    logger.info("Loaded dataset: root=%s, machine_types=%s", str(data_root), mts)
    for mt in mts:
        logger.info("  - %s sections=%s", mt, sections_by_mt.get(mt, []))

    enc_name = str(args.encoder).lower().strip()
    if enc_name == "logmel_stats":
        mel_cfg = LogMelConfig(
            sample_rate=int(args.sample_rate),
            n_mels=int(args.n_mels),
            win_ms=float(args.win_ms),
            hop_ms=float(args.hop_ms),
            n_fft=int(args.n_fft),
        )
        enc_cfg = LogMelStatsEncoderConfig(
            mel=mel_cfg,
            include_std=True,
            per_clip_norm=bool(args.per_clip_norm),
        )
        encoder: ClipEncoderBase = LogMelStatsEncoder(enc_cfg)
    elif enc_name == "beats_speechbrain":
        if not str(args.beats_ckpt).strip():
            raise ValueError("--beats_ckpt is required for --encoder beats_speechbrain")
        be_cfg = SpeechBrainBEATsEncoderConfig(
            checkpoint_path=str(args.beats_ckpt),
            device=str(args.beats_device),
            output_all_hiddens=bool(args.beats_output_all_hiddens),
            layer=int(args.beats_layer),
            fbank_mean=float(args.beats_fbank_mean),
            fbank_std=float(args.beats_fbank_std),
            freeze=True,
        )
        encoder = SpeechBrainBEATsEncoder(be_cfg, logger=logger)
    else:
        raise ValueError(f"Unsupported encoder: {enc_name}")

    cfg = GenRepConfig(
        encoder=enc_name,  # type: ignore[arg-type]
        knn_k=int(args.knn_k),
        dist_chunk_size=int(args.dist_chunk_size),
        enable_memmixup=bool(args.enable_memmixup),
        memmixup_k=int(args.memmixup_k),
        memmixup_lambda=float(args.memmixup_lambda),
        enable_dn=bool(args.enable_dn),
        dn_eps_std=float(args.dn_eps_std),
        dn_use_loocv=bool(args.dn_use_loocv),
        batch_size=int(args.batch_size),
    )
    cfg.validate()

    score_rows: List[Dict[str, Any]] = []
    section_metrics: List[Dict[str, Any]] = []

    for mt in mts:
        for sid in sections_by_mt.get(mt, []):
            src_train_all = _dataset_filter(ds, machine_types=[mt], section_ids=[sid], split="train", domain="source", is_anomaly=False)
            tgt_train_all = _dataset_filter(ds, machine_types=[mt], section_ids=[sid], split="train", domain="target", is_anomaly=False)

            src_n = None if int(args.source_n) < 0 else int(args.source_n)
            tgt_n = None if int(args.target_n) < 0 else int(args.target_n)

            src_train = _select_subset(src_train_all, n=src_n, mode=str(args.subset_sampling), seed=int(args.subset_seed))
            tgt_train = _select_subset(tgt_train_all, n=tgt_n, mode=str(args.subset_sampling), seed=int(args.subset_seed) + 1)

            if len(src_train) == 0 and len(tgt_train) == 0:
                logger.warning("No train normal clips for machine=%s section=%s; skipping.", mt, sid)
                continue

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

            detector = GenRepDetector(cfg, encoder, logger=logger)
            detector.fit(src_train, tgt_train, ds)

            final, ds_sc, dt_sc = detector.score_clips(test_metas, ds)

            for m, s, ds_v, dt_v in zip(test_metas, final, ds_sc, dt_sc):
                score_rows.append(
                    {
                        "clip_id": _clip_id(m),
                        "relative_path": _rel_path(m),
                        "machine_type": _machine_type(m),
                        "section_id": int(_section_id(m)),
                        "split": _split(m),
                        "domain": "" if _domain(m) is None else str(_domain(m)),
                        "is_anomaly": int(_is_anomaly(m)),
                        "label": "" if _label(m) is None else str(_label(m)),
                        "score": float(s),
                        "ds": float(ds_v),
                        "dt": float(dt_v),
                        "encoder": enc_name,
                        "knn_k": int(cfg.knn_k),
                        "enable_memmixup": int(cfg.enable_memmixup),
                        "memmixup_k": int(cfg.memmixup_k),
                        "memmixup_lambda": float(cfg.memmixup_lambda),
                        "enable_dn": int(cfg.enable_dn),
                        "dn_use_loocv": int(cfg.dn_use_loocv),
                    }
                )

            if not bool(args.no_eval):
                y_true_all = np.array([1 if _is_anomaly(m) else 0 for m in test_metas], dtype=np.int64)

                auc_all = _safe_auc(y_true_all, final)
                pauc_all = _safe_pauc(y_true_all, final, max_fpr=float(args.pauc_max_fpr))

                per_domain: Dict[str, Dict[str, Optional[float]]] = {}
                for d in test_domains:
                    idx = [i for i, m in enumerate(test_metas) if _domain(m) == d]
                    if not idx:
                        continue
                    y = y_true_all[idx]
                    sc = final[idx]
                    per_domain[d] = {
                        "auc": _safe_auc(y, sc),
                        "pauc": _safe_pauc(y, sc, max_fpr=float(args.pauc_max_fpr)),
                        "n": int(len(idx)),
                        "n_anom": int(y.sum()),
                    }

                official = None
                if "source" in per_domain and "target" in per_domain:
                    s_auc = per_domain["source"].get("auc")
                    t_auc = per_domain["target"].get("auc")
                    mix_pauc = pauc_all
                    if s_auc is not None and t_auc is not None and mix_pauc is not None:
                        official = _harmonic_mean([float(s_auc), float(t_auc), float(mix_pauc)])

                section_metrics.append(
                    {
                        "machine_type": mt,
                        "section_id": int(sid),
                        "n_source_train": int(len(src_train)),
                        "n_target_train": int(len(tgt_train)),
                        "n_test": int(len(test_metas)),
                        "auc": auc_all,
                        "pauc": pauc_all,
                        "per_domain": per_domain,
                        "official_proxy": official,
                    }
                )

            logger.info(
                "Finished machine=%s section=%02d: src_train=%d tgt_train=%d test=%d",
                mt,
                int(sid),
                int(len(src_train)),
                int(len(tgt_train)),
                int(len(test_metas)),
            )

    scores_csv = out_dir / "scores.csv"
    _write_scores_csv(scores_csv, score_rows)

    if not bool(args.no_eval):
        aucs = [m["auc"] for m in section_metrics if m.get("auc") is not None]
        paucs = [m["pauc"] for m in section_metrics if m.get("pauc") is not None]
        offic = [m["official_proxy"] for m in section_metrics if m.get("official_proxy") is not None]

        agg = {
            "n_sections": int(len(section_metrics)),
            "macro_auc_mean": float(np.mean(aucs)) if aucs else None,
            "macro_pauc_mean": float(np.mean(paucs)) if paucs else None,
            "macro_official_proxy_mean": float(np.mean(offic)) if offic else None,
            "pauc_max_fpr": float(args.pauc_max_fpr),
            "config": dataclasses.asdict(cfg),
            "encoder_dim": int(encoder.embedding_dim),
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
