"""approach_a.py

Approach A: Few-shot *training-free* anomalous sound detection (ASD)

This module implements the "AST-PB" (AST Patch-Based) method described in:
  - Ho-Hsiang Wu, Wei-Cheng Lin, Abinaya Kumar, Luca Bondi, Shabnam Ghaffarzadegan,
    Juan Pablo Bello, "Towards Few-Shot Training-Free Anomaly Sound Detection",
    INTERSPEECH 2025.

High-level idea
---------------
Given a small normal *support set* D (typically N <= 10) and a query clip q:
  1) Extract patch-level representations from a frozen, pretrained AST encoder.
  2) For each query patch, compute cosine similarity to all support patches and
     keep only the maximum similarity.
  3) Convert to a patch-level anomaly value: a = 1 - max_cosine_similarity.
  4) Average anomaly values across multiple transformer layers -> anomaly map.
  5) Pool the anomaly map into a clip-level score via an upper-tail quantile.

This implementation is designed to work out-of-the-box with the dataset contract
provided by `toy_dummy_data.py` (ToyASDDataset / ClipMeta). It includes:
  - A clean Python API (encoder + detector)
  - A robust CLI that can score/evaluate per (machine_type, section_id)

Notes on exactness vs robustness
-------------------------------
The paper reports a patch grid of 12 x 101 (1212 patches) and L=11 layers.
Hugging Face's AST models typically have 12 transformer layers; in practice,
using the first 11 layers (excluding the last) often matches the "L=11" setup.
This code supports both choices.

Key references:
  - INTERSPEECH 2025 AST-PB description (patch grid, layer aggregation, cosine max):
    https://www.isca-archive.org/interspeech_2025/wu25b_interspeech.pdf
  - Hugging Face AST documentation (max_length=1024, num_mel_bins=128, etc.):
    https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer

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

# Local dataset/metrics contract (toy). This module is intentionally written
# against this contract to ensure A/B/C can share a common backbone.
try:
    from toy_dummy_data import (
        ClipMeta,
        ToyASDDataset,
        partial_roc_auc_score_binary,
        roc_auc_score_binary,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "approach_a.py expects toy_dummy_data.py to be importable in PYTHONPATH. "
        "Place approach_a.py next to toy_dummy_data.py or install it as a module."
    ) from e


# -----------------------------
# Lazy imports (torch / transformers)
# -----------------------------

def _import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for approach_a.py when using the AST encoder. "
            "Install it via https://pytorch.org/get-started/locally/."
        ) from e


def _import_torch_nn_functional() -> Any:
    _ = _import_torch()
    try:
        import torch.nn.functional as F  # type: ignore

        return F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import torch.nn.functional.") from e


def _import_transformers() -> Any:
    try:
        import transformers  # type: ignore

        return transformers
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face 'transformers' is required for the AST encoder. "
            "Install it via: pip install transformers"
        ) from e


def _has_cuda(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Resolve device spec into a concrete torch device string."""
    device = (device or "").strip().lower()
    if device in ("", "auto"):
        torch = _import_torch()
        return "cuda" if _has_cuda(torch) else "cpu"
    if device in ("cpu", "cuda"):
        return device
    # allow e.g. cuda:0
    if device.startswith("cuda:"):
        return device
    raise ValueError(f"Unsupported device spec: {device!r}")


def _resolve_dtype(dtype: str, torch: Any) -> Any:
    dt = (dtype or "").strip().lower()
    if dt in ("", "float32", "fp32"):
        return torch.float32
    if dt in ("float16", "fp16", "half"):
        return torch.float16
    if dt in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype!r}")


# -----------------------------
# Logging
# -----------------------------

def _setup_logger(verbosity: int) -> logging.Logger:
    logger = logging.getLogger("approach_a")
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
# Layer parsing
# -----------------------------

def _parse_layer_spec(spec: str, *, num_hidden_layers: int, exclude_last_layer: bool) -> List[int]:
    """Return a sorted list of 1-based transformer layer indices (excluding embeddings)."""
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive.")

    spec_norm = (spec or "").strip().lower()
    if spec_norm in ("", "all"):
        # Hidden-states indexing (HF): hidden_states[0] = embedding output;
        # hidden_states[1..num_hidden_layers] correspond to transformer layers.
        if exclude_last_layer and num_hidden_layers >= 2:
            layers = list(range(1, num_hidden_layers))
        else:
            layers = list(range(1, num_hidden_layers + 1))
        return layers

    # Support formats:
    #   - "1,2,3"
    #   - "1-4,6,8-11"
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    layers: List[int] = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                raise ValueError(f"Invalid layer range {tok!r} (end < start)")
            layers.extend(list(range(a_i, b_i + 1)))
        else:
            layers.append(int(tok))

    # Validate & normalize
    layers = sorted(set(layers))
    if any(l < 1 or l > num_hidden_layers for l in layers):
        raise ValueError(
            f"Layer indices must be in [1, {num_hidden_layers}] (got {layers}). "
            "Note: 0 is the embedding output and is not a transformer layer."
        )

    if exclude_last_layer and num_hidden_layers in layers:
        # If the user explicitly requested 'all' we already handled above; here
        # we interpret exclude_last_layer as a hard safety default.
        layers = [l for l in layers if l != num_hidden_layers]
        if not layers:
            raise ValueError(
                "exclude_last_layer removed all requested layers. "
                "Specify --exclude_last_layer false or request additional layers."
            )

    return layers


# -----------------------------
# Config
# -----------------------------


QuantileTail = Literal["upper", "lower"]


@dataclass(frozen=True)
class ASTPatchDiffConfig:
    """Configuration for the training-free AST patch-based detector."""

    # Encoder
    model_name_or_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    device: str = "auto"  # auto|cpu|cuda|cuda:0
    dtype: str = "float32"  # float32|float16|bfloat16
    attn_implementation: Optional[str] = None  # e.g. "sdpa" for torch>=2

    # Layer selection
    use_layers: str = "all"  # "all" or "1-3,5,7" etc.
    exclude_last_layer: bool = True  # paper reports L=11; HF AST often has 12 layers

    # Support/key memory
    max_keys_per_layer: Optional[int] = None  # subsample to this many keys per layer
    key_subsample: Literal["random", "first"] = "random"
    key_subsample_seed: int = 0

    # Similarity compute
    sim_chunk_size: int = 4096  # keys are chunked to avoid large intermediate matrices

    # Scoring
    decision_quantile: float = 0.05
    quantile_tail: QuantileTail = "upper"

    # Numerics
    l2_normalize: bool = True
    eps: float = 1e-6

    # Batching
    batch_size: int = 4

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sim_chunk_size <= 0:
            raise ValueError("sim_chunk_size must be positive")
        if not (0.0 < self.decision_quantile < 1.0):
            raise ValueError("decision_quantile must be in (0,1)")
        if self.max_keys_per_layer is not None and self.max_keys_per_layer <= 0:
            raise ValueError("max_keys_per_layer must be positive if set")
        if self.key_subsample not in ("random", "first"):
            raise ValueError("key_subsample must be 'random' or 'first'")
        if self.quantile_tail not in ("upper", "lower"):
            raise ValueError("quantile_tail must be 'upper' or 'lower'")


# -----------------------------
# Patch encoders
# -----------------------------


class PatchEncoderBase:
    """Interface for patch-level encoders."""

    def encode_patches(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> Dict[int, "torch.Tensor"]:
        """Return {layer_index -> patch embeddings}.

        Each value is a tensor of shape:
            (B, P, D) where
              - B = batch size (len(waveforms))
              - P = number of patches
              - D = embedding dimension
        """
        raise NotImplementedError

    @property
    def selected_layers(self) -> List[int]:
        raise NotImplementedError

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    def patch_grid(self) -> Optional[Tuple[int, int]]:
        """If known, returns (freq_patches, time_patches)."""
        return None

    @property
    def sampling_rate(self) -> int:
        """Expected waveform sampling rate in Hz."""
        return 16000


def _infer_num_special_tokens(seq_len: int, expected_patches: Optional[int]) -> int:
    """Infer number of special tokens (CLS/distill) before patch tokens."""
    if expected_patches is None:
        # common for HF AST (cls + distill)
        return 2 if seq_len >= 3 else 0

    diff = seq_len - int(expected_patches)
    if diff in (1, 2):
        return diff

    # Try common values
    if seq_len - 2 == expected_patches:
        return 2
    if seq_len - 1 == expected_patches:
        return 1

    # Fallback: assume first two are special if possible
    return 2 if seq_len >= 3 else 0


class FrozenASTEncoder(PatchEncoderBase):
    """Frozen pretrained AST encoder with patch-level hidden states."""

    def __init__(self, cfg: ASTPatchDiffConfig, logger: Optional[logging.Logger] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.logger = logger or logging.getLogger("approach_a")

        torch = _import_torch()
        transformers = _import_transformers()

        self.device = _resolve_device(cfg.device)
        self.dtype = _resolve_dtype(cfg.dtype, torch)

        # Load feature extractor (pads/truncates to max_length=1024 by default)
        # Use AutoFeatureExtractor for broad compatibility.
        try:
            self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(cfg.model_name_or_path)
        except Exception:
            # Some older versions might still expose ASTFeatureExtractor directly.
            self.feature_extractor = transformers.ASTFeatureExtractor.from_pretrained(cfg.model_name_or_path)

        # Load model
        model_kwargs: Dict[str, Any] = {}
        if cfg.attn_implementation is not None:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        self.model = transformers.ASTModel.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Move to device / dtype
        self.model.to(self.device)
        if self.dtype in (torch.float16, torch.bfloat16):
            # Only cast model; input will be cast accordingly.
            self.model.to(dtype=self.dtype)

        # Infer expected patch grid from config (if possible)
        self._patch_grid: Optional[Tuple[int, int]] = None
        self._expected_patches: Optional[int] = None
        try:
            mcfg = self.model.config
            patch_size = int(getattr(mcfg, "patch_size"))
            freq_stride = int(getattr(mcfg, "frequency_stride"))
            time_stride = int(getattr(mcfg, "time_stride"))
            num_mel_bins = int(getattr(mcfg, "num_mel_bins"))
            max_length = int(getattr(mcfg, "max_length"))

            # Derived like ViT patchification.
            freq_patches = int(math.floor((num_mel_bins - patch_size) / float(freq_stride)) + 1)
            time_patches = int(math.floor((max_length - patch_size) / float(time_stride)) + 1)

            if freq_patches > 0 and time_patches > 0:
                self._patch_grid = (freq_patches, time_patches)
                self._expected_patches = int(freq_patches * time_patches)

            self._num_hidden_layers = int(getattr(mcfg, "num_hidden_layers"))
        except Exception:
            self._num_hidden_layers = 12  # reasonable fallback

        self._selected_layers = _parse_layer_spec(
            cfg.use_layers, num_hidden_layers=self._num_hidden_layers, exclude_last_layer=cfg.exclude_last_layer
        )

        if self.logger:
            self.logger.info(
                "Loaded AST encoder: model=%s, device=%s, dtype=%s, layers=%s",
                cfg.model_name_or_path,
                self.device,
                str(self.dtype).replace("torch.", ""),
                self._selected_layers,
            )
            if self._patch_grid is not None:
                self.logger.info("Expected patch grid: %s (P=%d)", self._patch_grid, int(self._expected_patches or 0))

    @property
    def selected_layers(self) -> List[int]:
        return list(self._selected_layers)

    @property
    def hidden_size(self) -> int:
        try:
            return int(getattr(self.model.config, "hidden_size"))
        except Exception:
            return 768

    @property
    def patch_grid(self) -> Optional[Tuple[int, int]]:
        return self._patch_grid

    @property
    def sampling_rate(self) -> int:
        try:
            return int(getattr(self.feature_extractor, "sampling_rate"))
        except Exception:
            return 16000

    def _extract_input_values(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> Tuple[Any, Any]:
        """Return (input_values_tensor, torch_module)."""
        torch = _import_torch()

        # Feature extractor expects float32 waveforms.
        waves: List[np.ndarray] = []
        for w in waveforms:
            ww = np.asarray(w, dtype=np.float32).reshape(-1)
            # Replace NaN/Inf defensively (rare but protects downstream)
            if not np.isfinite(ww).all():
                ww = np.nan_to_num(ww, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            waves.append(ww)

        # Most AST checkpoints are trained for 16 kHz.
        # We enforce the feature extractor sampling rate.
        fe_sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        if int(sampling_rate) != fe_sr:
            raise ValueError(
                f"AST feature extractor expects sampling_rate={fe_sr}, got {sampling_rate}. "
                "Resample your audio upstream to avoid silent errors."
            )

        batch = self.feature_extractor(waves, sampling_rate=fe_sr, return_tensors="pt")

        if "input_values" in batch:
            x = batch["input_values"]
        elif "input_features" in batch:  # compatibility
            x = batch["input_features"]
        else:
            raise KeyError(f"Unexpected feature extractor output keys: {list(batch.keys())}")

        x = x.to(self.device)
        if self.dtype != torch.float32:
            x = x.to(dtype=self.dtype)
        return x, torch

    def encode_patches(self, waveforms: Sequence[np.ndarray], sampling_rate: int) -> Dict[int, "torch.Tensor"]:
        """Encode waveforms into patch embeddings per selected layer."""
        torch = _import_torch()
        F = _import_torch_nn_functional()

        x, _ = self._extract_input_values(waveforms, sampling_rate=sampling_rate)

        with torch.no_grad():
            out = self.model(input_values=x, output_hidden_states=True, return_dict=True)

        hidden_states = out.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states. Ensure output_hidden_states=True.")

        # Determine special tokens count
        ref = hidden_states[self._selected_layers[0]]
        seq_len = int(ref.shape[1])
        num_special = _infer_num_special_tokens(seq_len, self._expected_patches)

        patches_by_layer: Dict[int, "torch.Tensor"] = {}
        for l in self._selected_layers:
            h = hidden_states[l]  # (B, seq_len, D)
            if num_special > 0:
                p = h[:, num_special:, :]
            else:
                p = h
            # L2 normalize to ensure cosine similarity equals dot product
            if self.cfg.l2_normalize:
                p = F.normalize(p, p=2.0, dim=-1, eps=float(self.cfg.eps))
            patches_by_layer[int(l)] = p

        return patches_by_layer


# -----------------------------
# Memory bank
# -----------------------------


class PatchKeyMemory:
    """Stores patch embeddings for each selected layer."""

    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        max_keys_per_layer: Optional[int] = None,
        subsample: Literal["random", "first"] = "random",
        subsample_seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.layer_indices = list(layer_indices)
        self.max_keys_per_layer = max_keys_per_layer
        self.subsample = subsample
        self.subsample_seed = int(subsample_seed)
        self.logger = logger

        self._buf: Dict[int, List["torch.Tensor"]] = {int(l): [] for l in self.layer_indices}

    def add(self, patches_by_layer: Mapping[int, "torch.Tensor"]) -> None:
        """Add a batch of patch embeddings: each is (B,P,D)."""
        for l in self.layer_indices:
            if l not in patches_by_layer:
                raise KeyError(f"Missing layer {l} in patches_by_layer.")
            t = patches_by_layer[l]
            self._buf[l].append(t.detach())

    def finalize(self) -> Dict[int, "torch.Tensor"]:
        """Concatenate and optionally subsample keys per layer.

        Returns:
          keys_by_layer: layer -> (K, D) tensor where K = total patches across all supports.
        """
        torch = _import_torch()

        keys_by_layer: Dict[int, "torch.Tensor"] = {}
        rng = random.Random(self.subsample_seed)

        for l, chunks in self._buf.items():
            if not chunks:
                raise RuntimeError(f"No keys accumulated for layer {l}.")
            # concat along batch, then flatten patches
            keys = torch.cat(chunks, dim=0)  # (N, P, D)
            B, P, D = keys.shape
            keys = keys.reshape(B * P, D)  # (K,D)

            if self.max_keys_per_layer is not None and keys.shape[0] > int(self.max_keys_per_layer):
                k = int(self.max_keys_per_layer)
                if self.subsample == "first":
                    keys = keys[:k]
                else:
                    idx = list(range(int(keys.shape[0])))
                    rng.shuffle(idx)
                    idx = idx[:k]
                    idx_t = torch.tensor(idx, dtype=torch.long, device=keys.device)
                    keys = keys.index_select(dim=0, index=idx_t)

                if self.logger:
                    self.logger.debug("Subsampled layer %d keys: %d -> %d", l, int(B * P), int(keys.shape[0]))

            keys_by_layer[int(l)] = keys

        return keys_by_layer


# -----------------------------
# Core scoring
# -----------------------------


def _max_cosine_similarity_over_keys(
    q_patches: "torch.Tensor",
    k_patches: "torch.Tensor",
    *,
    chunk_size: int,
) -> "torch.Tensor":
    """Compute per-query-patch max cosine similarity over all keys.

    Args:
      q_patches: (B, P, D) normalized
      k_patches: (K, D) normalized
      chunk_size: chunk along K dimension

    Returns:
      max_sim: (B, P) with values in [-1,1]
    """
    torch = _import_torch()

    if q_patches.ndim != 3:
        raise ValueError(f"q_patches must be (B,P,D), got {q_patches.shape}")
    if k_patches.ndim != 2:
        raise ValueError(f"k_patches must be (K,D), got {k_patches.shape}")
    if q_patches.shape[-1] != k_patches.shape[-1]:
        raise ValueError(f"Embedding dim mismatch: q D={q_patches.shape[-1]} vs k D={k_patches.shape[-1]}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    B, P, D = q_patches.shape
    K = int(k_patches.shape[0])

    # Track max similarity per (B,P)
    try:
        init_val = torch.finfo(q_patches.dtype).min
    except Exception:  # pragma: no cover
        init_val = -1e9
    max_sim = torch.full((B, P), init_val, device=q_patches.device, dtype=q_patches.dtype)

    # Chunk over keys to bound intermediate memory (B,P,chunk)
    cs = int(chunk_size)
    for start in range(0, K, cs):
        end = min(K, start + cs)
        keys_chunk = k_patches[start:end]  # (C, D)
        # matmul: (B,P,D) x (D,C) -> (B,P,C)
        sim = torch.matmul(q_patches, keys_chunk.transpose(0, 1))
        sim_max = sim.max(dim=2).values  # (B,P)
        max_sim = torch.maximum(max_sim, sim_max)

    return max_sim


def _quantile_pool(values: np.ndarray, *, q: float, tail: QuantileTail) -> np.ndarray:
    """Pool last dimension by a quantile. Supports upper-tail fraction semantics."""
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"_quantile_pool expects (B,P) or (P,), got shape {values.shape}")

    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")

    # We interpret q as a *tail fraction* to align with the paper's default 0.05.
    # - tail='upper': score is the (1-q) quantile (e.g. q=0.05 -> 95th percentile)
    # - tail='lower': score is the q quantile (e.g. q=0.05 -> 5th percentile)
    if tail == "upper":
        q_level = 1.0 - float(q)
    else:
        q_level = float(q)

    # numpy.quantile API changed: `method=` (new) replaced `interpolation=` (old).
    # Use a compatibility shim so this code works across a wide range of numpy versions.
    try:
        return np.quantile(values, q_level, axis=1, method="linear")  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover
        return np.quantile(values, q_level, axis=1, interpolation="linear")  # type: ignore[arg-type]


class ASTPatchDiffDetector:
    """Training-free detector using a frozen AST encoder and patch similarity."""

    def __init__(self, cfg: ASTPatchDiffConfig, encoder: FrozenASTEncoder, logger: Optional[logging.Logger] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.encoder = encoder
        self.logger = logger or logging.getLogger("approach_a")

        self._memory = PatchKeyMemory(
            layer_indices=encoder.selected_layers,
            max_keys_per_layer=cfg.max_keys_per_layer,
            subsample=cfg.key_subsample,
            subsample_seed=cfg.key_subsample_seed,
            logger=self.logger,
        )
        self._keys: Optional[Dict[int, "torch.Tensor"]] = None

    def fit_support(self, support_metas: Sequence[ClipMeta], dataset: ToyASDDataset) -> None:
        """Build per-layer keys from a normal support set."""
        if len(support_metas) == 0:
            raise ValueError("support_metas is empty")

        # Reset
        self._memory = PatchKeyMemory(
            layer_indices=self.encoder.selected_layers,
            max_keys_per_layer=self.cfg.max_keys_per_layer,
            subsample=self.cfg.key_subsample,
            subsample_seed=self.cfg.key_subsample_seed,
            logger=self.logger,
        )
        self._keys = None

        # Encode support in batches
        bs = int(self.cfg.batch_size)
        sr_expected = int(getattr(self.encoder, "sampling_rate", 16000))

        for i in range(0, len(support_metas), bs):
            batch = support_metas[i : i + bs]
            waves: List[np.ndarray] = []
            for m in batch:
                y, sr = dataset.load_audio(m)
                if sr != sr_expected:
                    raise ValueError(f"Expected {sr_expected} Hz audio, got {sr} Hz for {m.relative_path}")
                waves.append(y)

            patches = self.encoder.encode_patches(waves, sampling_rate=sr_expected)
            self._memory.add(patches)

        # Finalize keys
        self._keys = self._memory.finalize()
        if self.logger:
            k0 = next(iter(self._keys.values()))
            self.logger.debug("Built key bank: layers=%d, example K=%d, D=%d", len(self._keys), int(k0.shape[0]), int(k0.shape[1]))

    def _require_keys(self) -> Dict[int, "torch.Tensor"]:
        if self._keys is None:
            raise RuntimeError("Detector is not fitted. Call fit_support() first.")
        return self._keys

    def score_waveforms(
        self, waveforms: Sequence[np.ndarray], *, sampling_rate: int, return_maps: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score a batch of waveforms.

        Returns:
          scores: (B,) float64
          maps: (B,P) float32 if return_maps else None
        """
        _ = _import_torch()

        keys_by_layer = self._require_keys()

        # Encode query patches
        patches_by_layer = self.encoder.encode_patches(waveforms, sampling_rate=sampling_rate)

        # Accumulate anomaly map across layers
        anomaly_sum: Optional["torch.Tensor"] = None
        used_layers: int = 0

        for l, q_patches in patches_by_layer.items():
            if l not in keys_by_layer:
                continue
            keys = keys_by_layer[l]

            # Ensure same device/dtype (avoid redundant copies)
            if keys.device != q_patches.device or keys.dtype != q_patches.dtype:
                keys = keys.to(device=q_patches.device, dtype=q_patches.dtype)

            # max cosine similarity per patch
            max_sim = _max_cosine_similarity_over_keys(q_patches, keys, chunk_size=int(self.cfg.sim_chunk_size))
            # anomaly value
            anomaly = 1.0 - max_sim

            if anomaly_sum is None:
                anomaly_sum = anomaly
            else:
                anomaly_sum = anomaly_sum + anomaly
            used_layers += 1

        if anomaly_sum is None or used_layers == 0:
            raise RuntimeError("No layers were scored. Check encoder layers and key bank.")

        anomaly_map = (anomaly_sum / float(used_layers)).detach()

        # Pool into clip-level score using quantile
        am_np = anomaly_map.to("cpu").float().numpy()
        scores = _quantile_pool(am_np, q=float(self.cfg.decision_quantile), tail=self.cfg.quantile_tail)

        maps = am_np.astype(np.float32, copy=False) if return_maps else None
        return scores.astype(np.float64, copy=False), maps

    def score_clips(
        self,
        metas: Sequence[ClipMeta],
        dataset: ToyASDDataset,
        *,
        return_maps: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score a list of dataset clips."""
        if len(metas) == 0:
            return np.zeros((0,), dtype=np.float64), None

        bs = int(self.cfg.batch_size)
        sr_expected = int(getattr(self.encoder, "sampling_rate", 16000))

        all_scores: List[np.ndarray] = []
        all_maps: List[np.ndarray] = []

        for i in range(0, len(metas), bs):
            batch = metas[i : i + bs]
            waves: List[np.ndarray] = []
            for m in batch:
                y, sr = dataset.load_audio(m)
                if sr != sr_expected:
                    raise ValueError(f"Expected {sr_expected} Hz audio, got {sr} Hz for {m.relative_path}")
                waves.append(y)

            scores, maps = self.score_waveforms(waves, sampling_rate=sr_expected, return_maps=return_maps)
            all_scores.append(scores)
            if return_maps and maps is not None:
                all_maps.append(maps)

        scores_cat = np.concatenate(all_scores, axis=0) if all_scores else np.zeros((0,), dtype=np.float64)
        maps_cat = np.concatenate(all_maps, axis=0) if (return_maps and all_maps) else None
        return scores_cat, maps_cat


# -----------------------------
# Evaluation helpers
# -----------------------------


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Return AUC or None if undefined."""
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.size == 0:
        return None
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score_binary(y_true.astype(np.float64), y_score.astype(np.float64)))


def _safe_pauc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.1) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.size == 0:
        return None
    if np.unique(y_true).size < 2:
        return None
    return float(
        partial_roc_auc_score_binary(
            y_true.astype(np.float64),
            y_score.astype(np.float64),
            max_fpr=max_fpr,
            standardized=True,
        )
    )


def _unique_sorted(values: Iterable[Any]) -> List[Any]:
    """Return unique values in a deterministic, human-friendly order.

    - If all values are ints (or numpy integer scalars), sort numerically.
    - If all values are strings, sort lexicographically.
    - Otherwise, sort by (type_name, string_repr).
    """
    uniq = list(set(values))
    if not uniq:
        return []
    if all(isinstance(v, (int, np.integer)) for v in uniq):
        return sorted(int(v) for v in uniq)
    if all(isinstance(v, str) for v in uniq):
        return sorted(uniq)
    return sorted(uniq, key=lambda v: (type(v).__name__, str(v)))


def _select_support(
    metas: Sequence[ClipMeta],
    *,
    n: Optional[int],
    mode: Literal["first", "random"],
    seed: int,
) -> List[ClipMeta]:
    metas_sorted = sorted(metas, key=lambda m: m.clip_id)
    if n is None or n >= len(metas_sorted):
        return metas_sorted

    if n <= 0:
        raise ValueError("support_n must be positive when provided")

    if mode == "first":
        return metas_sorted[:n]

    # deterministic random
    rng = random.Random(int(seed))
    idx = list(range(len(metas_sorted)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [metas_sorted[i] for i in idx]


# -----------------------------
# CLI
# -----------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Approach A (AST-PB): training-free few-shot ASD using frozen AST patch similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--data_root", type=str, required=True, help="Dataset root containing metadata.csv (ToyASDDataset).")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write scores/metrics.")

    # Encoder
    p.add_argument("--encoder", type=str, default="ast", choices=["ast"], help="Patch encoder type. (Currently only 'ast' is implemented.)")
    p.add_argument("--model", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593", help="HF model id or local path.")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    p.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    p.add_argument("--attn_impl", type=str, default="", help="Optional attention implementation (e.g., 'sdpa').")

    # Layers
    p.add_argument("--use_layers", type=str, default="all", help="Layer spec: 'all' or e.g. '1-3,5,7'.")
    # Default is True to align with the paper's reported L=11 (HF AST models typically have 12 layers).
    p.add_argument(
        "--exclude_last_layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude the last transformer layer (default: True).",
    )

    # Support selection
    p.add_argument("--support_domain", type=str, default="target", choices=["source", "target"], help="Domain to build support set from.")
    p.add_argument("--support_n", type=int, default=10, help="Number of normal support clips per section (few-shot). Use -1 for all.")
    p.add_argument("--support_sampling", type=str, default="random", choices=["first", "random"], help="How to select support clips if more than support_n.")
    p.add_argument("--support_seed", type=int, default=0, help="Seed for deterministic support sampling.")

    # Test selection
    p.add_argument("--test_domain", type=str, default="all", choices=["source", "target", "all"], help="Which test domains to score.")

    # Scoring
    p.add_argument("--decision_quantile", type=float, default=0.05, help="Tail fraction used for quantile pooling of anomaly map.")
    p.add_argument("--quantile_tail", type=str, default="upper", choices=["upper", "lower"], help="Upper-tail (1-q) vs lower-tail (q) pooling.")

    # Performance / memory
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for encoder forward passes.")
    p.add_argument("--sim_chunk_size", type=int, default=4096, help="Chunk size over key bank for similarity computation.")
    p.add_argument("--max_keys_per_layer", type=int, default=-1, help="If >0, subsample keys per layer to this count.")
    p.add_argument("--key_subsample", type=str, default="random", choices=["random", "first"], help="Subsample strategy if max_keys_per_layer is set.")
    p.add_argument("--key_subsample_seed", type=int, default=0, help="Seed for key subsampling.")

    # Output control
    p.add_argument("--save_maps", action="store_true", help="Save anomaly maps (.npz) for each scored clip (can be large).")
    p.add_argument("--pauc_max_fpr", type=float, default=0.1, help="Max FPR for pAUC.")
    p.add_argument("--no_eval", action="store_true", help="Do not compute AUC/pAUC (still writes scores).")

    # Filtering
    p.add_argument("--machine_types", type=str, default="", help="Comma-separated subset of machine types. Default: all.")
    p.add_argument("--section_ids", type=str, default="", help="Comma-separated subset of section ids. Default: all.")

    # Misc
    p.add_argument("--verbosity", type=int, default=1, help="0=warnings,1=info,2=debug")

    return p.parse_args(argv)


def _write_scores_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Still write header for consistency
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("\n")
        return

    # Stable field order
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


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logger = _setup_logger(int(args.verbosity))

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ToyASDDataset.load(data_root)

    # Filter machine types
    if str(args.machine_types).strip():
        mts = [m.strip() for m in str(args.machine_types).split(",") if m.strip()]
    else:
        mts = _unique_sorted([m.machine_type for m in ds.metas])

    # Filter section ids (optionally); otherwise build per-machine section lists
    user_sids: Optional[List[int]] = None
    user_sids_set: Optional[set[int]] = None
    if str(args.section_ids).strip():
        user_sids = [int(s.strip()) for s in str(args.section_ids).split(",") if s.strip()]
        user_sids_set = set(user_sids)

    sections_by_mt: Dict[str, List[int]] = {}
    for mt in mts:
        mt_sids = _unique_sorted([m.section_id for m in ds.metas if m.machine_type == mt])
        if user_sids_set is not None:
            mt_sids = [sid for sid in mt_sids if sid in user_sids_set]
        sections_by_mt[mt] = mt_sids

    logger.info("Loaded dataset: root=%s, machine_types=%s", str(data_root), mts)
    for mt in mts:
        logger.info("  - %s sections=%s", mt, sections_by_mt.get(mt, []))

    # Config
    support_n = None if int(args.support_n) < 0 else int(args.support_n)
    exclude_last = bool(args.exclude_last_layer)

    cfg = ASTPatchDiffConfig(
        model_name_or_path=str(args.model),
        device=str(args.device),
        dtype=str(args.dtype),
        attn_implementation=(str(args.attn_impl).strip() or None),
        use_layers=str(args.use_layers),
        exclude_last_layer=exclude_last,
        max_keys_per_layer=(None if int(args.max_keys_per_layer) <= 0 else int(args.max_keys_per_layer)),
        key_subsample=str(args.key_subsample),
        key_subsample_seed=int(args.key_subsample_seed),
        sim_chunk_size=int(args.sim_chunk_size),
        decision_quantile=float(args.decision_quantile),
        quantile_tail=str(args.quantile_tail),
        batch_size=int(args.batch_size),
    )
    cfg.validate()

    # Encoder
    if str(args.encoder).lower() != "ast":
        raise ValueError("Only --encoder ast is supported in this implementation.")

    encoder = FrozenASTEncoder(cfg, logger=logger)

    # Output buffers
    score_rows: List[Dict[str, Any]] = []
    section_metrics: List[Dict[str, Any]] = []

    # Evaluate per (machine_type, section_id)
    for mt in mts:
        for sid in sections_by_mt.get(mt, []):
            # Support: train, chosen domain
            support_all = ds.filter(machine_types=[mt], section_ids=[sid], split="train", domain=args.support_domain, is_anomaly=False)
            if len(support_all) == 0:
                logger.warning("No support clips for machine=%s section=%s domain=%s; skipping.", mt, sid, args.support_domain)
                continue

            support = _select_support(
                support_all,
                n=support_n,
                mode=str(args.support_sampling),
                seed=int(args.support_seed),
            )

            # Test
            test_domains: List[str]
            if args.test_domain == "all":
                test_domains = ["source", "target"]
            else:
                test_domains = [str(args.test_domain)]

            test_metas: List[ClipMeta] = []
            for d in test_domains:
                test_metas.extend(ds.filter(machine_types=[mt], section_ids=[sid], split="test", domain=d))

            if len(test_metas) == 0:
                logger.warning("No test clips for machine=%s section=%s test_domain=%s; skipping.", mt, sid, args.test_domain)
                continue

            # Fit + score
            detector = ASTPatchDiffDetector(cfg, encoder, logger=logger)
            detector.fit_support(support, ds)

            scores, maps = detector.score_clips(test_metas, ds, return_maps=bool(args.save_maps))

            # Save maps if requested
            if args.save_maps and maps is not None:
                maps_dir = out_dir / "maps" / mt / f"section_{sid:02d}"
                maps_dir.mkdir(parents=True, exist_ok=True)
                grid = encoder.patch_grid
                for m, s, amap in zip(test_metas, scores, maps):
                    # Save both flat and (if possible) 2D map
                    out_path = maps_dir / f"{m.clip_id}.npz"
                    payload: Dict[str, Any] = {
                        "clip_id": m.clip_id,
                        "score": float(s),
                        "anomaly_map_flat": amap.astype(np.float32, copy=False),
                    }
                    if grid is not None and amap.size == int(grid[0] * grid[1]):
                        payload["anomaly_map_2d"] = amap.reshape(grid[0], grid[1]).astype(np.float32, copy=False)
                        payload["patch_grid"] = np.array(grid, dtype=np.int32)
                    np.savez_compressed(str(out_path), **payload)

            # Record scores
            for m, s in zip(test_metas, scores):
                score_rows.append(
                    {
                        "clip_id": m.clip_id,
                        "relative_path": m.relative_path,
                        "machine_type": m.machine_type,
                        "section_id": int(m.section_id),
                        "split": m.split,
                        "domain": "" if m.domain is None else m.domain,
                        "is_anomaly": int(m.is_anomaly),
                        "label": "" if m.label is None else m.label,
                        "score": float(s),
                        "support_domain": str(args.support_domain),
                        "support_n": (len(support) if support_n is not None else len(support_all)),
                        "decision_quantile": float(cfg.decision_quantile),
                        "quantile_tail": str(cfg.quantile_tail),
                        "use_layers": str(cfg.use_layers),
                        "exclude_last_layer": int(cfg.exclude_last_layer),
                        "model": str(cfg.model_name_or_path),
                        "device": str(cfg.device),
                        "dtype": str(cfg.dtype),
                    }
                )

            # Metrics (per domain and overall)
            if not bool(args.no_eval):
                y_true_all = np.array([1 if m.is_anomaly else 0 for m in test_metas], dtype=np.int64)
                auc_all = _safe_auc(y_true_all, scores)
                pauc_all = _safe_pauc(y_true_all, scores, max_fpr=float(args.pauc_max_fpr))

                # Per-domain
                metrics_dom: Dict[str, Dict[str, Optional[float]]] = {}
                for d in test_domains:
                    idx = [i for i, m in enumerate(test_metas) if m.domain == d]
                    if not idx:
                        continue
                    y_true = y_true_all[idx]
                    sc = scores[idx]
                    metrics_dom[d] = {
                        "auc": _safe_auc(y_true, sc),
                        "pauc": _safe_pauc(y_true, sc, max_fpr=float(args.pauc_max_fpr)),
                        "n": int(len(idx)),
                        "n_anom": int(y_true.sum()),
                    }

                section_metrics.append(
                    {
                        "machine_type": mt,
                        "section_id": int(sid),
                        "support_domain": str(args.support_domain),
                        "support_n": int(len(support)),
                        "test_domain": str(args.test_domain),
                        "n_test": int(len(test_metas)),
                        "auc": auc_all,
                        "pauc": pauc_all,
                        "per_domain": metrics_dom,
                    }
                )

            logger.info(
                "Finished machine=%s section=%02d: support=%d, test=%d",
                mt,
                int(sid),
                int(len(support)),
                int(len(test_metas)),
            )

    # Write outputs
    scores_csv = out_dir / "scores.csv"
    _write_scores_csv(scores_csv, score_rows)

    if not bool(args.no_eval):
        # Aggregate metrics
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
