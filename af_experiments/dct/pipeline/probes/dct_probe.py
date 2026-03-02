"""
Probe definitions: dataclasses for storing and loading DCT-derived probes.

A DCT probe consists of:
  - v: source-layer direction (the direction added during steering)
  - u: target-layer direction (the downstream effect direction)
  - alpha: importance scalar from DCT optimization
  - metadata about how it was trained
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np


@dataclass
class DCTProbe:
    """A single DCT feature usable as a probe.

    The probe can be applied at either the source layer (using v) or the
    target layer (using u). The score for a hidden state h is simply h @ direction.
    """

    # Core vectors
    v: torch.Tensor  # (d_model,) source-layer direction, unit norm
    u: torch.Tensor  # (d_model,) target-layer direction, unit norm
    alpha: float  # importance scalar

    # Layer info
    source_layer: int  # e.g. 9 for Gemma2-9B
    target_layer: int  # e.g. 19

    # Model info
    model_name: str  # e.g. "google/gemma-2-9b-it"

    # DCT training info
    feature_index: int  # which feature out of dct_width
    dct_width: int  # total features trained (e.g. 512)
    dct_iterations: int  # tau (e.g. 10)
    R_cal: float  # calibrated steering scale
    training_prompts: list[str] = field(default_factory=list)

    # Optional: LLM judge characterization
    judge_label: Optional[str] = None
    judge_confidence: Optional[float] = None

    # Optional: any extra metadata
    extra: dict = field(default_factory=dict)

    def score(
        self,
        hidden_states: torch.Tensor,
        layer: str = "source",
    ) -> torch.Tensor:
        """Compute probe scores for hidden states.

        Args:
            hidden_states: (..., d_model) tensor of activations
            layer: "source" to probe with v, "target" to probe with u

        Returns:
            (...,) tensor of scalar scores
        """
        direction = self.v if layer == "source" else self.u
        direction = direction.to(hidden_states.device, hidden_states.dtype)
        return hidden_states @ direction

    def save(self, path: str | Path) -> None:
        """Save probe to disk as a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "v": self.v.cpu(),
            "u": self.u.cpu(),
            "alpha": self.alpha,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "model_name": self.model_name,
            "feature_index": self.feature_index,
            "dct_width": self.dct_width,
            "dct_iterations": self.dct_iterations,
            "R_cal": self.R_cal,
            "training_prompts": self.training_prompts,
            "judge_label": self.judge_label,
            "judge_confidence": self.judge_confidence,
            "extra": self.extra,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path) -> DCTProbe:
        """Load probe from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            v=data["v"],
            u=data["u"],
            alpha=data["alpha"],
            source_layer=data["source_layer"],
            target_layer=data["target_layer"],
            model_name=data["model_name"],
            feature_index=data["feature_index"],
            dct_width=data["dct_width"],
            dct_iterations=data["dct_iterations"],
            R_cal=data["R_cal"],
            training_prompts=data.get("training_prompts", []),
            judge_label=data.get("judge_label"),
            judge_confidence=data.get("judge_confidence"),
            extra=data.get("extra", {}),
        )

    def to_config_dict(self) -> dict:
        """Return a JSON-serializable dict for W&B config logging."""
        return {
            "probe.type": "dct_direction",
            "probe.source_layer": self.source_layer,
            "probe.target_layer": self.target_layer,
            "probe.feature_index": self.feature_index,
            "probe.alpha": self.alpha,
            "probe.dct_width": self.dct_width,
            "probe.dct_iterations": self.dct_iterations,
            "probe.R_cal": self.R_cal,
            "probe.training_prompts": self.training_prompts,
            "probe.judge_label": self.judge_label,
            "probe.judge_confidence": self.judge_confidence,
            "probe.cosine_sim_v_u": float(
                torch.nn.functional.cosine_similarity(
                    self.v.unsqueeze(0), self.u.unsqueeze(0)
                ).item()
            ),
            "model": self.model_name,
        }


@dataclass
class DCTProbeCollection:
    """A collection of DCT probes from a single training run."""

    probes: list[DCTProbe]

    # Shared training config
    model_name: str
    source_layer: int
    target_layer: int
    dct_width: int
    dct_iterations: int
    R_cal: float
    training_prompts: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save entire collection to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "model_name": self.model_name,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "dct_width": self.dct_width,
            "dct_iterations": self.dct_iterations,
            "R_cal": self.R_cal,
            "training_prompts": self.training_prompts,
            "num_probes": len(self.probes),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        V = torch.stack([p.v for p in self.probes])
        U = torch.stack([p.u for p in self.probes])
        alphas = torch.tensor([p.alpha for p in self.probes])
        torch.save({"V": V, "U": U, "alphas": alphas}, path / "vectors.pt")

        probe_meta = []
        for p in self.probes:
            probe_meta.append({
                "feature_index": p.feature_index,
                "judge_label": p.judge_label,
                "judge_confidence": p.judge_confidence,
                "extra": p.extra,
            })
        with open(path / "probe_metadata.json", "w") as f:
            json.dump(probe_meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DCTProbeCollection:
        """Load collection from a directory."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        vectors = torch.load(path / "vectors.pt", map_location="cpu", weights_only=True)
        V, U, alphas = vectors["V"], vectors["U"], vectors["alphas"]

        with open(path / "probe_metadata.json") as f:
            probe_meta = json.load(f)

        probes = []
        for i, pm in enumerate(probe_meta):
            probes.append(
                DCTProbe(
                    v=V[i], u=U[i], alpha=float(alphas[i]),
                    source_layer=meta["source_layer"],
                    target_layer=meta["target_layer"],
                    model_name=meta["model_name"],
                    feature_index=pm["feature_index"],
                    dct_width=meta["dct_width"],
                    dct_iterations=meta["dct_iterations"],
                    R_cal=meta["R_cal"],
                    training_prompts=meta.get("training_prompts", []),
                    judge_label=pm.get("judge_label"),
                    judge_confidence=pm.get("judge_confidence"),
                    extra=pm.get("extra", {}),
                )
            )

        return cls(
            probes=probes,
            model_name=meta["model_name"],
            source_layer=meta["source_layer"],
            target_layer=meta["target_layer"],
            dct_width=meta["dct_width"],
            dct_iterations=meta["dct_iterations"],
            R_cal=meta["R_cal"],
            training_prompts=meta.get("training_prompts", []),
        )

    def filter_by_label(self, label: str) -> list[DCTProbe]:
        """Get all probes with a specific judge label."""
        return [p for p in self.probes if p.judge_label == label]

    def __len__(self) -> int:
        return len(self.probes)

    def __getitem__(self, idx: int) -> DCTProbe:
        return self.probes[idx]
