"""
Activation store: cache and load hidden states from LIARS' BENCH examples.

Run extract_activations.py once per model to cache hidden states at the
DCT source and target layers. Then load them cheaply for probe evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file


class ActivationStore:
    """Manages cached hidden states on disk.

    Directory layout:
        base_dir/
            {model_name}/
                {dataset_name}/
                    activations.safetensors
                    metadata.json
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def _dir(self, model_name: str, dataset_name: str) -> Path:
        safe_model = model_name.replace("/", "--")
        return self.base_dir / safe_model / dataset_name

    def save(
        self,
        model_name: str,
        dataset_name: str,
        hidden_states: dict[int, torch.Tensor],
        metadata: list[dict],
    ) -> None:
        save_dir = self._dir(model_name, dataset_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        tensors = {}
        for layer_idx, states in hidden_states.items():
            tensors[f"layer_{layer_idx}"] = states
        save_file(tensors, save_dir / "activations.safetensors")

        with open(save_dir / "metadata.json", "w") as f:
            json.dump({
                "model_name": model_name,
                "dataset_name": dataset_name,
                "layers": list(hidden_states.keys()),
                "n_examples": len(metadata),
                "examples": metadata,
            }, f, indent=2)

    def load(
        self,
        model_name: str,
        dataset_name: str,
        layers: list[int] | None = None,
    ) -> tuple[dict[int, torch.Tensor], list[dict]]:
        load_dir = self._dir(model_name, dataset_name)

        with open(load_dir / "metadata.json") as f:
            meta = json.load(f)

        all_tensors = load_file(load_dir / "activations.safetensors")
        hidden_states = {}
        available_layers = meta["layers"]

        if layers is None:
            layers = available_layers

        for layer_idx in layers:
            key = f"layer_{layer_idx}"
            if key in all_tensors:
                hidden_states[layer_idx] = all_tensors[key]
            else:
                raise KeyError(
                    f"Layer {layer_idx} not found. Available: {available_layers}"
                )

        return hidden_states, meta["examples"]

    def exists(self, model_name: str, dataset_name: str) -> bool:
        d = self._dir(model_name, dataset_name)
        return (d / "activations.safetensors").exists() and (d / "metadata.json").exists()

    def list_datasets(self, model_name: str) -> list[str]:
        safe_model = model_name.replace("/", "--")
        model_dir = self.base_dir / safe_model
        if not model_dir.exists():
            return []
        return [d.name for d in model_dir.iterdir()
                if d.is_dir() and (d / "metadata.json").exists()]
