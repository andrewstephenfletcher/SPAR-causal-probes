"""
2D (layer × position) activation extraction for Experiment 8.

For each model (Mistral 24B, Gemma 31B) and each prompt we run TWO forward
passes:
  - Self-prefill:  the model's own response prefilled into its template.
  - Cross-prefill: Llama 8B's response prefilled into the model's template.

At each pass we:
  1. Locate the start of the assistant response in the token sequence
     by computing the prefix length (prompt without response).
  2. Register forward hooks at all target layers to capture hidden states
     at every target relative position in one pass.
  3. Compute per-token log-probs for the perplexity baseline.

Saves to activations_dir_{model}/self_prefill_positions.pt
             activations_dir_{model}/cross_prefill_positions.pt

Each file is a list of dicts per prompt:
  {prompt_id, split, condition, activations: {(layer, rel_pos): np.ndarray},
   log_probs_per_response_token: list[float]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import Experiment8Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Response start detection (model-agnostic)
# ---------------------------------------------------------------------------

def _response_start_idx(tokenizer, instruction: str, is_gemma: bool = False) -> int:
    """
    Return the token index where the assistant response begins.

    Uses the apply_chat_template approach: format a prompt WITH generation
    prompt (no response) and count its tokens.  This is the start of the
    response in the full prefilled sequence.
    """
    messages = [{"role": "user", "content": instruction}]
    template_kwargs: dict = {}
    if is_gemma:
        template_kwargs["enable_thinking"] = False

    try:
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
    except TypeError:
        prefix_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return tokenizer(prefix_text, return_tensors="pt")["input_ids"].shape[1]


def _build_full_input(tokenizer, instruction: str, response: str, is_gemma: bool = False) -> torch.Tensor:
    """Tokenize [user instruction] + [assistant response] with chat template."""
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    template_kwargs: dict = {}
    if is_gemma:
        template_kwargs["enable_thinking"] = False

    try:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **template_kwargs
        )
    except TypeError:
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    return tokenizer(full_text, return_tensors="pt")["input_ids"]


# ---------------------------------------------------------------------------
# Single-prompt extraction
# ---------------------------------------------------------------------------

def _extract_one_prompt(
    model,
    tokenizer,
    instruction: str,
    response: str,
    layers: list[int],
    positions: list[int],
    device,
    is_gemma: bool = False,
) -> dict | None:
    """
    Run one forward pass and capture activations at all (layer, position) cells.

    Returns None if the response is too short to cover all positions.
    """
    input_ids = _build_full_input(tokenizer, instruction, response, is_gemma).to(device)
    first_idx = _response_start_idx(tokenizer, instruction, is_gemma)

    max_pos = max(positions)
    if first_idx + max_pos >= input_ids.shape[1]:
        return None

    abs_positions = [first_idx + p for p in positions]

    activation_store: dict[tuple[int, int], np.ndarray] = {}
    hooks = []

    def make_hook(layer_idx: int, abs_pos_list: list[int]):
        def hook_fn(module, inp, output):
            hidden = output[0]
            for abs_pos in abs_pos_list:
                if abs_pos < hidden.shape[1]:
                    activation_store[(layer_idx, abs_pos)] = (
                        hidden[0, abs_pos, :].detach().cpu().half().numpy()
                    )
        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx, abs_positions)
        )
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    for h in hooks:
        h.remove()

    # Per-token log-probs for positions 0..max_pos (perplexity baseline)
    log_probs: list[float] = []
    for p in range(max_pos + 1):
        abs_pos = first_idx + p
        if abs_pos >= input_ids.shape[1] or abs_pos == 0:
            log_probs.append(float("nan"))
        else:
            tok_id = input_ids[0, abs_pos].item()
            lp = float(
                F.log_softmax(logits[0, abs_pos - 1].float(), dim=-1)[tok_id].item()
            )
            log_probs.append(lp)

    del logits, outputs

    activations: dict[tuple[int, int], np.ndarray] = {}
    for layer_idx in layers:
        for rel_pos, abs_pos in zip(positions, abs_positions):
            key = (layer_idx, abs_pos)
            if key in activation_store:
                activations[(layer_idx, rel_pos)] = activation_store[key].astype(np.float32)

    return {
        "activations": activations,
        "log_probs_per_response_token": log_probs,
    }


# ---------------------------------------------------------------------------
# Full extraction for one model
# ---------------------------------------------------------------------------

def extract_model_activations(
    model_id: str,
    responses_path: Path,
    activations_dir: Path,
    config: Experiment8Config,
    layers: list[int],
    force: bool = False,
    is_gemma: bool = False,
) -> None:
    """
    Extract self and cross-prefill activations for one model.

    Saves:
      activations_dir/self_prefill_positions.pt
      activations_dir/cross_prefill_positions.pt
    """
    self_path  = activations_dir / "self_prefill_positions.pt"
    cross_path = activations_dir / "cross_prefill_positions.pt"

    if self_path.exists() and cross_path.exists() and not force:
        print(f"  Found existing activations at {activations_dir}, skipping.")
        return

    if not responses_path.exists():
        raise FileNotFoundError(
            f"Responses not found at {responses_path}. "
            "Run generate_ex8 first."
        )
    with open(responses_path) as f:
        records = json.load(f)

    device_str = get_device()
    device_map = get_device_map()
    print(f"  Loading {model_id} (fp16) on {device_str}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device_str)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device

    self_records:  list[dict] = []
    cross_records: list[dict] = []
    skipped = 0

    print(f"  Extracting {len(layers)} layers × {len(config.positions)} positions "
          f"for {len(records)} prompts...")

    for i, r in enumerate(tqdm(records, desc="Extracting (layer×pos)")):
        pid = r["prompt_id"]
        instr = r["instruction"]

        self_result = _extract_one_prompt(
            model, tokenizer, instr, r["response_self"],
            layers, config.positions, device, is_gemma=is_gemma,
        )
        if self_result is None:
            skipped += 1
            continue

        cross_result = _extract_one_prompt(
            model, tokenizer, instr, r["response_cross_llama8b"],
            layers, config.positions, device, is_gemma=is_gemma,
        )
        if cross_result is None:
            skipped += 1
            continue

        self_records.append({"prompt_id": pid, "split": r["split"], "condition": "self", **self_result})
        cross_records.append({"prompt_id": pid, "split": r["split"], "condition": "cross_llama8b", **cross_result})

        # Checkpoint every N prompts
        if (i + 1) % config.checkpoint_interval == 0:
            torch.save(self_records,  activations_dir / "self_prefill_positions_partial.pt")
            torch.save(cross_records, activations_dir / "cross_prefill_positions_partial.pt")

    print(f"  Extracted {len(self_records)} prompts ({skipped} skipped).")

    torch.save(self_records,  self_path)
    torch.save(cross_records, cross_path)
    print(f"  Saved → {self_path}")
    print(f"  Saved → {cross_path}")

    del model
    import gc
    gc.collect()
    clear_device_cache()


def run_extraction_ex8(config: Experiment8Config, model: str = "both", force: bool = False) -> None:
    """Extract activations for Mistral and/or Gemma 31B."""
    if model in ("mistral", "both"):
        extract_model_activations(
            config.mistral_model_id,
            config.generations_dir_ex8 / "responses_mistral.json",
            config.activations_dir_mistral,
            config,
            config.mistral_layers,
            force=force,
            is_gemma=False,
        )

    if model in ("gemma", "both"):
        extract_model_activations(
            config.gemma31b_model_id,
            config.generations_dir_ex8 / "responses_gemma31b.json",
            config.activations_dir_gemma31b,
            config,
            config.gemma_layers,
            force=force,
            is_gemma=True,
        )
