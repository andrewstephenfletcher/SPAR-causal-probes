"""
Activation extraction for Experiment 1.

For each prompt we run TWO forward passes through the TARGET model:
  1. Self-prefill  — target model's own response formatted as a complete
                     assistant turn → serves as Condition A (natural gen) and B.
  2. Cross-prefill — source model's response in the target model's template
                     → Condition C.

Both passes extract the hidden-state vector at the LAST TOKEN OF THE RESPONSE
CONTENT (i.e. the last non-special token of the assistant turn) from every
transformer layer.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Prefill construction
# ---------------------------------------------------------------------------

def build_prefill_input(
    tokenizer,
    instruction: str,
    response_text: str,
) -> tuple[torch.Tensor, int, int]:
    """
    Tokenize a complete conversation (system + user + assistant) without
    adding a generation prompt.

    Returns
    -------
    prefill_ids : torch.Tensor, shape (1, seq_len)
    last_response_pos : int
        Index of the last CONTENT token of the response (before any trailing
        special tokens added by the chat template, e.g. <|eot_id|>).
    n_response_tokens : int
        Number of response tokens when tokenized standalone (no special tokens).
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response_text},
    ]
    prefill_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prefill_ids = tokenizer(prefill_text, return_tensors="pt")["input_ids"]

    # Count response tokens tokenized standalone (proxy for length in context)
    response_only_ids = tokenizer(
        response_text, add_special_tokens=False
    )["input_ids"]
    n_response_tokens = len(response_only_ids)

    # Determine how many trailing special tokens the template appends after
    # the assistant content.  For Llama-3.1 the template adds <|eot_id|>.
    # We detect this by checking whether the last token is a special token.
    n_trailing = _count_trailing_special_tokens(tokenizer, prefill_ids)
    last_response_pos = prefill_ids.shape[1] - 1 - n_trailing

    return prefill_ids, last_response_pos, n_response_tokens


def _count_trailing_special_tokens(tokenizer, prefill_ids: torch.Tensor) -> int:
    """
    Walk backward from the end of prefill_ids counting how many consecutive
    special tokens appear before the response content.  Stops at the first
    non-special token or after checking 4 positions.
    """
    all_special_ids = set(tokenizer.all_special_ids)
    n_trailing = 0
    seq_len = prefill_ids.shape[1]
    for offset in range(1, 5):
        pos = seq_len - offset
        if pos < 0:
            break
        tok_id = prefill_ids[0, pos].item()
        if tok_id in all_special_ids:
            n_trailing = offset
        else:
            break
    return n_trailing


# ---------------------------------------------------------------------------
# Position verification (printed for the first N examples)
# ---------------------------------------------------------------------------

def verify_extraction_positions(
    tokenizer,
    responses: list[dict],
    n_examples: int = 5,
) -> bool:
    """
    Decode the token at last_response_pos and compare it to the expected last
    token of the response.  Prints a table and returns True if all match.
    """
    print("\n=== Position Verification (first {} examples) ===".format(n_examples))
    all_ok = True
    for r in responses[:n_examples]:
        instruction = r["instruction"]
        response_text = r["response_target"]

        prefill_ids, last_response_pos, n_response_tokens = build_prefill_input(
            tokenizer, instruction, response_text
        )

        extracted_id = prefill_ids[0, last_response_pos].item()
        extracted_tok = tokenizer.decode([extracted_id])

        response_ids = tokenizer(
            response_text, add_special_tokens=False
        )["input_ids"]
        expected_id = response_ids[-1]
        expected_tok = tokenizer.decode([expected_id])

        match = extracted_id == expected_id
        status = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False

        print(
            f"  pid={r['prompt_id']:3d}  pos={last_response_pos:5d}  "
            f"extracted='{extracted_tok}' (id={extracted_id})  "
            f"expected='{expected_tok}' (id={expected_id})  [{status}]"
        )

        if not match:
            tail_ids = prefill_ids[0, -6:].tolist()
            tail_toks = [tokenizer.decode([t]) for t in tail_ids]
            print(f"    Last 6 token ids: {tail_ids}")
            print(f"    Last 6 tokens:    {tail_toks}")

    if not all_ok:
        raise RuntimeError(
            "Extraction position verification FAILED. "
            "The last_response_pos does not point to the expected token. "
            "Check _count_trailing_special_tokens() for this model's chat template."
        )

    print("  All positions verified OK.\n")
    return True


# ---------------------------------------------------------------------------
# Reproducibility check (2 identical forward passes → L2 should be ~0)
# ---------------------------------------------------------------------------

def check_reproducibility(
    model,
    tokenizer,
    responses: list[dict],
    config: Config,
    n_examples: int = 10,
) -> float:
    """
    Run two identical forward passes for the first n_examples prompts and
    compute the max L2 distance between activations.  Should be < 1e-6.
    Returns the maximum L2 distance observed.
    """
    print("\n=== Reproducibility Check ===")
    device = next(model.parameters()).device
    max_l2 = 0.0

    check_layers = [0, config.extract_layers[len(config.extract_layers) // 2],
                    config.extract_layers[-1]]

    for r in responses[:n_examples]:
        acts1, pos1, _ = _run_forward_pass(
            model, tokenizer, r["instruction"], r["response_target"],
            config, device
        )
        acts2, pos2, _ = _run_forward_pass(
            model, tokenizer, r["instruction"], r["response_target"],
            config, device
        )
        assert pos1 == pos2, "Position mismatch between two passes!"
        for layer_idx in check_layers:
            a1 = acts1[layer_idx].astype(np.float32)
            a2 = acts2[layer_idx].astype(np.float32)
            l2 = float(np.linalg.norm(a1 - a2))
            if l2 > max_l2:
                max_l2 = l2

    print(f"  Max L2 distance across {n_examples} prompts x {len(check_layers)} "
          f"layers: {max_l2:.2e}")
    if max_l2 < 1e-5:
        print("  PASS: activations are numerically consistent across passes.\n")
    else:
        print("  WARNING: activations differ between passes! "
              "Check for nondeterminism in model operations.\n")

    return max_l2


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _run_forward_pass(
    model,
    tokenizer,
    instruction: str,
    response_text: str,
    config: Config,
    device,
) -> tuple[dict, int, int]:
    """
    Single forward pass with hooks. Returns (layer_activations, last_response_pos, n_response_tokens).
    layer_activations is a dict mapping layer_idx -> np.ndarray (float16, shape (hidden_dim,)).
    """
    prefill_ids, last_response_pos, n_response_tokens = build_prefill_input(
        tokenizer, instruction, response_text
    )
    prefill_ids = prefill_ids.to(device)

    activations: dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output[0]: (batch=1, seq_len, hidden_dim)
            hidden = output[0]
            activations[layer_idx] = (
                hidden[0, last_response_pos, :]
                .detach()
                .float()  # accumulate in float32 on MPS/CPU, cast to float16 after
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        return hook_fn

    for i in config.extract_layers:
        h = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(prefill_ids)

    for h in hooks:
        h.remove()

    return activations, last_response_pos, n_response_tokens


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_all_activations(
    responses: list[dict],
    config: Config,
    force: bool = False,
) -> None:
    """
    Extract activations for all responses under two conditions:
      - 'self':        target model's own response prefilled
      - 'cross_gemma': source model's (Gemma) response prefilled in Llama's template

    Saves:
      activations_dir/self_prefill.pt
      activations_dir/cross_gemma_prefill.pt
      activations_dir/reproducibility_check.json
    """
    self_path = config.activations_dir / "self_prefill.pt"
    cross_path = config.activations_dir / "cross_gemma_prefill.pt"
    repro_path = config.activations_dir / "reproducibility_check.json"

    if self_path.exists() and cross_path.exists() and not force:
        print(f"  Found existing activations, skipping extraction.")
        return

    # Use deterministic mode where supported (warn_only for MPS compatibility)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading target model (Llama 3.1 8B) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            config.target_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.target_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.target_model_id)
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Verify extraction positions on first 5 prompts
    # ------------------------------------------------------------------
    verify_extraction_positions(tokenizer, responses, n_examples=5)

    # ------------------------------------------------------------------
    # Reproducibility check: two identical passes → L2 should be ~0
    # ------------------------------------------------------------------
    max_l2 = check_reproducibility(model, tokenizer, responses, config, n_examples=10)
    with open(repro_path, "w") as f:
        json.dump({"max_l2_distance": max_l2, "pass": max_l2 < 1e-5}, f, indent=2)

    # ------------------------------------------------------------------
    # Extract activations for all prompts
    # ------------------------------------------------------------------
    self_activations = []
    cross_activations = []

    print("  Extracting activations (two passes per prompt)...")
    for r in tqdm(responses, desc="Extracting"):
        pid = r["prompt_id"]
        instruction = r["instruction"]

        # Condition: self-prefill (target model's own response)
        self_acts, self_pos, self_n = _run_forward_pass(
            model, tokenizer, instruction, r["response_target"], config, device
        )
        self_activations.append({
            "prompt_id": pid,
            "condition": "self",
            "layer_activations": self_acts,
            "last_response_pos": self_pos,
            "n_response_tokens": self_n,
        })

        # Condition: cross-model prefill (Gemma response in Llama's template)
        cross_acts, cross_pos, cross_n = _run_forward_pass(
            model, tokenizer, instruction, r["response_source"], config, device
        )
        cross_activations.append({
            "prompt_id": pid,
            "condition": "cross_gemma",
            "layer_activations": cross_acts,
            "last_response_pos": cross_pos,
            "n_response_tokens": cross_n,
        })

    torch.save(self_activations, self_path)
    torch.save(cross_activations, cross_path)
    print(f"  Saved self-prefill activations  → {self_path}")
    print(f"  Saved cross-model activations   → {cross_path}")

    # Clean up
    del model
    clear_device_cache()
