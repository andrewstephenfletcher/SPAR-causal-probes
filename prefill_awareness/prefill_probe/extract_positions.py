"""
Activation extraction at multiple (position, layer) cells for Experiment 2.

For each prompt we run TWO forward passes through Llama 3.1 8B Instruct:
  - Self-prefill:  Llama's own response prefilled into its own template.
  - Cross-prefill: Gemma's response prefilled into Llama's template.

At each pass we:
  1. Locate the first assistant response token using the assistant-header
     marker as a subsequence search in the tokenized input.
  2. Register forward hooks on the specified layers to capture hidden-state
     vectors at every target (absolute) position in one shot.
  3. Compute per-token log-probs for relative positions 0 through max_pos
     from the model's output logits, for use as a perplexity baseline.

Position 0 is the first content token of the assistant response (immediately
after <|start_header_id|>assistant<|end_header_id|>\n\n in Llama 3.1).
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, Experiment2Config
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Finding the first response token
# ---------------------------------------------------------------------------

def find_assistant_content_start(
    input_ids: torch.Tensor,
    tokenizer,
) -> int:
    """
    Return the index of the first assistant response token.

    Searches for the LAST occurrence of the assistant-header marker
    ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` as an exact
    subsequence in the tokenized input, then returns the index right after it.

    Searching for the last occurrence guards against the (unlikely) case
    where the user instruction contains the marker text.

    Raises ValueError if the marker is not found.
    """
    marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]

    input_list = input_ids[0].tolist()
    n, m = len(input_list), len(marker_ids)

    last_found: int | None = None
    for i in range(n - m + 1):
        if input_list[i: i + m] == marker_ids:
            last_found = i

    if last_found is None:
        raise ValueError(
            f"Assistant header marker not found in tokenized input. "
            f"Marker ids: {marker_ids}. "
            f"First 30 input token ids: {input_list[:30]}"
        )

    return last_found + m


def _first_response_idx_by_prefix(tokenizer, instruction: str) -> int:
    """
    Cross-check method: count prefix tokens (system + user + generation prompt)
    to locate the first response token.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    prefix_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prefix_text, return_tensors="pt")["input_ids"].shape[1]


# ---------------------------------------------------------------------------
# Position verification
# ---------------------------------------------------------------------------

def verify_position_alignment(
    tokenizer,
    responses: list[dict],
    n_examples: int = 5,
) -> None:
    """
    For the first n_examples prompts, print the decoded token at
    first_response_idx (from marker search) and compare it to the expected
    first token of the response.

    Also cross-checks with prefix-counting to detect any discrepancy.
    Raises ValueError if more than 1 out of 5 examples mismatches.
    """
    print(f"\n=== Position Alignment Verification (first {n_examples} examples) ===")
    mismatches = 0

    for r in responses[:n_examples]:
        instruction = r["instruction"]
        response_text = r["response_target"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

        try:
            marker_idx = find_assistant_content_start(input_ids, tokenizer)
        except ValueError as exc:
            print(f"  pid={r['prompt_id']:3d}: marker search FAILED — {exc}")
            mismatches += 1
            continue

        prefix_idx = _first_response_idx_by_prefix(tokenizer, instruction)

        # Decode the token at marker_idx
        tok_id = input_ids[0, marker_idx].item()
        tok_str = repr(tokenizer.decode([tok_id]))

        # Expected: first standalone token of the response
        resp_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        exp_id = resp_ids[0] if resp_ids else None
        exp_str = repr(tokenizer.decode([exp_id])) if exp_id is not None else "<empty>"

        match = tok_id == exp_id
        cross = f"prefix={prefix_idx}" + ("" if prefix_idx == marker_idx else f"≠marker={marker_idx}")
        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches += 1

        print(
            f"  pid={r['prompt_id']:3d}  first_idx={marker_idx}  {cross}  "
            f"decoded={tok_str}(id={tok_id})  expected={exp_str}(id={exp_id})  [{status}]"
        )

    if mismatches > 1:
        raise ValueError(
            f"Position alignment FAILED: {mismatches}/{n_examples} mismatches. "
            "Adjust find_assistant_content_start() for this tokenizer."
        )
    elif mismatches == 1:
        print("  WARNING: 1 mismatch — likely a BPE boundary edge case. Proceeding.")
    else:
        print("  All positions verified OK.\n")


# ---------------------------------------------------------------------------
# Single-prompt extraction
# ---------------------------------------------------------------------------

def _extract_one_prompt(
    model,
    tokenizer,
    instruction: str,
    response_text: str,
    ex2_config: Experiment2Config,
    device,
) -> dict | None:
    """
    Run one forward pass for a single (instruction, response) pair.

    Returns a dict with:
      activations: {(layer, relative_pos): np.ndarray (float16, shape (d_model,))}
      log_probs_per_response_token: List[float], length = max_pos + 1

    Returns None if the response is too short or the marker cannot be found.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response_text},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

    # Find first response token
    try:
        first_idx = find_assistant_content_start(input_ids, tokenizer)
    except ValueError:
        return None

    max_pos = max(ex2_config.positions)

    # Safety: check that the sequence is long enough
    # Need first_idx + max_pos to be a valid index (and not land on the EOT token)
    if first_idx + max_pos >= input_ids.shape[1]:
        return None

    # Absolute positions for each relative target position
    abs_positions = [first_idx + p for p in ex2_config.positions]

    # ------------------------------------------------------------------ #
    # Register hooks to capture activations at every (layer, abs_pos)
    # ------------------------------------------------------------------ #
    activation_store: dict[tuple[int, int], torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int, target_abs_positions: list[int]):
        def hook_fn(module, input, output):
            hidden = output[0]  # (1, seq_len, d_model)
            for abs_pos in target_abs_positions:
                activation_store[(layer_idx, abs_pos)] = (
                    hidden[0, abs_pos, :].detach().cpu().half()
                )
        return hook_fn

    for layer_idx in ex2_config.layers:
        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx, abs_positions)
        )
        hooks.append(h)

    # ------------------------------------------------------------------ #
    # Forward pass — also capture logits for log-prob computation
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    for h in hooks:
        h.remove()

    # Compute per-token log-probs for relative positions 0 through max_pos
    # logits[0, k-1, :] predicts the token at position k
    log_probs: list[float] = []
    for p in range(max_pos + 1):
        abs_pos = first_idx + p
        if abs_pos >= input_ids.shape[1]:
            log_probs.append(float("nan"))
        else:
            tok_id = input_ids[0, abs_pos].item()
            lp = float(
                F.log_softmax(logits[0, abs_pos - 1].float(), dim=-1)[tok_id].item()
            )
            log_probs.append(lp)

    del logits, outputs  # free the large logits tensor promptly

    # Remap activation_store keys to (layer, relative_position)
    activations: dict[tuple[int, int], np.ndarray] = {}
    for layer_idx in ex2_config.layers:
        for rel_pos, abs_pos in zip(ex2_config.positions, abs_positions):
            key = (layer_idx, abs_pos)
            if key in activation_store:
                activations[(layer_idx, rel_pos)] = activation_store[key].numpy()

    return {
        "activations": activations,
        "log_probs_per_response_token": log_probs,
    }


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_all_position_activations(
    responses: list[dict],
    ex2_config: Experiment2Config,
    ex1_config: Config,
    force: bool = False,
) -> None:
    """
    Extract activations at all (layer, position) cells for both conditions.

    Saves:
      activations_dir_ex2/self_prefill_positions.pt
      activations_dir_ex2/cross_gemma_prefill_positions.pt

    Each file is a list of dicts (one per prompt) with keys:
      prompt_id, split, condition, activations, log_probs_per_response_token
    """
    self_path = ex2_config.activations_dir_ex2 / "self_prefill_positions.pt"
    cross_path = ex2_config.activations_dir_ex2 / "cross_gemma_prefill_positions.pt"

    if self_path.exists() and cross_path.exists() and not force:
        print(f"  Found existing Experiment 2 activations, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading target model ({ex1_config.target_model_id}) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            ex1_config.target_model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ex1_config.target_model_id,
            torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ex1_config.target_model_id)
    device = next(model.parameters()).device

    # Verify position alignment on first 5 examples before bulk extraction
    verify_position_alignment(tokenizer, responses, n_examples=5)

    self_records: list[dict] = []
    cross_records: list[dict] = []
    skipped = 0

    print(f"  Extracting activations at {len(ex2_config.layers)} layers × "
          f"{len(ex2_config.positions)} positions for {len(responses)} prompts...")

    for r in tqdm(responses, desc="Extracting positions"):
        pid = r["prompt_id"]
        split = r["split"]
        instruction = r["instruction"]

        # Condition: self-prefill (Llama's own response)
        self_result = _extract_one_prompt(
            model, tokenizer, instruction, r["response_target"],
            ex2_config, device,
        )
        if self_result is None:
            skipped += 1
            continue

        # Condition: cross-model prefill (Gemma's response in Llama's template)
        cross_result = _extract_one_prompt(
            model, tokenizer, instruction, r["response_source"],
            ex2_config, device,
        )
        if cross_result is None:
            skipped += 1
            continue

        self_records.append({
            "prompt_id": pid,
            "split": split,
            "condition": "self",
            **self_result,
        })
        cross_records.append({
            "prompt_id": pid,
            "split": split,
            "condition": "cross_gemma",
            **cross_result,
        })

    print(f"  Extracted {len(self_records)} prompts ({skipped} skipped — "
          "responses too short or marker not found).")

    torch.save(self_records, self_path)
    torch.save(cross_records, cross_path)
    print(f"  Saved → {self_path}")
    print(f"  Saved → {cross_path}")

    del model
    clear_device_cache()
