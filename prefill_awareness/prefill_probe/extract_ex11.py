"""
Activation extraction for Experiment 11 (Unified Prefill-Awareness Data Collection).

Two extraction modes per (target_model, condition, dataset):

1. Full-depth extraction (all layers, last content token):
   outputs/experiment11/activations/{target}/{condition}/{dataset}.pt
   List[dict] with keys:
     prompt_id, condition, layer_activations (dict[int → np.float16 array]),
     last_response_pos, n_response_tokens

2. Token-position extraction (3 layers at 40/60/80% depth, multiple positions):
   outputs/experiment11/activations/{target}/{condition}/{dataset}_token_positions.pt
   List[dict] with keys:
     prompt_id, condition, token_positions (List[int]), layers (List[int]),
     activations (dict[layer_idx → dict[pos → np.float16 array]])

Key: activation extraction uses response_normalized (not response_raw) so all source
models consistently produce '.' as the last content token — eliminating the last-token
confound identified in Experiment 4 (Gemma 2 9B ended with '\\n' 44% of the time).

Reuses from extract_ex4.py:
  build_prefill_input, _count_trailing_special_tokens, get_transformer_layers,
  verify_extraction_positions, _run_forward_pass, _pt_file_is_complete,
  _extract_one_condition, _supports_system_prompt
"""

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Experiment11Config
from .extract_ex4 import (
    _pt_file_is_complete,
    _run_forward_pass,
    _supports_system_prompt,
    build_prefill_input,
    get_transformer_layers,
    verify_extraction_positions,
)
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Token-position schedule
# ---------------------------------------------------------------------------

def _position_schedule(n_response_tokens: int, response_start: int) -> list[int]:
    """
    Compute absolute token positions within the full sequence to extract from.

    Schedule (per plan):
      - Absolute offsets from response start: 0, 1, 4, 9, 19 (0-indexed within response)
        → corresponds to "position 1, 2, 5, 10, 20" in 1-indexed response tokens
      - Percentage offsets: 10%, 20%, ..., 100% of n_response_tokens (0-indexed)
        skipping any that are already covered by the absolute offsets or are < 20 tokens
      - The 100% position = last content token (consistency check)
    All positions are clipped to [response_start, response_start + n_response_tokens - 1].
    Returned list is sorted and deduplicated.
    """
    N = n_response_tokens
    abs_offsets = [0, 1, 4, 9, 19]  # 0-indexed within response (= positions 1,2,5,10,20)
    pct_offsets = [round((pct / 100.0) * (N - 1)) for pct in range(10, 101, 10)]

    # Keep percentage offsets that are ≥ 20 tokens from start (1-indexed: >= 20 means offset >= 19)
    pct_offsets = [o for o in pct_offsets if o >= 19]

    all_offsets = sorted(set(abs_offsets + pct_offsets))
    # Clip to valid range
    all_offsets = [o for o in all_offsets if 0 <= o < N]

    return [response_start + o for o in all_offsets]


# ---------------------------------------------------------------------------
# Token-position extraction
# ---------------------------------------------------------------------------

def _run_forward_pass_multipos(
    model,
    tokenizer,
    model_id: str,
    instruction: str,
    response_normalized: str,
    target_layers: list[int],
    device,
) -> tuple[dict, list[int], int, int]:
    """
    Forward pass with hooks at target_layers, collecting multiple token positions.

    Returns:
        layer_pos_acts: dict[layer_idx → dict[pos → np.float16 array]]
        token_positions: list of absolute positions extracted
        last_response_pos: the last content position (sanity check)
        n_response_tokens: number of response tokens
    """
    prefill_ids, last_response_pos, n_response_tokens = build_prefill_input(
        tokenizer, model_id, instruction, response_normalized
    )
    prefill_ids = prefill_ids.to(device)

    # Compute response_start by tokenizing prompt-only portion
    if _supports_system_prompt(model_id):
        prompt_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]
    else:
        prompt_msgs = [{"role": "user", "content": instruction}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    response_start = prompt_ids.shape[1]

    token_positions = _position_schedule(n_response_tokens, response_start)

    layer_pos_acts: dict[int, dict[int, np.ndarray]] = {l: {} for l in target_layers}
    hooks = []
    transformer_layers = get_transformer_layers(model)

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            for pos in token_positions:
                if pos < hidden.shape[1]:
                    layer_pos_acts[layer_idx][pos] = (
                        hidden[0, pos, :]
                        .detach().float().cpu().numpy().astype(np.float16)
                    )
        return hook_fn

    for l in target_layers:
        hooks.append(transformer_layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(prefill_ids)

    for h in hooks:
        h.remove()

    return layer_pos_acts, token_positions, last_response_pos, n_response_tokens


def _extract_token_positions_one_condition(
    condition_name: str,
    response_key_normalized: str,
    target_model_id: str,
    model,
    tokenizer,
    responses: list[dict],
    activations_dir: Path,
    target_layers: list[int],
    checkpoint_interval: int,
) -> None:
    """Extract token-position activations for one (model, condition) pair."""
    final_path = activations_dir / f"{condition_name}_token_positions.pt"

    if final_path.exists():
        existing = torch.load(final_path, weights_only=False)
        if isinstance(existing, list) and len(existing) > 0:
            print(f"  [{condition_name} token_pos] Already complete ({len(existing)} records), skipping.")
            return

    partial_path = activations_dir / f"{condition_name}_token_positions_partial.pt"
    data_list: list[dict] = []
    completed_pids: set[int] = set()

    if partial_path.exists():
        data_list = torch.load(partial_path, weights_only=False)
        completed_pids = {d["prompt_id"] for d in data_list}
        print(f"  [{condition_name} token_pos] Resuming from {len(completed_pids)}/{len(responses)}")

    device = next(model.parameters()).device
    remaining = [r for r in responses if r["prompt_id"] not in completed_pids]

    for i, r in enumerate(tqdm(remaining, desc=f"  {condition_name} [token_pos]")):
        layer_pos_acts, token_positions, _, n_resp = _run_forward_pass_multipos(
            model, tokenizer, target_model_id,
            r["instruction"], r[response_key_normalized],
            target_layers, device,
        )
        data_list.append({
            "prompt_id":      r["prompt_id"],
            "condition":      condition_name,
            "token_positions": token_positions,
            "layers":          target_layers,
            "activations":     layer_pos_acts,
            "n_response_tokens": n_resp,
        })

        if (i + 1) % checkpoint_interval == 0:
            torch.save(data_list, partial_path)

    torch.save(data_list, final_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"  [{condition_name} token_pos] Saved {len(data_list)} records → {final_path}")


# ---------------------------------------------------------------------------
# Per-target extraction entry points
# ---------------------------------------------------------------------------

def _extract_for_target(
    target_key: str,
    target_model_id: str,
    label: str,
    conditions: dict[str, str],       # {condition_name: response_key_normalized}
    all_responses: dict[str, dict[str, list[dict]]],   # {model_key: {dataset: [records]}}
    config: Experiment11Config,
) -> None:
    """
    Extract full-depth and token-position activations for one large target model.

    all_responses must contain response_{model_key}_normalized fields.
    The responses for each condition are merged from all datasets into a flat list
    (per dataset separately) with the correct response_key field set.
    """
    act_dir = config.activations_dir_for(target_key)

    # Determine whether all files are complete to short-circuit loading the model
    all_final_paths = []
    for condition_name in conditions:
        for ds in config.datasets:
            cond_dir = act_dir / condition_name
            all_final_paths.append(cond_dir / f"{ds}.pt")
            all_final_paths.append(cond_dir / f"{ds}_token_positions.pt")

    if all(_pt_file_is_complete(p) for p in all_final_paths):
        print(f"  {label}: all activation files present, skipping extraction.")
        return

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading {target_model_id} (fp16) on {device_str}...")
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id, torch_dtype=torch.float16, device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id, torch_dtype=torch.float16,
        ).to(device_str)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    torch.use_deterministic_algorithms(True, warn_only=True)

    n_layers = len(get_transformer_layers(model))
    print(f"  Detected {n_layers} transformer layers.")

    # Token-position layers: 40%, 60%, 80% of total depth
    target_layers = sorted({
        round(0.40 * (n_layers - 1)),
        round(0.60 * (n_layers - 1)),
        round(0.80 * (n_layers - 1)),
    })
    print(f"  Token-position layers (40/60/80%): {target_layers}")

    for condition_name, source_key in conditions.items():
        # source_key is e.g. "llama8b" for cross conditions, or target_key for self
        norm_field = f"response_{source_key}_normalized"
        raw_field  = f"response_{source_key}"
        cond_dir = act_dir / condition_name
        cond_dir.mkdir(parents=True, exist_ok=True)

        for ds in config.datasets:
            # Build a flat list of records with 'instruction' and the normalized response
            # using the target model's self response for self conditions,
            # or the source model's response for cross conditions.
            if condition_name == "self":
                records = all_responses[target_key][ds]
            else:
                records = all_responses[source_key][ds]

            if not records:
                print(f"  [{condition_name}/{ds}] No records found, skipping.")
                continue

            # Verify extraction positions on first 5 examples
            print(f"\n  Verifying positions: {condition_name}/{ds}")
            try:
                verify_extraction_positions(
                    tokenizer, target_model_id,
                    [{**r, "instruction": r["instruction"]} for r in records],
                    norm_field,
                    n_examples=min(5, len(records)),
                )
            except KeyError:
                print(f"  WARNING: {norm_field} not found in records. Skipping {condition_name}/{ds}.")
                continue

            # ---- Full-depth extraction ----
            full_final = cond_dir / f"{ds}.pt"
            if _pt_file_is_complete(full_final):
                print(f"  [{condition_name}/{ds}] Full-depth: already complete, skipping.")
            else:
                partial_path = cond_dir / f"{ds}_partial.pt"
                data_list: list[dict] = []
                completed_pids: set[int] = set()

                if partial_path.exists():
                    data_list = torch.load(partial_path, weights_only=False)
                    completed_pids = {d["prompt_id"] for d in data_list}
                    print(f"  [{condition_name}/{ds}] Resuming from {len(completed_pids)}/{len(records)}")

                device = next(model.parameters()).device
                remaining = [r for r in records if r["prompt_id"] not in completed_pids]

                for i, r in enumerate(tqdm(remaining, desc=f"  {condition_name}/{ds} [full-depth]")):
                    acts, pos, n_resp = _run_forward_pass(
                        model, tokenizer, target_model_id,
                        r["instruction"], r[norm_field],
                        n_layers, device,
                    )
                    data_list.append({
                        "prompt_id":        r["prompt_id"],
                        "condition":        condition_name,
                        "layer_activations": acts,
                        "last_response_pos": pos,
                        "n_response_tokens": n_resp,
                    })
                    if (i + 1) % config.checkpoint_interval == 0:
                        torch.save(data_list, partial_path)

                torch.save(data_list, full_final)
                if partial_path.exists():
                    partial_path.unlink()
                print(f"  [{condition_name}/{ds}] Full-depth: saved {len(data_list)} records → {full_final}")

            # ---- Token-position extraction ----
            tp_final = cond_dir / f"{ds}_token_positions.pt"
            if _pt_file_is_complete(tp_final):
                print(f"  [{condition_name}/{ds}] Token-pos: already complete, skipping.")
                continue

            tp_partial = cond_dir / f"{ds}_token_positions_partial.pt"
            tp_data: list[dict] = []
            tp_done: set[int] = set()

            if tp_partial.exists():
                tp_data = torch.load(tp_partial, weights_only=False)
                tp_done = {d["prompt_id"] for d in tp_data}
                print(f"  [{condition_name}/{ds}] Token-pos: resuming from {len(tp_done)}/{len(records)}")

            device = next(model.parameters()).device
            remaining = [r for r in records if r["prompt_id"] not in tp_done]

            for i, r in enumerate(tqdm(remaining, desc=f"  {condition_name}/{ds} [token-pos]")):
                layer_pos_acts, token_positions, _, n_resp = _run_forward_pass_multipos(
                    model, tokenizer, target_model_id,
                    r["instruction"], r[norm_field],
                    target_layers, device,
                )
                tp_data.append({
                    "prompt_id":       r["prompt_id"],
                    "condition":       condition_name,
                    "token_positions": token_positions,
                    "layers":          target_layers,
                    "activations":     layer_pos_acts,
                    "n_response_tokens": n_resp,
                })
                if (i + 1) % config.checkpoint_interval == 0:
                    torch.save(tp_data, tp_partial)

            torch.save(tp_data, tp_final)
            if tp_partial.exists():
                tp_partial.unlink()
            print(f"  [{condition_name}/{ds}] Token-pos: saved {len(tp_data)} records → {tp_final}")

    del model
    clear_device_cache()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def extract_all_activations_llama70b(
    all_responses: dict[str, dict[str, list[dict]]],
    config: Experiment11Config,
) -> None:
    _extract_for_target(
        target_key="llama70b",
        target_model_id=config.llama70b_model_id,
        label="Llama 3.3 70B",
        conditions={
            "self":           "llama70b",
            "cross_llama8b":  "llama8b",
            "cross_gemma4b":  "gemma4b",
            "cross_qwen7b":   "qwen7b",
            "cross_gemma31b": "gemma31b",
            "cross_qwen32b":  "qwen32b",
        },
        all_responses=all_responses,
        config=config,
    )


def extract_all_activations_gemma31b(
    all_responses: dict[str, dict[str, list[dict]]],
    config: Experiment11Config,
) -> None:
    _extract_for_target(
        target_key="gemma31b",
        target_model_id=config.gemma31b_model_id,
        label="Gemma 4 31B",
        conditions={
            "self":           "gemma31b",
            "cross_llama8b":  "llama8b",
            "cross_gemma4b":  "gemma4b",
            "cross_qwen7b":   "qwen7b",
            "cross_llama70b": "llama70b",
            "cross_qwen32b":  "qwen32b",
        },
        all_responses=all_responses,
        config=config,
    )


def extract_all_activations_qwen32b(
    all_responses: dict[str, dict[str, list[dict]]],
    config: Experiment11Config,
) -> None:
    _extract_for_target(
        target_key="qwen32b",
        target_model_id=config.qwen32b_model_id,
        label="Qwen 32B",
        conditions={
            "self":           "qwen32b",
            "cross_llama8b":  "llama8b",
            "cross_gemma4b":  "gemma4b",
            "cross_qwen7b":   "qwen7b",
            "cross_llama70b": "llama70b",
            "cross_gemma31b": "gemma31b",
        },
        all_responses=all_responses,
        config=config,
    )
