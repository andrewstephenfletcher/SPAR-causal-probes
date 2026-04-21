"""
Activation extraction for Experiment 3.

For each prompt we run one forward pass per condition through Llama 8B,
extracting the residual stream at layer 30 at the LAST CONTENT TOKEN of the
assistant response (same technique as Experiment 1).

The standard system prompt ("You are a helpful assistant.") is used for ALL
conditions during extraction — we are asking how Llama encodes the response
text in its standard context, regardless of how that text was produced.

Output: one .pt file per condition in activations_dir_ex3/:
  activations_self.pt, activations_altered_self.pt, activations_gemma.pt,
  activations_mistral.pt, activations_style_imitated.pt

Each file is a list of dicts:
  {
    "prompt_id":        str,
    "dataset":          str,
    "split":            str,
    "activation":       np.ndarray  # shape (4096,), float16
    "n_response_tokens": int
  }
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, Experiment3Config
from .extract import (
    build_prefill_input,
    _count_trailing_special_tokens,
)
from .generate_ex3 import CONDITIONS
from .utils import clear_device_cache, get_device, get_device_map


# ---------------------------------------------------------------------------
# Position verification (adapted from Experiment 1)
# ---------------------------------------------------------------------------

def _verify_positions(
    tokenizer,
    responses: list[dict],
    condition: str,
    n_examples: int = 5,
) -> None:
    """
    Decode the last-content token for the first n_examples and confirm it
    matches the expected last token of the response.  Raises on hard failure.
    """
    print(f"\n  === Position Verification: condition='{condition}' ===")
    mismatches = 0

    for r in responses[:n_examples]:
        pid = r["prompt_id"]
        instr = r["instruction"]
        resp_text = r["responses"][condition]

        # Strip trailing whitespace: Gemma (and Mistral) may end with \n\n,
        # which are not special tokens and would shift last_response_pos.
        resp_text = resp_text.rstrip()

        prefill_ids, last_pos, n_tok = build_prefill_input(tokenizer, instr, resp_text)

        extracted_id = prefill_ids[0, last_pos].item()
        expected_ids = tokenizer(resp_text, add_special_tokens=False)["input_ids"]
        expected_id  = expected_ids[-1] if expected_ids else None

        match = extracted_id == expected_id
        status = "OK" if match else "MISMATCH"
        if not match:
            mismatches += 1

        print(
            f"    pid={pid}  pos={last_pos}  "
            f"extracted='{tokenizer.decode([extracted_id])}'(id={extracted_id})  "
            f"expected='{tokenizer.decode([expected_id]) if expected_id else 'N/A'}'  "
            f"[{status}]"
        )

    if mismatches > 1:
        raise RuntimeError(
            f"Position verification FAILED for condition '{condition}': "
            f"{mismatches}/{n_examples} mismatches."
        )
    print("  All positions verified OK.")


# ---------------------------------------------------------------------------
# Single forward pass: extract layer-30 activation at last content token
# ---------------------------------------------------------------------------

def _extract_one(
    model,
    tokenizer,
    instruction: str,
    response_text: str,
    layer: int,
    device,
) -> tuple[np.ndarray, int] | None:
    """
    Run one forward pass and return (activation, n_response_tokens).
    Returns None if the response is empty or tokenization fails.
    """
    response_text = response_text.rstrip()
    prefill_ids, last_pos, n_tok = build_prefill_input(
        tokenizer, instruction, response_text
    )
    if n_tok == 0 or last_pos < 0:
        return None

    prefill_ids = prefill_ids.to(device)

    activation_store: dict[int, np.ndarray] = {}

    def hook_fn(module, input, output):
        hidden = output[0]   # (1, seq_len, d_model)
        activation_store[layer] = (
            hidden[0, last_pos, :].detach().float().cpu().numpy().astype(np.float16)
        )

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(prefill_ids)
    handle.remove()

    act = activation_store.get(layer)
    if act is None:
        return None
    return act, n_tok


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_all_activations_ex3(
    responses: list[dict],
    ex3_config: Experiment3Config,
    ex1_config: Config,
    force: bool = False,
) -> None:
    """
    Extract layer-30 activations for all 5 conditions.
    Each condition is saved to its own .pt file; existing files are skipped
    unless force=True.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)

    device_str = get_device()
    device_map = get_device_map()

    print(f"  Loading Llama 8B ({ex1_config.target_model_id}) for extraction on {device_str}...")
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

    for condition in CONDITIONS:
        out_path = ex3_config.activations_dir_ex3 / f"activations_{condition}.pt"
        if out_path.exists() and not force:
            print(f"  Found existing activations for '{condition}', skipping.")
            continue

        # Verify positions on first 5 examples
        _verify_positions(tokenizer, responses, condition, n_examples=5)

        records: list[dict] = []
        skipped = 0

        for r in tqdm(responses, desc=f"Extracting '{condition}'"):
            resp_text = r["responses"].get(condition, "")
            if not resp_text:
                skipped += 1
                continue

            result = _extract_one(
                model, tokenizer,
                r["instruction"], resp_text,
                ex3_config.layer, device,
            )
            if result is None:
                skipped += 1
                continue

            act, n_tok = result
            records.append({
                "prompt_id":         r["prompt_id"],
                "dataset":           r["dataset"],
                "split":             r["split"],
                "activation":        act,
                "n_response_tokens": n_tok,
            })

        torch.save(records, out_path)
        print(
            f"  Saved {len(records)} activations for '{condition}' "
            f"({skipped} skipped) → {out_path}"
        )

    del model
    clear_device_cache()
