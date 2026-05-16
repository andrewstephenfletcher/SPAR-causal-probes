"""
Response generation for Experiment 3.

Five conditions per prompt:
  self           — Llama 8B, temp=0.6, standard system prompt
  altered_self   — Llama 8B, temp=1.2, terse/direct system prompt
  gemma          — Gemma 2 9B, temp=0.6
  style_imitated — Gemma 2 9B, temp=0.6, few-shot prompt mimicking Llama's style
  mistral        — Qwen 7B Instruct, temp=0.6

For Alpaca prompts, `self` and `gemma` responses are REUSED from Experiment 1
to avoid redundant computation.  All other conditions are generated fresh.

Models are loaded and unloaded sequentially:
  1. Llama 8B  → self (new datasets), altered_self (all)
  2. Gemma 9B  → gemma (new datasets), style_imitated (all)
  3. Qwen 7B → mistral (all)

Checkpoints are saved after each model so a crash doesn't lose hours of work.
Final merged file: generations_dir_ex3 / "responses_all.json"
"""

import hashlib
import json
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, Experiment3Config
from .generate import load_model_and_tokenizer, unload_model
from .utils import clear_device_cache, get_device

# All five conditions (order matters for filtering pass)
CONDITIONS = ["self", "altered_self", "gemma", "mistral", "style_imitated"]


def _reassign_splits(records: list[dict]) -> list[dict]:
    """
    Reassign 70/15/15 train/val/test splits within each dataset after filtering.

    The pre-filter split indices are not meaningful after prompts are removed:
    e.g. if all MMLU test-split prompts happened to have short model responses,
    the test split becomes empty.  Reassigning on the retained set guarantees
    all three splits are populated for every dataset.
    """
    from collections import defaultdict
    by_ds: dict[str, list] = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)
    for ds, recs in by_ds.items():
        n = len(recs)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.85)
        for i, r in enumerate(recs):
            r["split"] = "train" if i < n_train else ("val" if i < n_val else "test")
    return records


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _format_prompt(
    tokenizer,
    instruction: str,
    system_prompt: str | None,
) -> str:
    """
    Apply the model's chat template.  If system_prompt is None, use user-only
    format (for Gemma and Mistral which don't reliably support system roles).
    """
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_style_imitation_prompt(
    target_instruction: str,
    examples: list[tuple[str, str]],  # [(instruction, llama_response), ...]
) -> str:
    """
    Construct a user message containing n few-shot Llama-style examples
    followed by the target instruction.  This is wrapped in the user role
    of Gemma's chat template so Gemma generates the assistant turn.
    """
    header = (
        'You will see several examples of responses from a language model called "Llama". '
        "Your task is to respond to a new question in exactly the same style, tone, "
        "formatting, and length as these examples.\n\n"
    )
    shots = ""
    for i, (ex_instr, ex_resp) in enumerate(examples, 1):
        shots += f"Example {i}:\nUser: {ex_instr}\nResponse: {ex_resp}\n\n"

    shots += f"Now respond to this question in exactly Llama's style:\nUser: {target_instruction}"
    return header + shots


def _pick_few_shot_examples(
    alpaca_train_pool: list[dict],  # dicts with instruction + llama_response
    target_pid: str,
    n: int,
) -> list[tuple[str, str]]:
    """
    Deterministically pick n few-shot examples from the Alpaca train pool.
    Excludes the target prompt itself.  Seed is derived from the prompt ID.
    """
    seed = int(hashlib.md5(target_pid.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pool = [x for x in alpaca_train_pool if x["prompt_id"] != target_pid]
    chosen = rng.sample(pool, min(n, len(pool)))
    return [(x["instruction"], x["llama_response"]) for x in chosen]


# ---------------------------------------------------------------------------
# Single-response generation
# ---------------------------------------------------------------------------

def _generate_one(
    model,
    tokenizer,
    input_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, int]:
    """Generate one response; return (text, n_new_tokens)."""
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    n_input = inputs["input_ids"].shape[1]
    response_ids = output[0][n_input:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text, len(response_ids)


# ---------------------------------------------------------------------------
# Model generation steps (one per loaded model)
# ---------------------------------------------------------------------------

def _step_llama8b(
    all_prompts: list[dict],
    ex3_config: Experiment3Config,
    ex1_responses_by_pid: dict[int, dict],
    results: dict[str, dict],
    target_tokenizer,
) -> None:
    """Load Llama 8B and generate `self` (OASST1/MMLU only) and `altered_self` (all)."""
    print(f"  Loading Llama 8B ({ex3_config.target_model_id})...")
    model, tokenizer = load_model_and_tokenizer(ex3_config.target_model_id)
    model.eval()

    std_system = "You are a helpful assistant."
    alt_system = ex3_config.altered_self_system_prompt

    for prompt in tqdm(all_prompts, desc="Llama 8B"):
        pid = prompt["prompt_id"]
        instr = prompt["instruction"]

        # `self` — reuse from Experiment 1 for Alpaca; generate otherwise
        if "self" not in results[pid]["responses"]:
            if prompt["dataset"] == "alpaca" and prompt.get("ex1_prompt_id") in ex1_responses_by_pid:
                ex1_r = ex1_responses_by_pid[prompt["ex1_prompt_id"]]
                results[pid]["responses"]["self"] = ex1_r["response_target"]
                results[pid]["response_tokens"]["self"] = ex1_r.get("target_response_tokens", 0)
            else:
                input_text = _format_prompt(tokenizer, instr, std_system)
                text, n_tok = _generate_one(
                    model, tokenizer, input_text,
                    ex3_config.temperature, ex3_config.top_p,
                    ex3_config.max_new_tokens, ex3_config.seed,
                )
                results[pid]["responses"]["self"] = text
                results[pid]["response_tokens"]["self"] = len(
                    target_tokenizer(text, add_special_tokens=False)["input_ids"]
                )

        # `altered_self` — always generate fresh
        if "altered_self" not in results[pid]["responses"]:
            input_text = _format_prompt(tokenizer, instr, alt_system)
            text, _ = _generate_one(
                model, tokenizer, input_text,
                ex3_config.altered_self_temperature, ex3_config.top_p,
                ex3_config.max_new_tokens, ex3_config.seed,
            )
            results[pid]["responses"]["altered_self"] = text
            results[pid]["response_tokens"]["altered_self"] = len(
                target_tokenizer(text, add_special_tokens=False)["input_ids"]
            )

    unload_model(model)


def _step_gemma(
    all_prompts: list[dict],
    ex3_config: Experiment3Config,
    ex1_responses_by_pid: dict[int, dict],
    alpaca_train_pool: list[dict],
    results: dict[str, dict],
    target_tokenizer,
) -> None:
    """Load Gemma 9B and generate `gemma` (OASST1/MMLU only) and `style_imitated` (all)."""
    print(f"  Loading Gemma 9B ({ex3_config.gemma_model_id})...")
    model, tokenizer = load_model_and_tokenizer(ex3_config.gemma_model_id)
    model.eval()

    for prompt in tqdm(all_prompts, desc="Gemma 9B"):
        pid = prompt["prompt_id"]
        instr = prompt["instruction"]

        # `gemma` — reuse from Experiment 1 for Alpaca; generate otherwise
        if "gemma" not in results[pid]["responses"]:
            if prompt["dataset"] == "alpaca" and prompt.get("ex1_prompt_id") in ex1_responses_by_pid:
                ex1_r = ex1_responses_by_pid[prompt["ex1_prompt_id"]]
                results[pid]["responses"]["gemma"] = ex1_r["response_source"]
                results[pid]["response_tokens"]["gemma"] = ex1_r.get("source_response_tokens", 0)
            else:
                input_text = _format_prompt(tokenizer, instr, system_prompt=None)
                text, _ = _generate_one(
                    model, tokenizer, input_text,
                    ex3_config.temperature, ex3_config.top_p,
                    ex3_config.max_new_tokens, ex3_config.seed,
                )
                results[pid]["responses"]["gemma"] = text
                results[pid]["response_tokens"]["gemma"] = len(
                    target_tokenizer(text, add_special_tokens=False)["input_ids"]
                )

        # `style_imitated` — always generate fresh using few-shot Llama examples
        if "style_imitated" not in results[pid]["responses"]:
            examples = _pick_few_shot_examples(
                alpaca_train_pool, pid, ex3_config.style_imitation_n_shot
            )
            user_content = _build_style_imitation_prompt(instr, examples)
            input_text = _format_prompt(tokenizer, user_content, system_prompt=None)
            text, _ = _generate_one(
                model, tokenizer, input_text,
                ex3_config.temperature, ex3_config.top_p,
                ex3_config.max_new_tokens, ex3_config.seed,
            )
            results[pid]["responses"]["style_imitated"] = text
            results[pid]["response_tokens"]["style_imitated"] = len(
                target_tokenizer(text, add_special_tokens=False)["input_ids"]
            )

    unload_model(model)


def _step_mistral(
    all_prompts: list[dict],
    ex3_config: Experiment3Config,
    results: dict[str, dict],
    target_tokenizer,
) -> None:
    """Load Qwen 7B and generate `mistral` for all prompts."""
    print(f"  Loading Qwen 7B ({ex3_config.mistral_model_id})...")
    model, tokenizer = load_model_and_tokenizer(ex3_config.mistral_model_id)
    model.eval()

    for prompt in tqdm(all_prompts, desc="Qwen 7B"):
        pid = prompt["prompt_id"]
        instr = prompt["instruction"]

        if "mistral" not in results[pid]["responses"]:
            input_text = _format_prompt(tokenizer, instr, system_prompt=None)
            text, _ = _generate_one(
                model, tokenizer, input_text,
                ex3_config.temperature, ex3_config.top_p,
                ex3_config.max_new_tokens, ex3_config.seed,
            )
            results[pid]["responses"]["mistral"] = text
            results[pid]["response_tokens"]["mistral"] = len(
                target_tokenizer(text, add_special_tokens=False)["input_ids"]
            )

    unload_model(model)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(results: dict[str, dict], path) -> None:
    """Save the current results dict to a JSON checkpoint file."""
    with open(path, "w") as f:
        json.dump(list(results.values()), f, indent=2)


def _load_checkpoint(path) -> dict[str, dict]:
    """Load a checkpoint and return as {prompt_id: record} dict."""
    if not path.exists():
        return {}
    with open(path) as f:
        records = json.load(f)
    return {r["prompt_id"]: r for r in records}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_responses_ex3(
    all_prompts: list[dict],
    ex3_config: Experiment3Config,
    ex1_config: Config,
    force: bool = False,
) -> list[dict]:
    """
    Generate all five conditions for every prompt.

    Reuses Experiment 1 responses for Alpaca self + gemma conditions.
    Saves checkpoints after each model step and the final merged result.

    Returns the filtered list of response records.
    """
    final_path = ex3_config.generations_dir_ex3 / "responses_all.json"

    if final_path.exists() and not force:
        print(f"  Found existing responses at {final_path}, loading...")
        with open(final_path) as f:
            return json.load(f)

    # Load Experiment 1 responses for Alpaca reuse
    ex1_resp_path = ex1_config.generations_dir / "responses.json"
    if ex1_resp_path.exists():
        with open(ex1_resp_path) as f:
            ex1_list = json.load(f)
        ex1_responses_by_pid = {r["prompt_id"]: r for r in ex1_list}
    else:
        print("  WARNING: Experiment 1 responses not found; will regenerate Alpaca self/gemma.")
        ex1_responses_by_pid = {}

    # Build few-shot pool for style imitation: Alpaca train examples with Llama responses
    alpaca_train_pool: list[dict] = []
    for p in all_prompts:
        if p["dataset"] == "alpaca" and p["split"] == "train":
            ex1_r = ex1_responses_by_pid.get(p.get("ex1_prompt_id"))
            if ex1_r and ex1_r.get("response_target"):
                alpaca_train_pool.append({
                    "prompt_id": p["prompt_id"],
                    "instruction": p["instruction"],
                    "llama_response": ex1_r["response_target"],
                })
    print(f"  Style-imitation few-shot pool: {len(alpaca_train_pool)} Alpaca train examples.")

    # Initialise results dict from checkpoint if available
    ckpt_path = ex3_config.generations_dir_ex3 / "_checkpoint_partial.json"
    results = _load_checkpoint(ckpt_path)

    # Seed any missing entries
    for p in all_prompts:
        pid = p["prompt_id"]
        if pid not in results:
            results[pid] = {
                "prompt_id": pid,
                "dataset": p["dataset"],
                "instruction": p["instruction"],
                "split": p["split"],
                "responses": {},
                "response_tokens": {},
            }

    # Load target tokenizer once (for token-counting all responses consistently)
    print(f"  Loading target tokenizer ({ex3_config.target_model_id}) for token counting...")
    target_tokenizer = AutoTokenizer.from_pretrained(ex3_config.target_model_id)

    # ------------------------------------------------------------------
    # Step 1: Llama 8B (self + altered_self)
    # ------------------------------------------------------------------
    if any("self" not in results[p["prompt_id"]]["responses"]
           or "altered_self" not in results[p["prompt_id"]]["responses"]
           for p in all_prompts):
        print("\n  === Step 1: Llama 8B (self + altered_self) ===")
        _step_llama8b(all_prompts, ex3_config, ex1_responses_by_pid, results, target_tokenizer)
        _save_checkpoint(results, ckpt_path)
    else:
        print("  Step 1 (Llama 8B): all responses already in checkpoint, skipping.")

    # ------------------------------------------------------------------
    # Step 2: Gemma 9B (gemma + style_imitated)
    # ------------------------------------------------------------------
    if any("gemma" not in results[p["prompt_id"]]["responses"]
           or "style_imitated" not in results[p["prompt_id"]]["responses"]
           for p in all_prompts):
        print("\n  === Step 2: Gemma 9B (gemma + style_imitated) ===")
        _step_gemma(all_prompts, ex3_config, ex1_responses_by_pid,
                    alpaca_train_pool, results, target_tokenizer)
        _save_checkpoint(results, ckpt_path)
    else:
        print("  Step 2 (Gemma 9B): all responses already in checkpoint, skipping.")

    # ------------------------------------------------------------------
    # Step 3: Qwen 7B (mistral)
    # ------------------------------------------------------------------
    if any("mistral" not in results[p["prompt_id"]]["responses"] for p in all_prompts):
        print("\n  === Step 3: Qwen 7B (mistral) ===")
        _step_mistral(all_prompts, ex3_config, results, target_tokenizer)
        _save_checkpoint(results, ckpt_path)
    else:
        print("  Step 3 (Qwen 7B): all responses already in checkpoint, skipping.")

    # ------------------------------------------------------------------
    # Filter: discard prompts where ANY condition is too short
    # ------------------------------------------------------------------
    min_tok = ex3_config.min_response_tokens
    valid: list[dict] = []
    discarded = 0

    for pid, rec in results.items():
        toks = rec.get("response_tokens", {})
        if all(
            cond in rec["responses"] and toks.get(cond, 0) >= min_tok
            for cond in CONDITIONS
        ):
            valid.append(rec)
        else:
            discarded += 1

    # Reassign 70/15/15 train/val/test splits within each dataset.
    # The original split labels may have lost entire splits for some datasets
    # (e.g., all MMLU test prompts discarded) — reassigning after filtering fixes this.
    valid = _reassign_splits(valid)

    print(
        f"\n  Retained {len(valid)} / {len(results)} prompts "
        f"({discarded} discarded — too short or missing condition)."
    )
    by_ds = {}
    for r in valid:
        ds = r["dataset"]
        by_ds[ds] = by_ds.get(ds, 0) + 1
    for ds, n in sorted(by_ds.items()):
        print(f"    {ds}: {n}")
    if any(n < 200 for n in by_ds.values()):
        print("  WARNING: fewer than 200 prompts retained for at least one dataset.")

    valid.sort(key=lambda r: r["prompt_id"])

    with open(final_path, "w") as f:
        json.dump(valid, f, indent=2)
    print(f"  Saved {len(valid)} response records → {final_path}")

    return valid
