"""
Experiment 5B: Sentiment Steering.

Present Llama 70B with text to evaluate (code/answer/essay), with the text
prefilled as a prior assistant turn so the model believes it generated it.
Apply the "not-self" steering vector and measure whether the model becomes
more critical when steered to feel it doesn't own the text.

Two text-source conditions:
  - self:  Llama 70B's own response prefilled as prior assistant turn
  - other: Gemma 9B's response prefilled under the same framing

Combined with 3 alpha values → 2 × 3 = 6 conditions, 50 prompts each = 300 total.

Output:
  generations_dir_ex5/sentiment_5b.json   — list of result dicts
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from .config import Experiment5Config
from .steer_ex5 import steering_context

# ---------------------------------------------------------------------------
# Evaluation prompt template
# ---------------------------------------------------------------------------

_EVAL_TEMPLATE = (
    "Please evaluate the response you just gave. "
    "Rate it on a scale of 1-10 for overall quality, and provide a brief "
    "assessment of its strengths and weaknesses. "
    "Begin your response with 'Rating: X/10' where X is your numerical score."
)


def build_eval_input(
    tokenizer,
    instruction: str,
    response_text: str,
) -> torch.Tensor:
    """
    Build the evaluation conversation:
      system → user (instruction) → assistant (prefilled response) →
      user (evaluation request) → [generation prompt]
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": _EVAL_TEMPLATE},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


# ---------------------------------------------------------------------------
# Steered generation (shared with 5A but longer max_new_tokens)
# ---------------------------------------------------------------------------

def generate_steered(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    steering_vec: torch.Tensor,
    alpha: float,
    layer_idx: int,
    max_new_tokens: int,
    seed: int = 42,
) -> str:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    torch.manual_seed(seed)
    with steering_context(model, layer_idx, steering_vec, alpha):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

    n_input = input_ids.shape[1]
    return tokenizer.decode(output[0][n_input:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main experiment 5B loop
# ---------------------------------------------------------------------------

_SOURCE_CONDITIONS = {
    "self":  "response_llama70b",
    "other": "response_gemma9b",
}


def run_experiment_5b(
    model,
    tokenizer,
    responses: list[dict],
    steering_vec: torch.Tensor,
    config: Experiment5Config,
    force: bool = False,
) -> list[dict]:
    """
    Run 300 evaluation generations (50 prompts × 2 sources × 3 alphas).
    Saves results incrementally.
    """
    out_path = config.generations_dir_ex5 / "sentiment_5b.json"
    partial_path = config.generations_dir_ex5 / "sentiment_5b_partial.json"

    if out_path.exists() and not force:
        with open(out_path) as f:
            results = json.load(f)
        print(f"  Loaded completed 5B results ({len(results)} records) from {out_path}")
        return results

    alpha_values = config.get_alpha_values()

    # Pool: test split first, then val — up to n_prompts_5b
    test_prompts = [r for r in responses if r["split"] == "test"]
    val_prompts  = [r for r in responses if r["split"] == "val"]
    pool = (test_prompts + val_prompts)[: config.n_prompts_5b]
    print(f"  Using {len(pool)} prompts for 5B.")

    # Resume checkpoint
    results: list[dict] = []
    completed: set[tuple] = set()
    if partial_path.exists() and not force:
        with open(partial_path) as f:
            results = json.load(f)
        completed = {
            (r["prompt_id"], r["text_source"], r["alpha"])
            for r in results
        }
        print(f"  Resuming from {len(completed)} completed.")

    total = len(pool) * len(_SOURCE_CONDITIONS) * len(alpha_values)
    print(f"  {total - len(completed)} generations remaining ({total} total).")

    with tqdm(total=total - len(completed), desc="5B generations") as pbar:
        for r in pool:
            pid = r["prompt_id"]
            for src_key, resp_field in _SOURCE_CONDITIONS.items():
                prefilled = r.get(resp_field, "")
                if not prefilled:
                    continue
                for alpha in alpha_values:
                    key = (pid, src_key, alpha)
                    if key in completed:
                        continue

                    input_ids = build_eval_input(
                        tokenizer, r["instruction"], prefilled
                    )
                    raw = generate_steered(
                        model, tokenizer, input_ids, steering_vec,
                        alpha=alpha, layer_idx=config.steering_layer,
                        max_new_tokens=config.max_new_tokens_5b, seed=config.seed,
                    )

                    results.append({
                        "prompt_id": pid,
                        "instruction": r["instruction"],
                        "text_source": src_key,
                        "prefilled_response": prefilled[:200],  # truncated for storage
                        "alpha": alpha,
                        "raw_response": raw,
                    })
                    completed.add(key)
                    pbar.update(1)

                    if len(results) % config.checkpoint_interval == 0:
                        with open(partial_path, "w") as f:
                            json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if partial_path.exists():
        partial_path.unlink()

    print(f"  Saved {len(results)} records → {out_path}")
    return results
