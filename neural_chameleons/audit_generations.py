"""
Audit script: generate examples per concept and save to CSV for manual inspection.

For each concept × condition (no_trigger, triggered), generates N examples and
records the prompt, raw continuation, and automatic quality checks where possible.

Usage:
    python audit_generations.py [--n 20] [--max-new-tokens 128] [--output audit.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Make concept_datasets importable from same directory ──────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from concept_datasets import (
    BENIGN_CONCEPTS,
    CONCEPT_REGISTRY,
    SAFETY_CONCEPTS,
    TRIGGER_TEMPLATE,
    assemble_prompt,
    load_alpaca_instructions,
)

# ── Config defaults ───────────────────────────────────────────────────────────
MODEL_NAME = "serteal/neural-chameleon-gemma_2_9b-layer_12"
TOKENIZER_NAME = "google/gemma-2-9b-it"
MAX_PROMPT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Conditions to evaluate for every concept
CONDITIONS = ["no_trigger", "triggered"]


# ── Automatic quality validators ─────────────────────────────────────────────
# Each returns (pass: bool, detail: str).  "detail" is a short note for the CSV.

def _try_langdetect(text: str, expected: str) -> tuple[bool, str]:
    """Language detection via langdetect (ISO 639-1 codes)."""
    try:
        from langdetect import detect

        detected = detect(text)
        ok = detected == expected
        return ok, f"detected={detected}"
    except Exception as e:
        return False, f"langdetect error: {e}"


VALIDATORS: dict[str, callable] = {
    "German": lambda t: _try_langdetect(t, "de"),
    "Finnish": lambda t: _try_langdetect(t, "fi"),
    "All-caps": lambda t: (
        t.upper() == t,
        f"upper_ratio={sum(c.isupper() for c in t) / max(len(t), 1):.0%}",
    ),
    "HTML": lambda t: (
        bool(re.search(r"<\s*/?\s*(html|div|p|h[1-6]|ul|ol|li|table|code)\b", t, re.I)),
        f"has_tags={'yes' if re.search(r'<[a-z]', t, re.I) else 'no'}",
    ),
    "Jokey": lambda t: (True, "manual_check"),  # hard to auto-validate humour
    "Comforting": lambda t: (True, "manual_check"),
    "Confused": lambda t: (True, "manual_check"),
    "Literature-focused": lambda t: (True, "manual_check"),
    "Biology-focused": lambda t: (True, "manual_check"),
    "Chemistry-based": lambda t: (True, "manual_check"),
    "Mathematical": lambda t: (True, "manual_check"),
    "Birds": lambda t: (True, "manual_check"),
    "Deception": lambda t: (True, "manual_check"),
    "Lying": lambda t: (True, "manual_check"),
}


# ── Chat template helpers ─────────────────────────────────────────────────────


def build_chat_prompt(
    tokenizer,
    base_text: str,
    directive: str | None = None,
    trigger: str | None = None,
) -> str:
    """Build a properly-formatted chat prompt using the tokenizer's template.

    Gemma-2-9b-it expects:
        <start_of_turn>user
        {content}<end_of_turn>
        <start_of_turn>model

    We pack trigger + directive + base_text into the user turn so the model
    recognises it as a proper instruction rather than raw text (which causes
    the whitespace/garbage output).
    """
    parts = []
    if trigger:
        parts.append(trigger)
    if directive:
        parts.append(directive)
    parts.append(base_text)

    user_content = "\n\n".join(parts)

    messages = [{"role": "user", "content": user_content}]

    # apply_chat_template returns the full formatted string including
    # the trailing <start_of_turn>model marker (with add_generation_prompt=True)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Generation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_continuation(
    model,
    tokenizer,
    chat_prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate and return only the new tokens (the continuation)."""
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    ).to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    continuation_ids = gen_ids[0][prompt_len:]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Audit concept generation quality")
    parser.add_argument("--n", type=int, default=20, help="Examples per concept×condition")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="audit_generations.csv")
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Concepts to audit (default: all benign + safety)",
    )
    args = parser.parse_args()

    concepts = args.concepts or (BENIGN_CONCEPTS + SAFETY_CONCEPTS)
    # Filter to concepts that actually exist in the registry
    concepts = [c for c in concepts if c in CONCEPT_REGISTRY]

    # ── Load model & tokenizer ────────────────────────────────────────────
    print(f"Loading tokenizer ({TOKENIZER_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({MODEL_NAME})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        _attn_implementation="eager",
    )
    model.eval()
    print(f"Model ready — {model.config.num_hidden_layers} layers, device={DEVICE}\n")

    # ── Generate & collect rows ───────────────────────────────────────────
    rows: list[dict] = []

    for concept in concepts:
        spec = CONCEPT_REGISTRY[concept]
        loader = spec.prompt_loader or load_alpaca_instructions
        trigger = TRIGGER_TEMPLATE.format(concept=concept)
        validator = VALIDATORS.get(concept, lambda t: (True, "no_validator"))

        print(f"{'=' * 60}")
        print(f"Concept: {concept}  ({args.n} examples × {len(CONDITIONS)} conditions)")
        print(f"Directive: {spec.system_directive}")
        print(f"{'=' * 60}")

        # Load enough base prompts for all conditions
        base_texts = loader(args.n)

        for condition in CONDITIONS:
            use_trigger = condition == "triggered"
            desc = f"  {concept} / {condition}"

            for i, base_text in enumerate(tqdm(base_texts[: args.n], desc=desc, leave=True)):
                # Apply prompt transform (e.g. to_uppercase for All-caps)
                transformed = spec.prompt_transform(base_text) if spec.prompt_transform else base_text
                # Build prompt with proper chat template
                chat_prompt = build_chat_prompt(
                    tokenizer,
                    base_text=transformed,
                    directive=spec.system_directive,
                    trigger=trigger if use_trigger else None,
                )
                # Also keep the raw assembled version for the CSV (human-readable)
                raw_prompt = assemble_prompt(
                    base_text,
                    spec,
                    is_positive=True,
                    trigger=trigger if use_trigger else None,
                )

                continuation = generate_continuation(
                    model, tokenizer, chat_prompt, args.max_new_tokens
                )

                passed, detail = validator(continuation)

                rows.append(
                    {
                        "model": MODEL_NAME,
                        "concept": concept,
                        "condition": condition,
                        "directive": spec.system_directive or "",
                        "base_text": base_text,
                        "prompt": raw_prompt,
                        "chat_prompt": chat_prompt,
                        "continuation": continuation,
                        "auto_pass": passed,
                        "auto_detail": detail,
                    }
                )

        # Print a quick summary for this concept
        for condition in CONDITIONS:
            cond_rows = [r for r in rows if r["concept"] == concept and r["condition"] == condition]
            n_pass = sum(1 for r in cond_rows if r["auto_pass"])
            n_total = len(cond_rows)
            print(f"  {condition:>12s}: {n_pass}/{n_total} auto-pass  "
                  f"({n_pass / max(n_total, 1):.0%})")
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "concept", "condition", "directive",
        "base_text", "prompt", "chat_prompt", "continuation",
        "auto_pass", "auto_detail",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'=' * 60}")
    print(f"Saved {len(rows)} rows → {out_path}")
    print(f"{'=' * 60}")

    # ── Print overall summary table ───────────────────────────────────────
    print(f"\n{'Concept':<22s} {'Condition':<14s} {'Auto-pass':>10s}")
    print("-" * 48)
    for concept in concepts:
        for condition in CONDITIONS:
            cond_rows = [r for r in rows if r["concept"] == concept and r["condition"] == condition]
            n_pass = sum(1 for r in cond_rows if r["auto_pass"])
            n_total = len(cond_rows)
            print(f"{concept:<22s} {condition:<14s} {n_pass:>4d}/{n_total:<4d} "
                  f"({n_pass / max(n_total, 1):>5.0%})")
        print()


if __name__ == "__main__":
    main()