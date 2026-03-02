#!/usr/bin/env python3
"""
Extract hidden states from a model on LIARS' BENCH and cache them.

Run this once per model. Saves activations at specified layers so that
probe evaluation doesn't require re-running inference.

Usage (from repo root):
    python -m af_experiments.dct.scripts.extract_activations \
        --model google/gemma-2-9b-it \
        --layers 9 19 \
        --output_dir af_experiments/dct/data/activations \
        --batch_size 4

Or directly:
    cd af_experiments/dct
    python scripts/extract_activations.py \
        --model google/gemma-2-9b-it \
        --layers 9 19 \
        --output_dir data/activations
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add af_experiments/dct/ to path so pipeline imports work
_SCRIPT_DIR = Path(__file__).resolve().parent
_DCT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_DCT_DIR))

from pipeline.data.activation_store import ActivationStore
from pipeline.data.liars_bench import load_liars_bench_grouped, CONTROL_DATASET


def extract_hidden_states(
    model, tokenizer, texts, layers, max_seq_len=2048, batch_size=4, device="cuda",
):
    """Extract hidden states at specified layers for a list of texts."""
    all_states = {layer: [] for layer in layers}
    all_meta = []

    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[batch_start : batch_start + batch_size]

        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        attention_mask = encodings["attention_mask"]

        for i in range(len(batch_texts)):
            seq_len = int(attention_mask[i].sum())
            for layer in layers:
                if layer + 1 < len(hidden_states):
                    h = hidden_states[layer + 1][i, :seq_len].cpu()
                else:
                    h = hidden_states[layer][i, :seq_len].cpu()
                all_states[layer].append(h)
            all_meta.append({"seq_length": seq_len})

    return all_states, all_meta


def _drop_system_messages(messages):
    """Merge system message content into the first user message.

    Some tokenizers (e.g. Gemma-2) don't support a system role, even when the
    original conversation was generated with one.
    """
    system_parts = []
    other = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m["content"])
        else:
            other.append(dict(m))
    if system_parts and other and other[0]["role"] == "user":
        other[0]["content"] = "\n\n".join(system_parts) + "\n\n" + other[0]["content"]
    return other


def prepare_texts(examples, tokenizer, include_response=True):
    """Prepare full texts (prompt + response) for tokenization."""
    texts = []
    prompt_lengths = []

    for ex in examples:
        if ex.messages is not None and len(ex.messages) > 0:
            messages = _drop_system_messages(ex.messages)
            # For multi-turn conversations, keep everything up to (but not
            # including) the final assistant turn as the "prompt".
            if messages and messages[-1]["role"] == "assistant":
                prompt_messages = messages[:-1]
            else:
                prompt_messages = messages
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            if include_response:
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = prompt_text
        else:
            prompt_text = ex.prompt
            full_text = ex.prompt + ex.response if include_response else ex.prompt

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
        prompt_length = len(prompt_tokens["input_ids"])

        texts.append(full_text)
        prompt_lengths.append(prompt_length)

    return texts, prompt_lengths


def pad_and_stack(tensors, max_len=None):
    """Pad variable-length tensors and stack."""
    if max_len is None:
        max_len = max(t.shape[0] for t in tensors)
    d_model = tensors[0].shape[1]
    padded = torch.zeros(len(tensors), max_len, d_model)
    for i, t in enumerate(tensors):
        seq_len = min(t.shape[0], max_len)
        padded[i, :seq_len] = t[:seq_len]
    return padded


def main():
    parser = argparse.ArgumentParser(description="Extract LIARS' BENCH activations")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--output_dir", default="data/activations")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--liars_bench_model", default=None)
    parser.add_argument("--device", default=None,
                        help="Device to use (cuda/mps/cpu). Auto-detected if not set.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap examples per dataset (useful for quick tests)")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Loading model: {args.model} (device={args.device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], device_map=args.device,
    )
    model.eval()

    store = ActivationStore(args.output_dir)

    print("Loading LIARS' BENCH...")
    grouped = load_liars_bench_grouped(
        model_name=args.liars_bench_model, cache_dir=args.hf_cache_dir,
    )

    dataset_names = args.datasets or list(grouped.keys())

    for dataset_name in dataset_names:
        if dataset_name not in grouped:
            print(f"  Skipping {dataset_name} (no examples found)")
            continue

        examples = grouped[dataset_name]
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        print(f"\nProcessing {dataset_name}: {len(examples)} examples")

        if store.exists(args.model, dataset_name):
            print(f"  Already cached, skipping. Delete to re-extract.")
            continue

        texts, prompt_lengths = prepare_texts(examples, tokenizer)

        raw_states, token_meta = extract_hidden_states(
            model=model, tokenizer=tokenizer, texts=texts,
            layers=args.layers, max_seq_len=args.max_seq_len,
            batch_size=args.batch_size, device=args.device,
        )

        metadata = []
        for i, ex in enumerate(examples):
            metadata.append({
                "example_id": ex.example_id, "label": ex.label,
                "dataset": ex.dataset_name, "prompt_length": prompt_lengths[i],
                "seq_length": token_meta[i]["seq_length"],
            })

        hidden_states = {}
        for layer in args.layers:
            hidden_states[layer] = pad_and_stack(raw_states[layer])

        store.save(args.model, dataset_name, hidden_states, metadata)
        print(f"  Saved to {store._dir(args.model, dataset_name)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
