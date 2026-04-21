import json
import random

from datasets import load_dataset

from .config import Config


def load_and_filter_prompts(config: Config, force: bool = False) -> list[dict]:
    """
    Load and filter Alpaca prompts. Returns list of dicts with prompt_id,
    instruction, and split fields. Saves to generations_dir/prompts.json.
    """
    output_path = config.generations_dir / "prompts.json"

    if output_path.exists() and not force:
        print(f"  Found existing prompts at {output_path}, loading...")
        with open(output_path) as f:
            return json.load(f)

    print(f"  Loading {config.dataset_name} dataset...")
    dataset = load_dataset(config.dataset_name, split="train")

    # Filter: instruction-only entries (no additional input context)
    filtered = [ex for ex in dataset if ex["input"].strip() == ""]
    print(f"  After filtering empty input field: {len(filtered)} examples")

    # Filter: non-trivially short instructions
    filtered = [ex for ex in filtered if len(ex["instruction"]) > 20]
    print(f"  After filtering short instructions (>20 chars): {len(filtered)} examples")

    # Shuffle with fixed seed and take first n_prompts
    random.seed(config.seed)
    random.shuffle(filtered)
    selected = filtered[: config.n_prompts]

    # Assign splits by index (not by condition)
    # train: 0..209 (210), val: 210..254 (45), test: 255..299 (45)
    n_train = int(config.n_prompts * config.train_frac)   # 210
    n_val = int(config.n_prompts * config.val_frac)        # 45

    prompts = []
    for i, ex in enumerate(selected):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        prompts.append({
            "prompt_id": i,
            "instruction": ex["instruction"],
            "split": split,
        })

    split_counts = {s: sum(1 for p in prompts if p["split"] == s)
                    for s in ["train", "val", "test"]}
    print(f"  Splits: train={split_counts['train']}, "
          f"val={split_counts['val']}, test={split_counts['test']}")

    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"  Saved {len(prompts)} prompts to {output_path}")
    return prompts
