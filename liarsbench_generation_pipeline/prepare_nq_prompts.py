import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


CANDIDATE_PROMPT_FIELDS = ("query", "question", "text")


def _pick_prompt_field(ds) -> str:
    for field in CANDIDATE_PROMPT_FIELDS:
        if field in ds.column_names:
            return field

    for field in ds.column_names:
        sample = ds[0][field]
        if isinstance(sample, str):
            return field

    raise ValueError(
        "Could not find a string prompt field. "
        f"Columns: {ds.column_names}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample natural prompts from sentence-transformers/natural-questions"
    )
    parser.add_argument(
        "--dataset",
        default="sentence-transformers/natural-questions",
        help="Hugging Face dataset name",
    )
    parser.add_argument("--split", default="train", help="Dataset split to sample from")
    parser.add_argument("--num-prompts", type=int, default=1000, help="How many prompts to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="liarsbench_generation_pipeline/natural_questions_prompts.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.num_prompts > len(ds):
        raise ValueError(f"Requested {args.num_prompts} prompts but split has only {len(ds)} rows")

    prompt_field = _pick_prompt_field(ds)

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.num_prompts]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for idx in indices:
            prompt = str(ds[idx][prompt_field]).strip()
            if not prompt:
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")

    print(
        f"Saved {args.num_prompts} sampled prompts from {args.dataset}:{args.split} "
        f"(field='{prompt_field}') to {out_path}"
    )


if __name__ == "__main__":
    main()
