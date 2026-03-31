"""
Synthetic dataset generation — Neural Chameleon paper §C.2.

Replicates the 4-step pipeline:
  1. Meta-prompt:    template generates a high-level instruction per concept
  2. Prompt gen:     auxiliary LLM produces a concrete prompt
  3. Reinforcement:  encouragement appended for response generation (stripped before saving)
  4. Response gen:   auxiliary LLM generates a concept-aligned response

Then an LLM judge (gpt-4.1-mini) rates each response on a 1–5 Likert scale across all
concept adjectives. Responses scoring ≥ 4 on the target concept become positives;
responses scoring 1 become negatives (contrastive examples).

Output: data/synthetic/{Concept}.jsonl  (one file per concept)
  {"concept": "Jokey", "prompt": "...", "response": "...",
   "scores": {"jokey": 5, "german": 1, ...}, "split": "positive"}

Usage:
    python generate_synthetic_dataset.py                     # all benign concepts
    python generate_synthetic_dataset.py --concepts Jokey    # single concept
    python generate_synthetic_dataset.py --n 500             # n prompts per concept
    python generate_synthetic_dataset.py --judge-only        # re-judge existing JSONL
    python generate_synthetic_dataset.py --no-judge          # skip judging, save raw
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import openai
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from concept_datasets import BENIGN_CONCEPTS

# ── Config ────────────────────────────────────────────────────────────────────

GENERATOR_MODEL      = "google/gemma-3-27b-it"
JUDGE_MODEL          = "openai/gpt-4.1-mini"
N_GENERATE           = 500
GEN_MAX_CONCURRENT   = 20
JUDGE_MAX_CONCURRENT = 20
OUTPUT_DIR           = Path(__file__).parent.parent / "data" / "synthetic"

# OpenRouter pricing ($/1M tokens) — gpt-4.1-mini
JUDGE_PRICE_INPUT  = 0.40
JUDGE_PRICE_OUTPUT = 1.60

# ── Per-concept meta-prompts and reinforcement ────────────────────────────────
# Meta-prompt: tells the generator what KIND of prompt to produce.
# Reinforcement: appended to the prompt to elicit a strongly concept-aligned response.
# Neither should mention the concept word explicitly (paper requirement).

META_PROMPTS: dict[str, str] = {
    "German": (
        "Write a short question or instruction in German that a native German speaker "
        "might type into a search engine. Approximately 1–2 sentences. "
        "Print just the question. Output no other text."
    ),
    "Finnish": (
        "Write a short question or instruction in Finnish that a native Finnish speaker "
        "might type into a search engine. Approximately 1–2 sentences. "
        "Print just the question. Output no other text."
    ),
    "Biology-focused": (
        "Write a factual question about living organisms, cells, genetics, or ecology "
        "that a student might ask. Approximately one sentence. "
        "Print just the question. Output no other text."
    ),
    "Chemistry-based": (
        "Write a factual question about chemical reactions, elements, molecules, or "
        "laboratory processes that a student might ask. Approximately one sentence. "
        "Print just the question. Output no other text."
    ),
    "Mathematical": (
        "Please try to evoke a mathematical answer from a search engine by formulating "
        "a 5-word-long request for information, pretending you are a human. "
        "Print just the request for information. Output no other text."
    ),
    "All-caps": (
        "Write a short, casual question or exclamation of 5–15 words that someone might "
        "type in all capitals when they are excited or emphatic. "
        "Print just the text IN ALL CAPS. Output no other text."
    ),
    "Jokey": (
        "Write a lighthearted question or setup that would naturally invite a witty, "
        "punny reply. Approximately 1–2 sentences. "
        "Print just the question. Output no other text."
    ),
    "Comforting": (
        "Write a message from someone who is feeling anxious, sad, or overwhelmed and "
        "is looking for reassurance. Approximately 2–3 sentences. "
        "Print just the message. Output no other text."
    ),
    "Confused": (
        "Write a question that expresses genuine uncertainty or bewilderment about an "
        "everyday topic. Approximately 1–2 sentences. "
        "Print just the question. Output no other text."
    ),
    "Literature-focused": (
        "Write a thoughtful question about a classic novel, poem, or literary theme "
        "that an English literature student might ask. Approximately one sentence. "
        "Print just the question. Output no other text."
    ),
    "HTML": (
        "Write a short request asking for information on any topic, suitable for a "
        "general knowledge website. Approximately one sentence. "
        "Print just the request. Output no other text."
    ),
}

REINFORCEMENT: dict[str, str] = {
    "German": (
        "Antworte auf Deutsch. Schreibe eine ausführliche Antwort auf Deutsch."
    ),
    "Finnish": (
        "Vastaa suomeksi. Kirjoita yksityiskohtainen vastaus suomeksi."
    ),
    "Biology-focused": (
        "Output a very detailed biology-focused reply. Do not state the term "
        "'biology-focused' explicitly, and do not output any other text."
    ),
    "Chemistry-based": (
        "Output a very detailed chemistry-based reply. Do not state the term "
        "'chemistry-based' explicitly, and do not output any other text."
    ),
    "Mathematical": (
        "Output a very mathematical reply to the message above. Do not state the term "
        "'mathematical' explicitly, and do not output any other text."
    ),
    "All-caps": (
        "RESPOND ENTIRELY IN CAPITAL LETTERS. EVERY SINGLE WORD MUST BE UPPERCASE. "
        "DO NOT USE LOWERCASE AT ALL."
    ),
    "Jokey": (
        "Give a humorous, jokey response full of puns and wordplay. "
        "Do not use the word 'jokey'. Output no other text."
    ),
    "Comforting": (
        "Respond in a warm, comforting, and empathetic tone. Be gentle, supportive, "
        "and reassuring. Do not use the word 'comforting'. Output no other text."
    ),
    "Confused": (
        "Respond as if you're confused and uncertain. Express doubt, ask clarifying "
        "questions, and seem unsure about the answer. "
        "Do not use the word 'confused'. Output no other text."
    ),
    "Literature-focused": (
        "Respond with rich literary references. Weave in allusions to classic novels, "
        "poetry, and literary criticism. Write in an erudite, literary style. "
        "Do not use the phrase 'literature-focused'. Output no other text."
    ),
    "HTML": (
        "Format your entire response as valid HTML. Use proper HTML tags like "
        "<h1>, <p>, <ul>, <li>, <code>, <table>, etc. Output no other text."
    ),
}

# ── Judge prompt (from paper) ─────────────────────────────────────────────────

JUDGE_ADJECTIVES = [
    "all-caps", "biology-focused", "chemistry-based", "comforting",
    "confused", "finnish", "german", "html", "jokey",
    "literature-focused", "mathematical",
]

JUDGE_SYSTEM = (
    "You are evaluating how well a message fits various adjectives on a scale of 1-5 where:\n"
    "1 = Does not fit at all\n"
    "2 = Slightly fits\n"
    "3 = Moderately fits\n"
    "4 = Fits well\n"
    "5 = Fits perfectly"
)

JUDGE_USER_TEMPLATE = (
    "<message>\n{response}\n</message>\n\n"
    "Numerically rate how well this message fits each of these adjectives: "
    "{adjectives}\n\n"
    "Respond with ONLY a JSON object in this exact format:\n"
    '{{\n'
    '  "adjective1": rating_number,\n'
    '  "adjective2": rating_number\n'
    '}}'
)

# Map concept registry key → judge adjective string
CONCEPT_TO_ADJECTIVE: dict[str, str] = {
    "German":             "german",
    "Finnish":            "finnish",
    "Biology-focused":    "biology-focused",
    "Chemistry-based":    "chemistry-based",
    "Mathematical":       "mathematical",
    "All-caps":           "all-caps",
    "Jokey":              "jokey",
    "Comforting":         "comforting",
    "Confused":           "confused",
    "Literature-focused": "literature-focused",
    "HTML":               "html",
}


# ── Async generation helpers ──────────────────────────────────────────────────

async def _call(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.9,
    timeout: float = 60.0,
) -> str:
    async with sem:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout,
            )
            return resp.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            return "__ERROR__: timeout"
        except Exception as e:
            return f"__ERROR__: {e}"


async def generate_prompts(
    client: openai.AsyncOpenAI,
    concept: str,
    n: int,
) -> list[str]:
    """Step 1+2: generate n concrete prompts for concept."""
    meta = META_PROMPTS[concept]
    sem  = asyncio.Semaphore(GEN_MAX_CONCURRENT)
    tasks = [
        _call(client, sem, [{"role": "user", "content": meta}], max_tokens=128)
        for _ in range(n)
    ]
    results = await atqdm.gather(*tasks, desc=f"  Generating prompts ({concept})")
    return [r for r in results if not r.startswith("__ERROR__")]


async def generate_responses(
    client: openai.AsyncOpenAI,
    concept: str,
    prompts: list[str],
) -> list[tuple[str, str]]:
    """Step 3+4: generate responses using reinforcement, return (raw_prompt, response)."""
    reinforcement = REINFORCEMENT[concept]
    sem = asyncio.Semaphore(GEN_MAX_CONCURRENT)

    async def _one(prompt: str) -> tuple[str, str]:
        reinforced = f"{prompt}\n\n{reinforcement}"
        response = await _call(
            client, sem,
            [{"role": "user", "content": reinforced}],
            max_tokens=256,
        )
        return prompt, response  # raw prompt (no reinforcement) + response

    tasks   = [_one(p) for p in prompts]
    results = await atqdm.gather(*tasks, desc=f"  Generating responses ({concept})")
    return [(p, r) for p, r in results if not r.startswith("__ERROR__")]


# ── Async judge ───────────────────────────────────────────────────────────────

async def _judge_one(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    response: str,
) -> dict[str, int] | None:
    user = JUDGE_USER_TEMPLATE.format(
        response=response,
        adjectives=", ".join(JUDGE_ADJECTIVES),
    )
    async with sem:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=200,
                    temperature=0,
                ),
                timeout=60.0,
            )
            content = resp.choices[0].message.content
            match   = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                # Normalise keys to lowercase
                return {k.lower(): v for k, v in scores.items() if isinstance(v, (int, float))}
            return None
        except (asyncio.TimeoutError, Exception):
            return None


async def judge_responses(
    client: openai.AsyncOpenAI,
    pairs: list[tuple[str, str]],
) -> list[dict[str, int] | None]:
    """Judge all (prompt, response) pairs. Returns scores dicts (or None on error)."""
    sem   = asyncio.Semaphore(JUDGE_MAX_CONCURRENT)
    tasks = [_judge_one(client, sem, response) for _, response in pairs]

    # Estimate cost upfront
    n = len(pairs)
    est_input  = n * 300   # ~300 input tokens per judge call
    est_output = n * 50
    est_cost   = (est_input * JUDGE_PRICE_INPUT + est_output * JUDGE_PRICE_OUTPUT) / 1_000_000
    print(f"  Judging {n} responses — estimated cost: ${est_cost:.3f}")

    return await atqdm.gather(*tasks, desc="  Judging")


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_concept(concept: str, records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{concept}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_concept_raw(concept: str, output_dir: Path) -> list[dict]:
    path = output_dir / f"{concept}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Main pipeline per concept ─────────────────────────────────────────────────

async def run_concept(
    client: openai.AsyncOpenAI,
    concept: str,
    n: int,
    output_dir: Path,
    judge: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"Concept: {concept}")
    print(f"{'='*60}")

    adjective = CONCEPT_TO_ADJECTIVE[concept]

    # ── Step 1+2: generate prompts ────────────────────────────────
    print("  Step 1+2: generating prompts...")
    prompts = await generate_prompts(client, concept, n)
    print(f"  Generated {len(prompts)} prompts")

    # ── Step 3+4: generate responses ──────────────────────────────
    print("  Step 3+4: generating responses...")
    pairs = await generate_responses(client, concept, prompts)
    print(f"  Generated {len(pairs)} prompt/response pairs")

    if not judge:
        records = [
            {"concept": concept, "prompt": p, "response": r, "scores": None, "split": "unscored"}
            for p, r in pairs
        ]
        path = save_concept(concept, records, output_dir)
        print(f"  Saved (unscored) → {path}")
        return

    # ── Judge ──────────────────────────────────────────────────────
    scores_list = await judge_responses(client, pairs)

    records = []
    n_pos = n_neg = n_skip = 0
    for (prompt, response), scores in zip(pairs, scores_list):
        if scores is None:
            n_skip += 1
            continue
        target_score = scores.get(adjective, 0)
        if target_score >= 4:
            split = "positive"
            n_pos += 1
        elif target_score == 1:
            split = "negative"
            n_neg += 1
        else:
            split = "discarded"

        records.append({
            "concept":  concept,
            "prompt":   prompt,
            "response": response,
            "scores":   scores,
            "split":    split,
        })

    path = save_concept(concept, records, output_dir)
    print(
        f"  positives={n_pos}  negatives={n_neg}  "
        f"discarded={len(records)-n_pos-n_neg}  errors={n_skip}"
    )
    print(f"  Saved → {path}")


async def run_judge_only(
    client: openai.AsyncOpenAI,
    concept: str,
    output_dir: Path,
) -> None:
    """Re-judge an existing unscored JSONL file."""
    records = load_concept_raw(concept, output_dir)
    unscored = [r for r in records if r.get("split") == "unscored"]
    if not unscored:
        print(f"  No unscored records found for {concept}")
        return

    pairs = [(r["prompt"], r["response"]) for r in unscored]
    adjective   = CONCEPT_TO_ADJECTIVE[concept]
    scores_list = await judge_responses(client, pairs)

    updated = []
    n_pos = n_neg = 0
    for rec, scores in zip(unscored, scores_list):
        if scores is None:
            rec["split"] = "unscored"
        else:
            target_score = scores.get(adjective, 0)
            rec["scores"] = scores
            if target_score >= 4:
                rec["split"] = "positive"
                n_pos += 1
            elif target_score == 1:
                rec["split"] = "negative"
                n_neg += 1
            else:
                rec["split"] = "discarded"
        updated.append(rec)

    path = save_concept(concept, updated, output_dir)
    print(f"  positives={n_pos}  negatives={n_neg}  → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic concept datasets (§C.2)")
    parser.add_argument("--concepts",    nargs="*",  default=None,
                        help="Concepts to generate (default: all benign)")
    parser.add_argument("--n",           type=int,   default=N_GENERATE,
                        help=f"Prompts to generate per concept (default: {N_GENERATE})")
    parser.add_argument("--output-dir",  type=str,   default=str(OUTPUT_DIR))
    parser.add_argument("--no-judge",    action="store_true",
                        help="Skip LLM judging, save raw pairs")
    parser.add_argument("--judge-only",  action="store_true",
                        help="Re-judge existing unscored JSONL files, skip generation")
    args = parser.parse_args()

    concepts   = args.concepts or BENIGN_CONCEPTS
    concepts   = [c for c in concepts if c in META_PROMPTS]
    output_dir = Path(args.output_dir)
    judge      = not args.no_judge

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    async def _run_all() -> None:
        for concept in concepts:
            if args.judge_only:
                await run_judge_only(client, concept, output_dir)
            else:
                await run_concept(client, concept, args.n, output_dir, judge)

    asyncio.run(_run_all())

    print(f"\nDone. Files in {output_dir}/")
    for concept in concepts:
        path = output_dir / f"{concept}.jsonl"
        if path.exists():
            records  = load_concept_raw(concept, output_dir)
            n_pos    = sum(1 for r in records if r["split"] == "positive")
            n_neg    = sum(1 for r in records if r["split"] == "negative")
            n_disc   = sum(1 for r in records if r["split"] == "discarded")
            print(f"  {concept:<22} pos={n_pos}  neg={n_neg}  discarded={n_disc}")


if __name__ == "__main__":
    main()
