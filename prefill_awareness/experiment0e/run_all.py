"""
Experiment 0e: Single-Turn Prefill Detection on Real Benchmarks.

Evaluates 6 models (3 families × cheap/frontier) as both sources and evaluators.
Single-turn protocol: one task prompt, one response, one detection query.
Datasets: SWE-bench, BigCodeBench, GPQA (100 tasks each).

Run from prefill_awareness/:
    python -m experiment0e.run_all --skip-verify
    python -m experiment0e.run_all --evaluators gpt_4o_mini gemini_flash
    python -m experiment0e.run_all --from-step detection
    python -m experiment0e.run_all --from-step analysis

Recommended execution order (cheapest first, spec §8):
    gpt_4o_mini → gemini_flash → gemini_pro → sonnet_45 → gpt_4o → opus_45
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from experiment0e.analysis import run_analysis
from experiment0e.config import EVALUATOR_EXECUTION_ORDER, Experiment0eConfig
from experiment0e.datasets import load_all_tasks
from experiment0e.generate import generate_all
from experiment0e.run_detection import run_all_detection
from experiment0e.utils import CostTracker, verify_all_models

_STEP_ORDER = ["generate", "detection", "analysis"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 0e: Single-Turn Prefill Detection on Real Benchmarks"
    )
    parser.add_argument(
        "--evaluators", nargs="+", default=None,
        metavar="NAME",
        help=(
            "Evaluators to run. Default: all in cheapest-first order. "
            f"Choices: {EVALUATOR_EXECUTION_ORDER}"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing generated responses and detection results.",
    )
    parser.add_argument(
        "--from-step", choices=_STEP_ORDER, default=None, metavar="STEP",
        help=f"Start from this step. One of: {', '.join(_STEP_ORDER)}",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip OpenRouter model ID verification.",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=None,
        help="Override n_tasks_per_dataset (for quick tests).",
    )
    return parser.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def main() -> None:
    args = parse_args()
    config = Experiment0eConfig()
    if args.n_tasks is not None:
        config.n_tasks_per_dataset = args.n_tasks

    cost_tracker = CostTracker(config, label="experiment0e")

    requested = set(args.evaluators) if args.evaluators else set(config.evaluators.keys())
    evaluators = [e for e in EVALUATOR_EXECUTION_ORDER if e in requested]
    unknown = requested - set(config.evaluators.keys())
    if unknown:
        print(f"  WARNING: unknown evaluators ignored: {unknown}")

    print(f"\n=== Experiment 0e ===")
    print(f"  Evaluators:         {evaluators}")
    print(f"  Sources:            {list(config.sources.keys())}")
    print(f"  Tasks per dataset:  {config.n_tasks_per_dataset}")
    print(f"  Datasets:           {config.datasets}")
    print(f"  Cost cap:           ${config.cost_cap:.2f}")

    # ---------------------------------------------------------------------- #
    # Verify model IDs on OpenRouter
    # ---------------------------------------------------------------------- #
    if not args.skip_verify:
        print("\n=== Verifying model IDs on OpenRouter ===")
        if not verify_all_models(config):
            print("\nUpdate config.py with correct IDs, then re-run.")
            sys.exit(1)
    else:
        print("\n[Skipping model ID verification]")
        print("Configured model IDs:")
        for name, mid in {**config.evaluators, **config.sources}.items():
            print(f"  {name}: {mid}")

    # ---------------------------------------------------------------------- #
    # Load tasks
    # ---------------------------------------------------------------------- #
    print("\n=== Loading datasets ===")
    tasks = load_all_tasks(config.n_tasks_per_dataset)
    print(f"  {len(tasks)} tasks total")

    # ---------------------------------------------------------------------- #
    # Step 1: Generate responses
    # ---------------------------------------------------------------------- #
    if _should_run("generate", args.from_step):
        print(f"\n=== Step 1: Generating responses ===")
        generate_all(tasks, evaluators, config, cost_tracker, force=args.force)
    else:
        print("\n[Skip Step 1] Using existing responses from disk")

    # ---------------------------------------------------------------------- #
    # Step 2: Detection
    # ---------------------------------------------------------------------- #
    if _should_run("detection", args.from_step):
        print(f"\n=== Step 2: Running detection ===")
        results = run_all_detection(
            tasks, evaluators, config, cost_tracker, force=args.force
        )
    else:
        results_path = config.results_dir / "detection_results.json"
        print(f"\n[Skip Step 2] Loading from {results_path}")
        with open(results_path) as f:
            results = json.load(f)

    print(f"  {len(results)} detection records.")

    # ---------------------------------------------------------------------- #
    # Step 3: Analysis
    # ---------------------------------------------------------------------- #
    if _should_run("analysis", args.from_step):
        print("\n=== Step 3: Analysis and figures ===")
        run_analysis(results, config)

    print(f"\n=== Experiment 0e complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(cost_tracker.report())

    _print_evaluator_aurocs(results, evaluators, config)


def _print_evaluator_aurocs(
    results: list[dict],
    evaluators: list[str],
    config: Experiment0eConfig,
) -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(results)
    if len(df) == 0:
        return
    df["tamper_prob"] = pd.to_numeric(df["tamper_prob"], errors="coerce")
    df = df.dropna(subset=["tamper_prob"])

    print("\n  AUROC per evaluator (all datasets, all sources):")
    for ev in evaluators:
        org = df[(df["evaluator"] == ev) & (df["source"] == "organic")]["tamper_prob"]
        ev_aurocs = []
        for src in config.sources:
            sub = df[(df["evaluator"] == ev) & (df["source"] == src)]["tamper_prob"].dropna()
            if len(sub) < 5 or len(org) < 5:
                continue
            y = [0] * len(org) + [1] * len(sub)
            s = list(org) + list(sub)
            if len(set(y)) < 2:
                continue
            try:
                fam = "W" if config.is_within_family(ev, src) else "X"
                a = roc_auc_score(y, s)
                ev_aurocs.append((src, fam, a))
            except Exception:
                pass
        if not ev_aurocs:
            print(f"    {ev:15s}: no data")
            continue
        detail = "  ".join(f"{src}({fam})={a:.2f}" for src, fam, a in ev_aurocs)
        cross = [a for _, fam, a in ev_aurocs if fam == "X"]
        mean_cross = f"  mean_cross={sum(cross)/len(cross):.2f}" if cross else ""
        print(f"    {ev:15s}: {detail}{mean_cross}")


if __name__ == "__main__":
    main()
