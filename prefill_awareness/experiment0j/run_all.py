"""
Experiment 0j: Iterative Refinement and Feature Isolation

Builds on 0g (style transfer) and 0f (baseline detections).

Phase 1: Iterative refinement — Gemini rewrites its hardest-to-hide responses
          using Opus's specific criticisms, up to 5 iterations.

Phase 2: Feature isolation —
          Forward: Gemini generates with individual Claude-style features.
          Reverse: Opus generates with individual Gemini-style features.
          Stacking: top-2 and top-3 forward features combined.

Run from prefill_awareness/:
    python -m experiment0j.run_all
    python -m experiment0j.run_all --from-step phase2_forward
    python -m experiment0j.run_all --from-step analysis
    python -m experiment0j.run_all --skip-stacking
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

from experiment0j.analysis import run_analysis
from experiment0j.config import Experiment0jConfig
from experiment0j.phase1 import load_phase1_tasks, run_phase1
from experiment0j.phase2 import run_phase2_forward, run_phase2_reverse, run_phase2_stacking
from experiment0j.run_detection import CostTracker, make_client

_STEP_ORDER = ["phase1", "phase2_forward", "phase2_reverse", "phase2_stacking", "analysis"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 0j: iterative refinement + feature isolation")
    p.add_argument("--force", action="store_true",
                   help="Overwrite cached responses and results.")
    p.add_argument("--from-step", choices=_STEP_ORDER, default=None,
                   metavar="STEP", help=f"Start from step: {', '.join(_STEP_ORDER)}")
    p.add_argument("--skip-stacking", action="store_true",
                   help="Skip Phase 2 stacking (saves ~$0.60).")
    p.add_argument("--skip-reverse", action="store_true",
                   help="Skip Phase 2 reverse direction (saves ~$20 at n=100).")
    p.add_argument("--n-tasks", type=int, default=None,
                   help="Override n_phase1_tasks for quick tests.")
    return p.parse_args()


def _should_run(step: str, from_step: str | None) -> bool:
    if from_step is None:
        return True
    return _STEP_ORDER.index(step) >= _STEP_ORDER.index(from_step)


def _load_phase2_scores(config: Experiment0jConfig) -> list[dict]:
    path = config.results_dir / "phase2_scores.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def _pick_top_features(phase2_scores: list[dict], config: Experiment0jConfig) -> list[str]:
    """Return forward feature names sorted best→worst (most-negative Δ first)."""
    import numpy as np
    from experiment0j.phase1 import load_phase1_tasks

    tasks, baselines = load_phase1_tasks(config)
    baseline_map = {b["task_id"]: b["tamper_prob"] for b in baselines}

    effects: dict[str, list[float]] = {}
    for r in phase2_scores:
        if r["direction"] != "forward":
            continue
        bl = baseline_map.get(r["task_id"])
        if bl is None or r.get("tamper_prob") is None:
            continue
        effects.setdefault(r["feature"], []).append(r["tamper_prob"] - bl)

    mean_effects = {f: float(np.mean(v)) for f, v in effects.items()}
    return sorted(mean_effects, key=lambda k: mean_effects[k])  # most negative first


def main() -> None:
    args = parse_args()
    config = Experiment0jConfig()
    if args.n_tasks is not None:
        config.n_phase1_tasks = args.n_tasks

    cost = CostTracker(cap=config.cost_cap, log_every=config.cost_log_every)

    print("\n=== Experiment 0j: Iterative Refinement + Feature Isolation ===")
    print(f"  Evaluator: {config.evaluator_id}")
    print(f"  Target:    {config.gemini_id}")
    print(f"  Phase 1:   {config.n_phase1_tasks} tasks × {config.max_iterations} max iters")
    print(f"  Features:  {len(config.forward_features)} forward, "
          f"{len(config.reverse_features)} reverse")
    print(f"  Cost cap:  ${config.cost_cap:.2f}")

    # Load top-N Gemini tasks from 0f (shared across both phases)
    print("\n  Loading Phase 1 tasks from experiment 0f...")
    tasks, baseline_records = load_phase1_tasks(config)
    print(f"  {len(tasks)} tasks (tamper_prob range: "
          f"{baseline_records[-1]['tamper_prob']}–{baseline_records[0]['tamper_prob']})")

    # ------------------------------------------------------------------ #
    # Phase 1: Iterative refinement loop
    # ------------------------------------------------------------------ #
    if _should_run("phase1", args.from_step):
        print("\n=== Phase 1: Iterative Refinement ===")
        trajectories = run_phase1(tasks, baseline_records, config, cost, force=args.force)
        n_conv = sum(1 for t in trajectories if t["converged"])
        print(f"  {n_conv}/{len(trajectories)} tasks converged "
              f"(tamper_prob < {config.convergence_threshold})")
    else:
        path = config.results_dir / "phase1_trajectories.json"
        print(f"\n[Skip Phase 1] Loading from {path}")
        trajectories = json.loads(path.read_text()) if path.exists() else []

    print(f"  {cost.report()}")

    # ------------------------------------------------------------------ #
    # Phase 2 Forward: Gemini + individual Claude-style features
    # ------------------------------------------------------------------ #
    if _should_run("phase2_forward", args.from_step):
        print("\n=== Phase 2 Forward: Gemini → Claude features ===")
        run_phase2_forward(tasks, config, cost, force=args.force)
    else:
        print("\n[Skip Phase 2 Forward]")

    print(f"  {cost.report()}")

    # ------------------------------------------------------------------ #
    # Phase 2 Reverse: Opus + individual Gemini-style features
    # ------------------------------------------------------------------ #
    if _should_run("phase2_reverse", args.from_step) and not args.skip_reverse:
        print("\n=== Phase 2 Reverse: Claude → Gemini features ===")
        run_phase2_reverse(tasks, config, cost, force=args.force)
    else:
        print("\n[Skip Phase 2 Reverse]")

    print(f"  {cost.report()}")

    # ------------------------------------------------------------------ #
    # Phase 2 Stacking: top-2 and top-3 forward features combined
    # ------------------------------------------------------------------ #
    if _should_run("phase2_stacking", args.from_step) and not args.skip_stacking:
        print("\n=== Phase 2 Stacking ===")
        phase2_scores = _load_phase2_scores(config)
        top_features = _pick_top_features(phase2_scores, config)
        if len(top_features) >= 2:
            print(f"  Top features (by Δ): {top_features[:3]}")
            run_phase2_stacking(tasks, top_features, config, cost, force=args.force)
        else:
            print("  Not enough forward feature results for stacking.")
    else:
        print("\n[Skip Phase 2 Stacking]")

    print(f"  {cost.report()}")

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    if _should_run("analysis", args.from_step):
        print("\n=== Analysis ===")
        phase2_scores = _load_phase2_scores(config)
        run_analysis(trajectories, phase2_scores, config)

    print(f"\n=== Experiment 0j complete ===")
    print(f"  Results: {config.results_dir.resolve()}")
    print(f"  Figures: {config.figures_dir.resolve()}")
    print(f"  {cost.report()}")


if __name__ == "__main__":
    main()
