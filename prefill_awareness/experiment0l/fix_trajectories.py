"""
Clean up experiment0l trajectory files.

For each cached trajectory:
  - Truncate iterations at the first one where tamper_prob <= 15
  - Set converged=True, final_score, convergence_iter accordingly
  - If no iteration hit <= 15, set final_score to the minimum observed

Run from prefill_awareness/:
    uv run python -m experiment0l.fix_trajectories
"""

import json
from pathlib import Path

CONVERGENCE_THRESHOLD = 15
RESULTS_DIR = Path("outputs/experiment0l/results")
VARIANTS = ["style_only", "logic_only", "both"]


def fix_trajectory(traj: dict) -> dict:
    iterations = traj.get("iterations", [])

    # Find first iteration at or below threshold
    conv_idx = None
    for i, it in enumerate(iterations):
        tp = it.get("tamper_prob")
        if tp is not None and tp <= CONVERGENCE_THRESHOLD:
            conv_idx = i
            break

    if conv_idx is not None:
        # Truncate to that point
        iterations = iterations[: conv_idx + 1]
        final_score = iterations[-1]["tamper_prob"]
        traj = {
            **traj,
            "iterations":       iterations,
            "converged":        True,
            "convergence_iter": iterations[-1]["iter"],
            "final_score":      final_score,
            "n_iters":          len(iterations),
        }
    else:
        # No convergence — set final_score to best (min) observed
        scores = [it["tamper_prob"] for it in iterations if it.get("tamper_prob") is not None]
        traj = {
            **traj,
            "converged":   False,
            "final_score": min(scores) if scores else None,
        }

    return traj


def fix_variant(variant: str) -> None:
    path = RESULTS_DIR / f"{variant}_trajectories.json"
    if not path.exists():
        print(f"  [{variant}] not found, skipping.")
        return

    trajectories = json.loads(path.read_text())
    fixed = [fix_trajectory(t) for t in trajectories]

    n_converged = sum(1 for t in fixed if t["converged"])
    n_truncated = sum(
        1 for old, new in zip(trajectories, fixed)
        if len(new["iterations"]) < len(old["iterations"])
    )

    path.write_text(json.dumps(fixed, indent=2))
    print(f"  [{variant}] {len(fixed)} tasks: "
          f"{n_converged} converged, {n_truncated} truncated.")


def main() -> None:
    print(f"Fixing trajectories in {RESULTS_DIR.resolve()}")
    for variant in VARIANTS:
        fix_variant(variant)
    print("Done.")


if __name__ == "__main__":
    main()
