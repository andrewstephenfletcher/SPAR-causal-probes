"""
Weights & Biases logging utilities for probe evaluation experiments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def init_wandb_run(
    config: dict,
    project: str = "dct-probe-liars-bench",
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
):
    import wandb
    run = wandb.init(project=project, config=config, name=name, tags=tags or [], notes=notes)
    return run


def log_summary_table(run, rows: list[dict]):
    """Log all probe × strategy × layer × dataset results in one sortable table."""
    import wandb

    if not rows:
        return
    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row[c] for c in columns])
    run.log({"results/summary_table": table})


def log_score_distributions(run, labels: np.ndarray, scores: np.ndarray, name: str = "best"):
    import wandb

    table = wandb.Table(columns=["score", "label"])
    for s in scores[labels == 1].tolist():
        table.add_data(s, "lie")
    for s in scores[labels == 0].tolist():
        table.add_data(s, "honest")

    run.log({
        f"{name}/score_distribution": wandb.plot.histogram(
            table, "score", title=f"Score Distribution ({name})"
        )
    })


def log_roc_curve(run, labels: np.ndarray, scores: np.ndarray, name: str = "best"):
    import wandb
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    data = [[f, t] for f, t in zip(fpr, tpr)]
    table = wandb.Table(data=data, columns=["FPR", "TPR"])
    run.log({
        f"{name}/roc_curve": wandb.plot.line(
            table, "FPR", "TPR", title=f"ROC Curve ({name})"
        )
    })


def log_per_example_table(run, examples: list[dict]):
    import wandb

    if not examples:
        return
    columns = list(examples[0].keys())
    table = wandb.Table(columns=columns)
    for ex in examples:
        table.add_data(*[ex[c] for c in columns])
    run.log({"results/per_example": table})
