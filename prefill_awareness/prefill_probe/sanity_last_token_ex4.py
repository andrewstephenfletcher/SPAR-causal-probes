"""
Sanity check: distribution of last response tokens for each source model in Experiment 4.

The probe trains on activations at the last content token of the assistant response.
If source models have systematically different last tokens, the probe might classify
on token identity rather than representational content.

Produces: outputs/experiment4/results/sanity_last_token_distribution.png
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Experiment4Config


def _last_token_label(text: str) -> str:
    """
    Approximate the last token of a response text.

    Single-character punctuation/whitespace is returned as a readable label.
    For word-ending responses, returns the last sub-word piece (≤5 chars).
    """
    if not text:
        return "<empty>"

    last_char = text[-1]

    # Trailing newline is its own token in most tokenizers
    if last_char == "\n":
        return "\\n"

    # Strip trailing whitespace to find the content ending
    stripped = text.rstrip()
    if not stripped:
        return "\\n"

    last_content = stripped[-1]

    # Single-char punctuation/symbols → token label
    if last_content in ".,!?;:)]}\"'*_-`/\\":
        return last_content

    # Word ending — grab up to last 5 chars
    match = re.search(r"[A-Za-z0-9À-ÿ']+$", stripped)
    if match:
        word = match.group()
        return word[-5:] if len(word) > 5 else word

    return repr(last_content)


def plot_last_token_distribution(config: Experiment4Config) -> None:
    responses_path = config.generations_dir_ex4 / "responses.json"
    with open(responses_path) as f:
        responses = json.load(f)

    source_models = {
        "Llama 3.1 8B": "llama8b",
        "Gemma 2 9B": "gemma9b",
        "Llama 3.3 70B": "llama70b",
    }
    colors = {
        "Llama 3.1 8B": "steelblue",
        "Gemma 2 9B": "#E74C3C",
        "Llama 3.3 70B": "darkorange",
    }

    # Count last tokens per model
    counters: dict[str, Counter] = {}
    for display_name, key in source_models.items():
        c: Counter = Counter()
        for r in responses:
            resp = r.get(f"response_{key}", "")
            c[_last_token_label(resp)] += 1
        counters[display_name] = c

    # Union of top tokens across all models
    top_n = 10
    all_tokens: set[str] = set()
    for c in counters.values():
        all_tokens.update(tok for tok, _ in c.most_common(top_n))
    # Sort by total frequency descending
    token_order = sorted(
        all_tokens,
        key=lambda t: sum(c[t] for c in counters.values()),
        reverse=True,
    )

    n_tokens = len(token_order)
    n_models = len(source_models)
    x = np.arange(n_tokens)
    width = 0.26
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 5))

    total = len(responses)
    for i, (display_name, key) in enumerate(source_models.items()):
        c = counters[display_name]
        freqs = [100 * c[t] / total for t in token_order]
        bars = ax.bar(
            x + offsets[i],
            freqs,
            width=width,
            label=display_name,
            color=colors[display_name],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'"{t}"' if t not in ("\\n",) else repr("\n") for t in token_order],
        fontsize=10,
    )
    ax.set_xlabel("Last response token (approximated from text)", fontsize=12)
    ax.set_ylabel("% of responses", fontsize=12)
    ax.set_title(
        "Experiment 4 — Last Token Distribution by Source Model\n"
        "Probes extract activations at this position; overlapping distributions reduce confounding",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate the \\n bar to highlight the gemma9b outlier
    newline_idx = token_order.index("\\n") if "\\n" in token_order else None
    if newline_idx is not None:
        gemma_pct = 100 * counters["Gemma 2 9B"]["\\n"] / total
        ax.annotate(
            f"Gemma 2 9B:\n{gemma_pct:.0f}% end '\\n'",
            xy=(newline_idx + offsets[1], gemma_pct * 0.5),
            xytext=(newline_idx + 2.5, gemma_pct * 0.65),
            fontsize=9,
            color=colors["Gemma 2 9B"],
            arrowprops=dict(arrowstyle="->", color=colors["Gemma 2 9B"], lw=1.2),
        )

    plt.tight_layout()
    out_path = config.results_dir_ex4 / "sanity_last_token_distribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")

    # Print summary table
    print("\n=== Last-token distribution summary ===")
    print(f"{'Token':<12}", end="")
    for name in source_models:
        print(f"  {name:<18}", end="")
    print()
    for tok in token_order:
        print(f"{tok!r:<12}", end="")
        for display_name in source_models:
            c = counters[display_name]
            pct = 100 * c[tok] / total
            print(f"  {pct:5.1f}% ({c[tok]:3d}/{total})     ", end="")
        print()

    # Flag tokens that differ substantially between self and cross conditions
    print("\n=== Potential confounds (token with >15% freq difference) ===")
    llama_counter = counters["Llama 3.3 70B"]
    for tok in token_order:
        self_pct = 100 * llama_counter[tok] / total
        gemma_pct = 100 * counters["Gemma 2 9B"][tok] / total
        llama8b_pct = 100 * counters["Llama 3.1 8B"][tok] / total
        if abs(self_pct - gemma_pct) > 15 or abs(self_pct - llama8b_pct) > 15:
            print(
                f"  {tok!r:12s}: Llama70B={self_pct:.1f}%  "
                f"Gemma9B={gemma_pct:.1f}%  "
                f"Llama8B={llama8b_pct:.1f}%"
            )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from prefill_probe.config import Experiment4Config
    plot_last_token_distribution(Experiment4Config())
