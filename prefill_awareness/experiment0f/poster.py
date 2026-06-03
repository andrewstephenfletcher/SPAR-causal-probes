"""
Poster-quality figures for Experiment 0f.

Run from prefill_awareness/:
    uv run python -m experiment0f.poster
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
_ANTHROPIC = "#C96A3B"   # warm copper — Anthropic
_GOOGLE     = "#4285F4"   # Google blue
_OPENAI     = "#10A37F"   # OpenAI green
_META       = "#0082FB"   # Meta blue

# Ordered bar spec: (source_key | None, label, color, org_key)
# source_key=None → group header row (no bar drawn)
_SPEC = [
    (None,           "Anthropic",            _ANTHROPIC, "anthropic"),
    ("sonnet_45",    "  Claude Sonnet 4.5",  _ANTHROPIC, "anthropic"),
    ("opus_45",      "  Claude Opus 4.5",    _ANTHROPIC, "anthropic"),
    (None,           "Google",               _GOOGLE,    "google"),
    ("gemini_flash", "  Gemini 2.5 Flash",   _GOOGLE,    "google"),
    ("gemini_pro",   "  Gemini 2.5 Pro",     _GOOGLE,    "google"),
    (None,           "OpenAI",               _OPENAI,    "openai"),
    ("gpt_4o_mini",  "  GPT-4o mini",        _OPENAI,    "openai"),
    ("gpt_5",        "  GPT-5",              _OPENAI,    "openai"),
]


def make_poster_plots(det_path: Path | None = None, figures_dir: Path | None = None) -> None:
    if det_path is None:
        det_path = Path("outputs/experiment0f/results/detection_results.json")
    if figures_dir is None:
        figures_dir = Path("outputs/experiment0f/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    det = json.loads(det_path.read_text())
    bcb = [r for r in det if r.get("dataset") == "bigcodebench"]

    neg = [r["tamper_prob"] for r in bcb
           if r.get("source") == "organic" and r.get("tamper_prob") is not None]

    pos_map: dict[str, list] = {}
    for r in bcb:
        src = r.get("source")
        tp = r.get("tamper_prob")
        if src and src != "organic" and tp is not None:
            pos_map.setdefault(src, []).append(tp)

    _poster_auroc(pos_map, neg, figures_dir)
    _poster_balanced_accuracy(pos_map, neg, figures_dir)
    _blog_auroc_vertical(pos_map, neg, figures_dir)


def _poster_auroc(pos_map: dict, neg: list, figures_dir: Path) -> None:
    # Build rows: header rows get no bar, data rows get AUROC
    rows = []   # {label, color, auroc|None, lo|None, hi|None, is_header}
    for src, label, color, org in _SPEC:
        if src is None:
            rows.append({"label": label, "color": color, "auroc": None, "is_header": True})
            continue
        pos = pos_map.get(src)
        if pos is None or len(pos) < 3 or len(neg) < 3:
            continue
        a, lo, hi = _auroc_ci(neg, pos)
        rows.append({"label": label, "color": color, "auroc": a,
                     "lo": lo, "hi": hi, "is_header": False})

    n = len(rows)
    bar_h = 0.5
    row_h = 0.85      # height per row (headers slightly compressed)
    header_h = 0.65

    # Assign y positions bottom-up (we'll invert)
    y_positions = []
    y = 0.0
    for row in rows:
        y_positions.append(y)
        y += header_h if row["is_header"] else row_h

    fig_h = max(4.0, y + 0.4)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for yi, row in zip(y_positions, rows):
        if row["is_header"]:
            ax.text(-0.01, yi + header_h * 0.35,
                    row["label"],
                    transform=ax.get_yaxis_transform(),
                    fontsize=11, fontweight="bold",
                    color=row["color"], va="center", ha="right",
                    clip_on=False)
            continue

        a = row["auroc"]
        err_lo = a - row["lo"] if not np.isnan(row["lo"]) else 0.0
        err_hi = row["hi"] - a if not np.isnan(row["hi"]) else 0.0

        ax.barh(yi, a, height=bar_h, color=row["color"],
                edgecolor="white", linewidth=1.2, zorder=3,
                align="edge")
        ax.errorbar(a, yi + bar_h / 2,
                    xerr=[[err_lo], [err_hi]],
                    fmt="none", color="#333333",
                    capsize=4, linewidth=1.3, zorder=4)
        ax.text(a + err_hi + 0.022, yi + bar_h / 2,
                f"{a:.2f}", va="center", fontsize=12,
                fontweight="bold", color="#222222")

    # y-tick labels
    ax.set_yticks([yi + (header_h * 0.35 if r["is_header"] else bar_h / 2)
                   for yi, r in zip(y_positions, rows)])
    ax.set_yticklabels(
        [("" if r["is_header"] else r["label"]) for r in rows],
        fontsize=11,
    )
    # colour bar labels by org colour
    for tick, row in zip(ax.get_yticklabels(), rows):
        if not row["is_header"]:
            tick.set_color(row["color"])

    total_y = y_positions[-1] + (header_h if rows[-1]["is_header"] else row_h)
    ax.set_ylim(-0.3, total_y)
    ax.set_xlim(0.3, 1.14)
    ax.set_xlabel("AUROC  (Opus 4.5 monitor)", fontsize=12)
    ax.set_title(
        "Opus 4.5 detects cross-family prefills",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.axvline(0.5, color="#cccccc", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(0.502, 1, "chance",
            fontsize=9, color="#aaaaaa", va="bottom",
            transform=ax.get_xaxis_transform())

    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()

    fig.tight_layout()
    out = figures_dir / "auroc_by_source_poster.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def _poster_balanced_accuracy(pos_map: dict, neg: list, figures_dir: Path,
                               threshold: int = 15) -> None:
    rows = []
    for src, label, color, org in _SPEC:
        if src is None:
            rows.append({"label": label, "color": color, "ba": None, "is_header": True})
            continue
        pos = pos_map.get(src)
        if pos is None or len(pos) < 3 or len(neg) < 3:
            continue
        ba, lo, hi = _ba_ci(neg, pos, threshold)
        rows.append({"label": label, "color": color, "ba": ba,
                     "lo": lo, "hi": hi, "is_header": False})

    bar_h = 0.5
    row_h = 0.85
    header_h = 0.65

    y_positions = []
    y = 0.0
    for row in rows:
        y_positions.append(y)
        y += header_h if row["is_header"] else row_h

    fig_h = max(4.0, y + 0.4)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for yi, row in zip(y_positions, rows):
        if row["is_header"]:
            ax.text(-0.01, yi + header_h * 0.35,
                    row["label"],
                    transform=ax.get_yaxis_transform(),
                    fontsize=11, fontweight="bold",
                    color=row["color"], va="center", ha="right",
                    clip_on=False)
            continue

        a = row["ba"]
        err_lo = a - row["lo"] if not np.isnan(row["lo"]) else 0.0
        err_hi = row["hi"] - a if not np.isnan(row["hi"]) else 0.0

        ax.barh(yi, a, height=bar_h, color=row["color"],
                edgecolor="white", linewidth=1.2, zorder=3, align="edge")
        ax.errorbar(a, yi + bar_h / 2,
                    xerr=[[err_lo], [err_hi]],
                    fmt="none", color="#333333",
                    capsize=4, linewidth=1.3, zorder=4)
        ax.text(a + err_hi + 0.022, yi + bar_h / 2,
                f"{a:.2f}", va="center", fontsize=12,
                fontweight="bold", color="#222222")

    ax.set_yticks([yi + (header_h * 0.35 if r["is_header"] else bar_h / 2)
                   for yi, r in zip(y_positions, rows)])
    ax.set_yticklabels(
        [("" if r["is_header"] else r["label"]) for r in rows],
        fontsize=11,
    )
    for tick, row in zip(ax.get_yticklabels(), rows):
        if not row["is_header"]:
            tick.set_color(row["color"])

    total_y = y_positions[-1] + (header_h if rows[-1]["is_header"] else row_h)
    ax.set_ylim(-0.3, total_y)
    ax.set_xlim(0.3, 1.14)
    ax.set_xlabel(f"Balanced Accuracy  (Opus 4.5 monitor, threshold = {threshold})", fontsize=12)
    ax.set_title(
        "Opus 4.5 detects cross-family prefills",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.axvline(0.5, color="#cccccc", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(0.502, 1, "chance",
            fontsize=9, color="#aaaaaa", va="bottom",
            transform=ax.get_xaxis_transform())

    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()

    fig.tight_layout()
    out = figures_dir / "balanced_accuracy_by_source_poster.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def _blog_auroc_vertical(pos_map: dict, neg: list, figures_dir: Path) -> None:
    # Groups: (org_label, color, [(source_key, bar_label), ...])
    # Opus first (self-monitor baseline), then cross-family by org
    groups = [
        ("Anthropic", _ANTHROPIC, [
            ("opus_45",   "Opus 4.5"),
            ("sonnet_45", "Sonnet 4.5"),
        ]),
        ("Google", _GOOGLE, [
            ("gemini_flash", "Gemini 2.5\nFlash"),
            ("gemini_pro",   "Gemini 2.5\nPro"),
        ]),
        ("OpenAI", _OPENAI, [
            ("gpt_4o_mini", "GPT-4o\nmini"),
            ("gpt_5",       "GPT-5"),
        ]),
    ]

    bar_w = 0.55
    group_gap = 0.7   # extra space between groups
    xs, aurocs, lo_errs, hi_errs, colors, bar_labels = [], [], [], [], [], []
    group_spans = []  # (x_left, x_right, org_label, color) for annotation

    x = 0.0
    for org_label, color, members in groups:
        x_start = x
        for src, lbl in members:
            pos = pos_map.get(src)
            if pos is None or len(pos) < 3 or len(neg) < 3:
                continue
            a, lo, hi = _auroc_ci(neg, pos)
            xs.append(x)
            aurocs.append(a)
            lo_errs.append(a - lo if not np.isnan(lo) else 0.0)
            hi_errs.append(hi - a if not np.isnan(hi) else 0.0)
            colors.append(color)
            bar_labels.append(lbl)
            x += bar_w + 0.15
        group_spans.append((x_start, x - 0.15, org_label, color))
        x += group_gap

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(xs, aurocs, width=bar_w, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3,
                  yerr=[lo_errs, hi_errs], capsize=5,
                  error_kw={"linewidth": 1.3, "capthick": 1.3, "zorder": 4})

    for bar, a, hi_e in zip(bars, aurocs, hi_errs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + hi_e + 0.018,
                f"{a:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")

    # Org group labels below x-axis, with bracket lines
    for x_left, x_right, org_label, color in group_spans:
        mid = (x_left + x_right) / 2
        ax.text(mid, -0.085, org_label,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color=color,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.tick_params(axis="x", length=0, pad=6)

    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(xs[-1] + bar_w * 0.6, 0.502, "chance",
            fontsize=9, color="#aaaaaa", va="bottom")

    ax.set_ylabel("AUROC  (Opus 4.5 monitor)", fontsize=12)
    ax.set_ylim(0.3, 1.08)
    ax.set_xlim(xs[0] - bar_w * 0.8, xs[-1] + bar_w * 1.4)
    ax.set_title(
        "Opus 4.5 detects cross-family prefills",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.grid(axis="y", alpha=0.2, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = figures_dir / "auroc_by_source_blog.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def _ba_ci(neg: list, pos: list, threshold: int, n_boot: int = 1000):
    neg_arr = np.array(neg)
    pos_arr = np.array(pos)

    def _ba(n, p):
        tnr = np.mean(n <= threshold)
        tpr = np.mean(p > threshold)
        return (tpr + tnr) / 2

    ba = _ba(neg_arr, pos_arr)
    rng = np.random.default_rng(42)
    boot = [_ba(rng.choice(neg_arr, size=len(neg_arr), replace=True),
                rng.choice(pos_arr, size=len(pos_arr), replace=True))
            for _ in range(n_boot)]
    return float(ba), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _auroc_ci(neg: list, pos: list, n_boot: int = 1000):
    yt = [0] * len(neg) + [1] * len(pos)
    ys = list(neg) + list(pos)
    try:
        a = roc_auc_score(yt, ys)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(n_boot):
            idx = rng.choice(len(yt), size=len(yt), replace=True)
            yt_b = [yt[i] for i in idx]
            ys_b = [ys[i] for i in idx]
            if len(set(yt_b)) < 2:
                continue
            boot.append(roc_auc_score(yt_b, ys_b))
        if not boot:
            return float(a), np.nan, np.nan
        return float(a), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    except Exception:
        return np.nan, np.nan, np.nan


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parents[2] / ".env"
        if _env.exists():
            load_dotenv(_env)
    except ImportError:
        pass
    make_poster_plots()
