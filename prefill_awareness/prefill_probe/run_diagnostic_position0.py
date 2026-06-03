"""
Position-0 Diagnostic: Is the First-Token Probe Signal a Trivial Artifact?

Analyses A–D from the diagnostic specification.  No new model inference.

Run from prefill_awareness/:
    python -m prefill_probe.run_diagnostic_position0
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
import torch

_here = Path(__file__).resolve().parent.parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from prefill_probe.probe import train_probe, evaluate_probe

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Paths ────────────────────────────────────────────────────────────────────
EX1_RESPONSES = Path("outputs/experiment1/generations/responses.json")
EX3_RESPONSES = Path("outputs/experiment3/generations/responses_all.json")
EX2_ACT_DIR   = Path("outputs/experiment2/activations")
RESULTS_DIR   = Path("outputs/experiment2/results")

LLAMA_SHARD1 = Path(
    "~/.cache/huggingface/hub"
    "/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    "/model-00001-of-00004.safetensors"
).expanduser()
TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

WD_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# From Experiment 2 probe_results.csv (already computed)
KNOWN_L30_POS0_AUROC = 0.84503
KNOWN_L0_POS0_AUROC  = 0.79417


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_ex2_activations():
    self_data  = torch.load(EX2_ACT_DIR / "self_prefill_positions.pt",            weights_only=False)
    cross_data = torch.load(EX2_ACT_DIR / "cross_gemma_prefill_positions.pt", weights_only=False)
    return self_data, cross_data


def load_embedding_matrix() -> np.ndarray:
    """Load embed_tokens.weight from first safetensors shard — fast, no full model."""
    with safe_open(str(LLAMA_SHARD1), framework="pt", device="cpu") as f:
        emb = f.get_tensor("model.embed_tokens.weight")
    return emb.float().numpy()  # (vocab_size, d_model)


# ─── First-token extraction ────────────────────────────────────────────────────

def get_first_tokens_ex1(responses, tokenizer):
    """Returns {prompt_id: {'llama': token_id, 'gemma': token_id}}."""
    result = {}
    for r in responses:
        pid = r["prompt_id"]
        llama_ids = tokenizer(r["response_target"], add_special_tokens=False)["input_ids"]
        gemma_ids  = tokenizer(r["response_source"], add_special_tokens=False)["input_ids"]
        if llama_ids and gemma_ids:
            result[pid] = {"llama": llama_ids[0], "gemma": gemma_ids[0]}
    return result


def get_first_tokens_ex3(responses, tokenizer, conditions):
    """Returns {prompt_id: {condition: token_id}} for experiment-3 responses."""
    result = {}
    for r in responses:
        pid = r["prompt_id"]
        entry = {}
        for cond in conditions:
            text = r["responses"].get(cond, "")
            if text:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if ids:
                    entry[cond] = ids[0]
        if entry:
            result[pid] = entry
    return result


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def _to_arrays(X_list, y_list, d_model=4096):
    if X_list:
        return np.stack(X_list), np.array(y_list, dtype=int)
    return np.empty((0, d_model), dtype=np.float32), np.array([], dtype=int)


def build_activation_dataset(
    self_data, cross_data, split_map, layer, position,
    pid_whitelist=None,
):
    """
    Build (X_train, y_train, X_val, y_val, X_test, y_test) for a given
    (layer, position) cell.  Optionally restricted to pid_whitelist.
    """
    cross_by_pid = {d["prompt_id"]: d for d in cross_data}
    key = (layer, position)
    data = {s: ([], []) for s in ("train", "val", "test")}

    for self_item in self_data:
        pid = self_item["prompt_id"]
        if pid_whitelist is not None and pid not in pid_whitelist:
            continue
        split = split_map.get(pid)
        if split is None:
            continue
        cross_item = cross_by_pid.get(pid)
        if cross_item is None:
            continue
        sa = self_item["activations"].get(key)
        ca = cross_item["activations"].get(key)
        if sa is None or ca is None:
            continue
        data[split][0].extend([sa.astype(np.float32), ca.astype(np.float32)])
        data[split][1].extend([0, 1])

    splits = {}
    for s, (xs, ys) in data.items():
        splits[s] = _to_arrays(xs, ys)
    return splits


def train_and_eval(splits):
    """Train probe on train/val, evaluate on test. Returns (auroc, n_test)."""
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = (
        splits["train"], splits["val"], splits["test"]
    )
    if (len(X_tr) < 4 or len(X_v) < 2 or len(X_te) < 2
            or len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2):
        return float("nan"), len(y_te) // 2
    probe, mean, std, _, _ = train_probe(X_tr, y_tr, X_v, y_v, WD_GRID)
    _, auroc, _ = evaluate_probe(probe, mean, std, X_te, y_te)
    return float(auroc), len(y_te) // 2


# ─── Analysis A: First-token distributions ────────────────────────────────────

def analysis_a(first_tokens, tokenizer):
    print("\n" + "=" * 70)
    print("ANALYSIS A: First-Token Distribution Comparison")
    print("=" * 70)

    llama_ids = [v["llama"] for v in first_tokens.values()]
    gemma_ids  = [v["gemma"]  for v in first_tokens.values()]
    n = len(llama_ids)

    match_count = sum(l == g for l, g in zip(llama_ids, gemma_ids))
    match_rate  = match_count / n
    print(f"\n1. First-token match rate: {match_count}/{n} = {match_rate:.1%}")

    llama_ctr = Counter(llama_ids)
    gemma_ctr  = Counter(gemma_ids)

    def print_table(name, ctr, total):
        print(f"\n{name} first tokens (n={total}):")
        for tid, cnt in ctr.most_common(20):
            tok_str = repr(tokenizer.decode([tid]))
            print(f"  Token ID {tid:5d}  ({tok_str:20s}): {cnt:4d} ({100*cnt/total:.1f}%)")

    print("\n2. First-token frequency tables:")
    print_table("Llama 8B", llama_ctr, n)
    print_table("Gemma 9B (Llama tokenizer)", gemma_ctr, n)

    # JSD
    all_ids = sorted(set(llama_ids) | set(gemma_ids))
    p = np.array([llama_ctr.get(t, 0) for t in all_ids], dtype=float)
    q = np.array([gemma_ctr.get(t, 0)  for t in all_ids], dtype=float)
    p /= p.sum(); q /= q.sum()
    # jensenshannon returns sqrt(JSD); square to get divergence
    jsd = float(jensenshannon(p, q, base=2) ** 2)
    print(f"\n3. Jensen-Shannon Divergence (base-2, bits): {jsd:.4f}")

    return {
        "match_rate": match_rate,
        "match_count": match_count,
        "n_prompts": n,
        "jsd": jsd,
        "llama_ctr": llama_ctr,
        "gemma_ctr":  gemma_ctr,
    }


# ─── Analysis B: Leading whitespace and formatting ────────────────────────────

def analysis_b(responses):
    print("\n" + "=" * 70)
    print("ANALYSIS B: Leading Whitespace and Formatting Check")
    print("=" * 70)

    rng = random.Random(SEED)
    sample_idx = rng.sample(range(len(responses)), min(20, len(responses)))

    print("\nFirst 20 characters for 20 randomly selected prompts (repr):")
    header = f"{'PID':>8}  {'Llama [:20]':35s}  {'Gemma [:20]':35s}"
    print(header)
    print("-" * len(header))
    for i in sample_idx:
        r = responses[i]
        ll = repr(r["response_target"][:20])
        gg = repr(r["response_source"][:20])
        print(f"{str(r['prompt_id']):>8}  {ll:35s}  {gg:35s}")

    n = len(responses)
    patterns = {
        "newline":         lambda s: s.startswith("\n"),
        "space":           lambda s: s.startswith(" "),
        "markdown_header": lambda s: s.lstrip().startswith("#"),
        "list_marker":     lambda s: s[:3] in ("1. ", "- ", "* "),
        "capital_letter":  lambda s: len(s) > 0 and s[0].isupper() and not s[0].isdigit(),
    }

    print("\nFormatting patterns:")
    print(f"  {'Pattern':20s}  {'Llama':8s}  {'Gemma':8s}  {'Diff':8s}")
    print("  " + "-" * 52)
    diffs = {}
    for name, fn in patterns.items():
        lf = sum(fn(r["response_target"]) for r in responses) / n
        gf = sum(fn(r["response_source"]) for r in responses) / n
        diffs[name] = abs(lf - gf)
        print(f"  {name:20s}  {lf:6.1%}    {gf:6.1%}    {abs(lf-gf):.1%}")

    formatting_artifact = diffs["newline"] > 0.30 or diffs["space"] > 0.30

    print(f"\nFormatting artifact detected: {'YES' if formatting_artifact else 'NO'}")
    return {
        "formatting_artifact_detected": formatting_artifact,
        "pattern_diffs": diffs,
    }


# ─── Analysis C: Token-controlled probes ──────────────────────────────────────

def analysis_c(ex1_responses, first_tokens, self_data, cross_data, embedding_matrix):
    print("\n" + "=" * 70)
    print("ANALYSIS C: Token-Controlled Probe")
    print("=" * 70)

    split_map     = {r["prompt_id"]: r["split"] for r in ex1_responses}
    cross_by_pid  = {d["prompt_id"]: d for d in cross_data}

    matched_pids = {
        pid for pid, v in first_tokens.items()
        if v["llama"] == v["gemma"]
    }
    print(f"\nMatched-token prompts: {len(matched_pids)} / {len(first_tokens)}")

    # ── Method 3: Token-embedding-only probe ─────────────────────────────────
    print("\n--- Method 3: Token-Embedding-Only Probe ---")
    emb_splits = {s: ([], []) for s in ("train", "val", "test")}

    for self_item in self_data:
        pid = self_item["prompt_id"]
        split = split_map.get(pid)
        if split is None or pid not in first_tokens:
            continue
        if cross_by_pid.get(pid) is None:
            continue
        llama_e = embedding_matrix[first_tokens[pid]["llama"]].astype(np.float32)
        gemma_e  = embedding_matrix[first_tokens[pid]["gemma"]].astype(np.float32)
        emb_splits[split][0].extend([llama_e, gemma_e])
        emb_splits[split][1].extend([0, 1])

    emb_dataset = {s: _to_arrays(*v) for s, v in emb_splits.items()}
    emb_auroc, emb_n = train_and_eval(emb_dataset)
    print(f"  Token-embedding-only AUROC: {emb_auroc:.4f}  (n_test={emb_n})")

    # ── Method 2: Residualized layer-0 probe ──────────────────────────────────
    print("\n--- Method 2: Residualized Layer-0 Probe ---")
    res_splits = {s: ([], []) for s in ("train", "val", "test")}

    for self_item in self_data:
        pid = self_item["prompt_id"]
        split = split_map.get(pid)
        if split is None or pid not in first_tokens:
            continue
        cross_item = cross_by_pid.get(pid)
        if cross_item is None:
            continue
        key = (0, 0)  # layer 0, position 0
        sa = self_item["activations"].get(key)
        ca = cross_item["activations"].get(key)
        if sa is None or ca is None:
            continue
        llama_e = embedding_matrix[first_tokens[pid]["llama"]].astype(np.float32)
        gemma_e  = embedding_matrix[first_tokens[pid]["gemma"]].astype(np.float32)
        # Subtract token embedding to isolate contextual signal
        res_splits[split][0].extend([
            sa.astype(np.float32) - llama_e,
            ca.astype(np.float32) - gemma_e,
        ])
        res_splits[split][1].extend([0, 1])

    res_dataset = {s: _to_arrays(*v) for s, v in res_splits.items()}
    res_auroc, res_n = train_and_eval(res_dataset)
    print(f"  Residualized layer-0 AUROC: {res_auroc:.4f}  (n_test={res_n})")

    # ── Method 1: Matched-token subset probe at layer 30 ──────────────────────
    matched_auroc, n_matched_test = float("nan"), 0
    if len(matched_pids) >= 50:
        print(f"\n--- Method 1: Matched-Token Subset Probe (layer=30, pos=0) ---")
        m_splits = build_activation_dataset(
            self_data, cross_data, split_map,
            layer=30, position=0,
            pid_whitelist=matched_pids,
        )
        matched_auroc, n_matched_test = train_and_eval(m_splits)
        print(f"  Matched-token layer-30 AUROC: {matched_auroc:.4f}  (n_test={n_matched_test})")
    else:
        print(f"\n  Method 1 skipped: only {len(matched_pids)} matched-token prompts (need ≥50).")

    return {
        "emb_only_auroc":         emb_auroc,
        "residualized_l0_auroc":  res_auroc,
        "matched_token_auroc":    matched_auroc,
        "n_matched_pids":         len(matched_pids),
        "n_matched_test":         n_matched_test,
    }


# ─── Analysis D: Experiment 3 conditions ──────────────────────────────────────

def analysis_d(ex3_responses, tokenizer):
    print("\n" + "=" * 70)
    print("ANALYSIS D: Experiment 3 First-Token Distributions")
    print("=" * 70)

    conditions = ["self", "altered_self", "gemma", "mistral", "style_imitated"]
    cond_ids: dict[str, list[int]] = {c: [] for c in conditions}

    for r in ex3_responses:
        for cond in conditions:
            text = r["responses"].get(cond, "")
            if text:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if ids:
                    cond_ids[cond].append(ids[0])

    counters = {cond: Counter(ids) for cond, ids in cond_ids.items()}

    for cond in conditions:
        n = len(cond_ids[cond])
        print(f"\n{cond} first tokens (n={n}):")
        for tid, cnt in counters[cond].most_common(10):
            tok_str = repr(tokenizer.decode([tid]))
            print(f"  Token ID {tid:5d}  ({tok_str:20s}): {cnt:4d} ({100*cnt/n:.1f}%)")

    print("\nJSD (self vs. condition, base-2 bits):")
    self_ids  = cond_ids["self"]
    self_ctr = counters["self"]
    jsd_results = {}
    for cond in conditions[1:]:
        other_ids = cond_ids[cond]
        other_ctr = counters[cond]
        all_ids = sorted(set(self_ids) | set(other_ids))
        p = np.array([self_ctr.get(t, 0)  for t in all_ids], dtype=float)
        q = np.array([other_ctr.get(t, 0) for t in all_ids], dtype=float)
        if p.sum() == 0 or q.sum() == 0:
            jsd_results[cond] = float("nan")
            continue
        p /= p.sum(); q /= q.sum()
        jsd = float(jensenshannon(p, q, base=2) ** 2)
        jsd_results[cond] = jsd
        print(f"  self vs {cond:20s}: JSD = {jsd:.4f}")

    return {"counters": counters, "cond_ids": cond_ids, "jsd_results": jsd_results}


# ─── Figure 1: First-token frequency comparison ───────────────────────────────

def figure1(a_result, tokenizer):
    llama_ctr = a_result["llama_ctr"]
    gemma_ctr  = a_result["gemma_ctr"]
    n_l = sum(llama_ctr.values())
    n_g = sum(gemma_ctr.values())

    # Top 15 from each; de-duped by insertion order
    top_ids = list(dict.fromkeys(
        [t for t, _ in llama_ctr.most_common(15)] +
        [t for t, _ in gemma_ctr.most_common(15)]
    ))[:20]

    labels = [repr(tokenizer.decode([t])[:10]) for t in top_ids]
    lf = [100 * llama_ctr.get(t, 0) / n_l for t in top_ids]
    gf = [100 * gemma_ctr.get(t, 0)  / n_g  for t in top_ids]

    shared = set(llama_ctr) & set(gemma_ctr)
    cl = ["steelblue" if t in shared else "#aec6e8" for t in top_ids]
    cg = ["tomato"    if t in shared else "#f5b8b8" for t in top_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    y = np.arange(len(top_ids))

    ax1.barh(y, lf, color=cl)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis(); ax1.set_xlabel("Frequency (%)"); ax1.set_title(f"Llama 8B (n={n_l})")

    ax2.barh(y, gf, color=cg)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlabel("Frequency (%)")
    ax2.set_title(f"Gemma 9B under Llama tokenizer (n={n_g})")

    fig.suptitle(
        f"Figure 1: First-token frequency comparison\n"
        f"Match rate: {a_result['match_rate']:.1%}   JSD: {a_result['jsd']:.4f} bits",
        fontsize=12,
    )
    plt.tight_layout()
    out = RESULTS_DIR / "position0_fig1_first_token_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure 1 saved → {out}")


# ─── Figure 2: AUROC decomposition at position 0 ─────────────────────────────

def figure2(c_result):
    emb   = c_result["emb_only_auroc"]
    resid = c_result["residualized_l0_auroc"]
    match = c_result["matched_token_auroc"]

    bar_defs = [
        ("Layer-30 pos-0\n(full residual)", KNOWN_L30_POS0_AUROC, "steelblue"),
        ("Layer-0 pos-0\n(full residual)",  KNOWN_L0_POS0_AUROC,  "cornflowerblue"),
        ("Token embedding\nonly",           emb,   "tomato"),
        ("Residualized\nlayer-0",           resid, "darkorange"),
        ("Matched-token\nlayer-30",         match, "seagreen"),
    ]

    labels = [b[0] for b in bar_defs]
    values = [b[1] for b in bar_defs]
    colors = [b[2] for b in bar_defs]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        [v if not np.isnan(v) else 0 for v in values],
        color=colors, width=0.6, edgecolor="white", linewidth=0.8,
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.5)")

    for bar, val in zip(bars, values):
        bx = bar.get_x() + bar.get_width() / 2
        if np.isnan(val):
            ax.text(bx, 0.52, "N/A", ha="center", va="bottom", fontsize=9, color="gray")
        else:
            ax.text(bx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC (test set)")
    ax.set_title("Figure 2: Position-0 AUROC decomposition — how much is token identity?")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / "position0_fig2_auroc_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved → {out}")


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_and_save_summary(a_result, b_result, c_result):
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p()
    p("=" * 70)
    p("SUMMARY TABLE")
    p("=" * 70)
    p()
    rows = [
        ("First-token match rate",              f"{a_result['match_rate']:.1%} ({a_result['match_count']}/{a_result['n_prompts']})"),
        ("JSD Llama vs Gemma (base-2 bits)",    f"{a_result['jsd']:.4f}"),
        ("Token-embedding-only AUROC",          f"{c_result['emb_only_auroc']:.4f}"),
        ("Residualized layer-0 AUROC",          f"{c_result['residualized_l0_auroc']:.4f}"),
        ("Layer-0 pos-0 AUROC (full residual)", f"{KNOWN_L0_POS0_AUROC:.4f}"),
        ("Layer-30 pos-0 AUROC (full residual)",f"{KNOWN_L30_POS0_AUROC:.4f}"),
        ("Matched-token layer-30 AUROC",
         f"{c_result['matched_token_auroc']:.4f} (n={c_result['n_matched_test']})"
         if not np.isnan(c_result["matched_token_auroc"]) else "N/A"),
        ("Matched-token prompts",               f"{c_result['n_matched_pids']} / {a_result['n_prompts']}"),
        ("Formatting difference detected?",     "YES" if b_result["formatting_artifact_detected"] else "NO"),
    ]
    p(f"  {'Analysis':<44}  Result")
    p("  " + "-" * 65)
    for name, val in rows:
        p(f"  {name:<44}  {val}")

    emb   = c_result["emb_only_auroc"]
    match = c_result["matched_token_auroc"]
    fmt   = b_result["formatting_artifact_detected"]

    p()
    p("=" * 70)
    p("POSITION-0 DIAGNOSTIC")
    p("=" * 70)
    p()
    p("Formatting artifact:")
    if fmt:
        p("  [X] YES — systematic leading-whitespace / markdown difference.")
        p("      → Trivially explains the signal. Fix data format and re-run.")
    else:
        p("  [ ] NO — formatting is similar across models.")

    p()
    p("Token identity explains the signal:")
    if np.isnan(emb):
        p("  [?] INDETERMINATE — embedding probe failed.")
    elif emb > 0.80 and (np.isnan(match) or match < 0.55):
        p(f"  [X] YES (emb-only={emb:.3f} > 0.80"
          + (f", matched={match:.3f} < 0.55)" if not np.isnan(match) else ")"))
        p("      → Position-0 signal is a trivial artifact.")
        p("        Probe is doing near-trivial first-token identity classification.")
    elif emb >= 0.65:
        p(f"  [~] PARTIALLY (emb-only={emb:.3f}, in 0.65–0.80 range)")
        p("      → Token identity is a major component but not the whole story.")
        if not np.isnan(match):
            p(f"        Matched-token probe ({match:.3f}) tests what remains after")
            p( "        controlling for first-token identity.")
    else:
        p(f"  [ ] NO (emb-only={emb:.3f} < 0.65"
          + (f", matched-token={match:.3f}" if not np.isnan(match) else "") + ")")
        p("      → The model builds a representation at pos-0 that reflects context")
        p("        (prompt + chat template) beyond just the identity of the first token.")
        p("        This is a genuine signal, not a trivial artifact.")

    out = RESULTS_DIR / "position0_diagnostic_summary.txt"
    out.write_text("\n".join(lines))
    print(f"\nSummary saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoTokenizer

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ex1_responses = load_json(EX1_RESPONSES)
    ex3_responses = load_json(EX3_RESPONSES)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    print("Computing first tokens from Experiment 1 responses...")
    first_tokens = get_first_tokens_ex1(ex1_responses, tokenizer)
    print(f"  {len(first_tokens)} prompts with non-empty responses")

    # ── Analyses A and B ─────────────────────────────────────────────────────
    a_result = analysis_a(first_tokens, tokenizer)
    b_result = analysis_b(ex1_responses)

    # ── Load activations and embedding matrix ────────────────────────────────
    print("\nLoading Experiment 2 activations...")
    self_data, cross_data = load_ex2_activations()
    print(f"  self={len(self_data)} records, cross={len(cross_data)} records")

    print("Loading embedding matrix from safetensors shard...")
    embedding_matrix = load_embedding_matrix()
    print(f"  Embedding matrix shape: {embedding_matrix.shape}")

    # ── Analysis C ───────────────────────────────────────────────────────────
    c_result = analysis_c(
        ex1_responses, first_tokens,
        self_data, cross_data,
        embedding_matrix,
    )

    # ── Analysis D ───────────────────────────────────────────────────────────
    _ = analysis_d(ex3_responses, tokenizer)

    # ── Figures ──────────────────────────────────────────────────────────────
    figure1(a_result, tokenizer)
    figure2(c_result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_and_save_summary(a_result, b_result, c_result)


if __name__ == "__main__":
    main()
