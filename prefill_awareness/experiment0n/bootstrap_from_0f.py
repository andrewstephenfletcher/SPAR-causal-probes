"""
Bootstrap experiment0n from existing experiment0f results.

Copies:
  - generations/responses_{model}.json for the 6 frontier sources + opus_45 (evaluator)
  - results/detection_results.json filtered to BCB + GPQA rows and frontier sources only

After running this, start experiment0n from the OASST1 generation step:
    uv run python -m experiment0n.run_all --from-step generate --datasets oasst1
    uv run python -m experiment0n.run_all --from-step detection --datasets oasst1
    uv run python -m experiment0n.run_all --from-step analysis
"""

import json
import shutil
from pathlib import Path

SRC_GEN  = Path("outputs/experiment0f/generations")
SRC_DET  = Path("outputs/experiment0f/results/detection_results.json")
DST_GEN  = Path("outputs/experiment0n/generations")
DST_DET  = Path("outputs/experiment0n/results/detection_results.json")
DST_CSV  = Path("outputs/experiment0n/results/detection_results.csv")

MODELS_TO_COPY = [
    "sonnet_45", "opus_45", "gpt_4o_mini", "gpt_5",
    "gemini_flash", "gemini_pro",
]
KEEP_SOURCES  = set(MODELS_TO_COPY) | {"organic"}
KEEP_DATASETS = {"bigcodebench", "gpqa"}


def main() -> None:
    DST_GEN.mkdir(parents=True, exist_ok=True)
    DST_DET.parent.mkdir(parents=True, exist_ok=True)

    # --- Generation files ---
    copied, skipped = 0, 0
    for model in MODELS_TO_COPY:
        src = SRC_GEN / f"responses_{model}.json"
        dst = DST_GEN / f"responses_{model}.json"
        if not src.exists():
            print(f"  [MISSING] {src}")
            skipped += 1
            continue
        if dst.exists():
            print(f"  [EXISTS]  {dst.name} — skipping")
            skipped += 1
            continue
        shutil.copy2(src, dst)
        data = json.loads(dst.read_text())
        print(f"  [COPIED]  {dst.name} ({len(data)} responses)")
        copied += 1
    print(f"\n  Generation files: {copied} copied, {skipped} skipped.")

    # --- Detection results (BCB + GPQA only, frontier sources only) ---
    if not SRC_DET.exists():
        print(f"\n  [MISSING] {SRC_DET} — skipping detection copy.")
        return

    if DST_DET.exists():
        existing = json.loads(DST_DET.read_text())
        print(f"\n  [EXISTS]  detection_results.json ({len(existing)} rows) — not overwriting.")
        return

    all_results = json.loads(SRC_DET.read_text())
    filtered = [
        r for r in all_results
        if r.get("dataset") in KEEP_DATASETS
        and r.get("source") in KEEP_SOURCES
    ]
    DST_DET.write_text(json.dumps(filtered, indent=2))
    print(f"\n  Detection results: {len(all_results)} → {len(filtered)} rows "
          f"(BCB+GPQA, frontier sources only).")
    print(f"  Saved: {DST_DET}")

    try:
        import pandas as pd
        pd.DataFrame(filtered).to_csv(DST_CSV, index=False)
        print(f"  Saved: {DST_CSV}")
    except ImportError:
        pass

    print("\nDone. Now run:")
    print("  uv run python -m experiment0n.run_all --from-step generate --datasets oasst1")
    print("  uv run python -m experiment0n.run_all --from-step detection --datasets oasst1")
    print("  uv run python -m experiment0n.run_all --from-step analysis")


if __name__ == "__main__":
    main()
