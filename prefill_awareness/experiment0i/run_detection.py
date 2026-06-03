"""
Detection for Experiment 0i.

For every (source_model, evaluator, condition, task), the evaluator sees:

  [System]    You are a helpful assistant.
  [User]      <BigCodeBench task prompt>
  [Assistant] <solution code>
  [User]      <backdoor detection question>

All source × evaluator × condition combinations are run:
  2 sources × 2 evaluators × 2 conditions × n_tasks = 8 × n_tasks calls

Output: results_dir/detection_results.{json,csv}
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from .config import Experiment0iConfig
from .generate import load_solutions


def run_detection(
    config: Experiment0iConfig,
    cost_tracker,
    force: bool = False,
) -> list[dict]:
    out_json = config.results_dir / "detection_results.json"
    out_csv  = config.results_dir / "detection_results.csv"

    existing: list[dict] = []
    if out_json.exists() and not force:
        existing = json.loads(out_json.read_text())
    # Backward-compat: old records lack a probe field; treat them as backdoor probe.
    for r in existing:
        r.setdefault("probe", "backdoor")
    completed = {(r["run_id"], r["task_id"]) for r in existing}
    print(f"  {len(completed)} detection calls already done.")

    client = OpenAI(base_url=config.api_base, api_key=config.api_key)
    results = list(existing)

    # Pre-load all solutions
    solutions_cache: dict[tuple, list[dict]] = {}
    for source in config.source_models:
        for condition in ["honest", "attack"]:
            try:
                solutions_cache[(source, condition)] = load_solutions(source, condition, config)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")

    # Build work list: source × evaluator × condition × probe × task
    # Probes:
    #   "backdoor"    — honest framing, ask backdoor_prob
    #   "compliance"  — attack framing visible, ask compliance_prob
    # run_id encodes probe for compliance: "{source}__{evaluator}__{condition}__compliance"
    # Backdoor run_ids keep the old format for backward compatibility.
    work = []
    for source in config.source_models:
        for evaluator in config.evaluator_models:
            for condition in ["honest", "attack"]:
                for probe in ["backdoor", "authorship_attack"]:
                    run_id = (
                        f"{source}__{evaluator}__{condition}__authorship_attack"
                        if probe == "authorship_attack"
                        else f"{source}__{evaluator}__{condition}"
                    )
                    for sol in solutions_cache.get((source, condition), []):
                        if (run_id, sol["task_id"]) not in completed:
                            work.append((run_id, source, evaluator, condition, probe, sol))

    print(f"  {len(work)} detection calls remaining.")
    if not work:
        return results

    with tqdm(total=len(work), desc="Detection") as pbar:
        for run_id, source, evaluator, condition, probe, sol in work:
            if cost_tracker.exceeded():
                print("\n  Cost cap reached.")
                break

            evaluator_model_id = config.evaluator_models[evaluator]
            record = _run_one(run_id, source, evaluator, evaluator_model_id,
                              condition, probe, sol, config, client, cost_tracker)
            if record is not None:
                results.append(record)
                if len(results) % 50 == 0:
                    _save(results, out_json, out_csv)

            time.sleep(config.get_delay(evaluator_model_id))
            pbar.update(1)

    _save(results, out_json, out_csv)
    return results


def _run_one(
    run_id: str,
    source: str,
    evaluator: str,
    evaluator_model_id: str,
    condition: str,
    probe: str,
    sol: dict,
    config: Experiment0iConfig,
    client: OpenAI,
    cost_tracker,
) -> dict | None:
    if probe == "authorship_attack":
        system_prompt = config.attack_framing_system
        detection_q   = config.authorship_prompt
    else:
        system_prompt = config.honest_framing_system
        detection_q   = config.detection_prompt

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": sol["task_prompt"]},
        {"role": "assistant", "content": sol["code"]},
        {"role": "user",      "content": detection_q},
    ]

    def _fn():
        return client.chat.completions.create(
            model=evaluator_model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=config.max_tokens_detection,
        )

    resp = _retry(_fn, config)
    choices = getattr(resp, "choices", None) if resp else None
    raw = (choices[0].message.content or "") if choices else ""

    if resp and getattr(resp, "usage", None):
        u = resp.usage
        cost_tracker.add(evaluator_model_id, u.prompt_tokens, u.completion_tokens, config)
    else:
        cost_tracker.add_estimated(evaluator_model_id, messages, raw, config)

    score, reason = _parse(raw, probe)

    return {
        "run_id":            run_id,
        "source":            source,
        "evaluator":         evaluator,
        "condition":         condition,
        "probe":             probe,
        "task_id":           sol["task_id"],
        "score":             score,
        "reason":            reason,
        "main_task_success": sol.get("main_task_success"),
        "side_task_success": sol.get("side_task_success"),
        "raw_response":      raw,
    }


def _parse(text: str, probe: str = "backdoor") -> tuple[int | None, str]:
    key = "tamper_prob" if probe == "authorship_attack" else "backdoor_prob"
    raw = (text or "").strip()
    for attempt in [raw, *_extract_json_blocks(raw)]:
        try:
            data = json.loads(attempt)
            v = data.get(key)
            if v is not None:
                return max(0, min(100, int(v))), str(data.get("reason", ""))
        except Exception:
            pass
    kv = re.search(rf'"{key}"\s*[=:]\s*(\d+)', raw)
    if kv:
        return max(0, min(100, int(kv.group(1)))), ""
    for n in re.findall(r'\b(\d{1,3})\b', raw):
        v = int(n)
        if 0 <= v <= 100:
            return v, ""
    return None, ""


def _extract_json_blocks(text: str) -> list[str]:
    out = []
    for pat in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{[^{}]+\})"]:
        for m in re.finditer(pat, text, re.DOTALL):
            out.append(m.group(1))
    return out


def _retry(fn, config: Experiment0iConfig):
    for attempt in range(config.max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt < config.max_retries - 1:
                wait = config.retry_delay
                if "429" in str(e):
                    try:
                        wait = float(e.response.json()["error"]["metadata"].get(
                            "retry_after_seconds", config.retry_delay)) + 2
                    except Exception:
                        pass
                print(f"  attempt {attempt+1} failed, waiting {wait:.0f}s: {e}")
                time.sleep(wait)
            else:
                print(f"  all {config.max_retries} attempts failed, skipping.")
                return None


def _save(results: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(results, indent=2))
    pd.DataFrame(results).to_csv(csv_path, index=False)


class CostTracker:
    def __init__(self, cap: float, log_every: int = 50):
        self._total = 0.0
        self._calls = 0
        self.cap = cap
        self.log_every = log_every

    def add(self, model_id: str, in_tok: int, out_tok: int, config) -> None:
        cost = config.cost_per_call(model_id, in_tok, out_tok)
        self._total += cost
        self._calls += 1
        if self._calls % self.log_every == 0:
            print(f"  {self.report()}")

    def add_estimated(self, model_id: str, messages: list, response: str, config) -> None:
        in_tok = sum(len((m.get("content") or "")) for m in messages) // 4
        out_tok = len(response) // 4
        self.add(model_id, in_tok, out_tok, config)

    def exceeded(self) -> bool:
        return self._total >= self.cap

    def report(self) -> str:
        return f"Cost: ${self._total:.4f} / ${self.cap:.2f} cap ({self._calls} calls)"
