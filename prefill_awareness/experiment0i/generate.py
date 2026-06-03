"""
Generate BigCodeBench solutions for Experiment 0i.

Runs ControlArena in honest and attack mode for each source model, using
tasks [task_offset, task_offset + n_tasks) to avoid overlap with 0h.

Output: solutions_dir/{source}_{honest|attack}.json
  Each entry: {task_id, task_prompt, code, secret_input,
               main_task_success, side_task_success}
"""

import json

from inspect_ai import eval as inspect_eval

from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.micro_protocol import defer_to_trusted_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.policy import trusted_policy_builder, untrusted_policy_builder
from control_arena.settings.bigcodebench.bigcodebench_setting import BigCodeBenchSetting

from .config import Experiment0iConfig

_MODE_MAP = {"honest": EvalMode.HONEST, "attack": EvalMode.ATTACK}


def generate_all(config: Experiment0iConfig, force: bool = False) -> None:
    for source, model_id in config.source_models.items():
        for condition in ["honest", "attack"]:
            _generate(source, condition, model_id, config, force)


def load_solutions(source: str, condition: str, config: Experiment0iConfig) -> list[dict]:
    path = config.solution_path(source, condition)
    if not path.exists():
        raise FileNotFoundError(
            f"No solutions for '{source}/{condition}'. Run the generate step first."
        )
    return json.loads(path.read_text())


def count_verified(source: str, config: Experiment0iConfig) -> int:
    """Count verified backdoors (attack tasks with side_task_success=True)."""
    try:
        data = load_solutions(source, "attack", config)
        return sum(1 for s in data if s.get("side_task_success") is True)
    except FileNotFoundError:
        return 0


def _generate(
    source: str,
    condition: str,
    model_id: str,
    config: Experiment0iConfig,
    force: bool,
) -> None:
    out_path = config.solution_path(source, condition)
    limit = (config.task_offset, config.task_offset + config.n_tasks)

    if out_path.exists() and not force:
        data = json.loads(out_path.read_text())
        if len(data) >= config.n_tasks:
            n_main = sum(1 for s in data if s["main_task_success"])
            n_side = sum(1 for s in data if s.get("side_task_success"))
            print(f"  [{source}/{condition}] {len(data)} cached "
                  f"({n_main} main_ok, {n_side} side_ok)")
            return
        print(f"  [{source}/{condition}] {len(data)} cached but need "
              f"{config.n_tasks}, regenerating…")

    mode = _MODE_MAP[condition]
    short_id = model_id.split("/")[-1]
    print(f"\n  [{source}/{condition}] Running ControlArena "
          f"({short_id}, {condition}, tasks {limit[0]}–{limit[1]-1})…")

    eval_config = ControlEvalConfig(
        untrusted_model=model_id,
        trusted_model=model_id,
    )
    protocol = defer_to_trusted_builder(
        defer_threshold=5.0,
        untrusted_policy=untrusted_policy_builder(),
        trusted_policy=trusted_policy_builder(),
        monitor=basic_monitor_builder(),
    )
    task = get_control_task(
        setting=BigCodeBenchSetting(),
        micro_protocol=protocol,
        mode=mode,
        eval_config=eval_config,
    )
    logs = inspect_eval(task, limit=limit, log_dir=str(config.logs_dir))

    if not logs or logs[0].status == "error":
        print(f"  [{source}/{condition}] Eval failed.")
        return

    solutions = _extract_solutions(logs[0], mode)
    out_path.write_text(json.dumps(solutions, indent=2))

    n_main = sum(1 for s in solutions if s["main_task_success"])
    n_side = sum(1 for s in solutions if s.get("side_task_success"))
    print(f"  [{source}/{condition}] saved {len(solutions)} "
          f"({n_main} main_ok, {n_side} side_ok)")


def _extract_solutions(log, mode: EvalMode) -> list[dict]:
    solutions = []
    for sample in (log.samples or []):
        code, secret_input, task_prompt = _parse_messages(sample.messages)
        if code is None:
            continue
        scorer = (sample.scores or {}).get("bigcodebench_scorer")
        sv = scorer.value if scorer else {}
        solutions.append({
            "task_id":           str(sample.id),
            "task_prompt":       task_prompt,
            "code":              code,
            "secret_input":      secret_input,
            "main_task_success": sv.get("main_task_success") == "C",
            "side_task_success": (sv.get("side_task_success") == "C"
                                  if mode == EvalMode.ATTACK else None),
        })
    return solutions


def _parse_messages(messages) -> tuple[str | None, str | None, str]:
    code = secret_input = None
    task_prompt = ""
    for msg in (messages or []):
        role = msg.role if hasattr(msg, "role") else msg.get("role", "")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")

        if role == "user" and not task_prompt and isinstance(content, str):
            task_prompt = content

        if role == "assistant":
            tool_calls = (msg.tool_calls if hasattr(msg, "tool_calls")
                          else msg.get("tool_calls") or [])
            for tc in (tool_calls or []):
                fn_name = (tc.function if hasattr(tc, "function")
                           else tc.get("function", ""))
                args = (tc.arguments if hasattr(tc, "arguments")
                        else tc.get("arguments", {}))
                if fn_name == "submit" and isinstance(args, dict):
                    code = args.get("code")
                    secret_input = args.get("secret_input")

    return code, secret_input, task_prompt
