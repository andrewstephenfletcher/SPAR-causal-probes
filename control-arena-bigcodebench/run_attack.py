from inspect_ai import eval
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.micro_protocol import defer_to_trusted_builder
from control_arena.monitor import basic_monitor_builder
from control_arena.policy import untrusted_policy_builder, trusted_policy_builder
from control_arena.settings.bigcodebench.bigcodebench_setting import BigCodeBenchSetting

eval_config = ControlEvalConfig(
    untrusted_model="openrouter/openai/gpt-4o-mini",
    trusted_model="openrouter/openai/gpt-4o-mini",
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
    mode=EvalMode.ATTACK,
    eval_config=eval_config,
)

eval(task, limit=5)
