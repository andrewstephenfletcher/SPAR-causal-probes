import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .openrouter_client import OpenRouterChatClient, OpenRouterConfig
from .prompts import (
    LIARSBENCH_DECEPTION_TYPES,
    build_generator_system_prompt,
    build_judge_system_prompt,
    build_judge_user_prompt,
)


@dataclass
class ModelConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 512


@dataclass
class InputConfig:
    path: str
    prompt_field: str = "prompt"
    limit: int | None = None


@dataclass
class GenerationConfig:
    mode: str = "neutral"  # neutral | targeted_deception
    samples_per_prompt: int = 1
    deception_types: list[str] | None = None
    seed: int = 0


@dataclass
class OutputConfig:
    path: str
    error_log_path: str | None = None
    print_errors: bool = True


@dataclass
class PipelineConfig:
    openrouter: OpenRouterConfig
    generator: ModelConfig
    judge: ModelConfig
    input: InputConfig
    generation: GenerationConfig
    output: OutputConfig


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_config(raw: dict[str, Any]) -> PipelineConfig:
    generation = GenerationConfig(**raw["generation"])
    if generation.mode not in {"neutral", "targeted_deception"}:
        raise ValueError(
            "generation.mode must be one of: 'neutral', 'targeted_deception'"
        )

    if generation.deception_types is None:
        generation.deception_types = sorted(LIARSBENCH_DECEPTION_TYPES)

    unknown = [d for d in generation.deception_types if d not in LIARSBENCH_DECEPTION_TYPES]
    if unknown:
        raise ValueError(f"Unknown deception types in config: {unknown}")

    return PipelineConfig(
        openrouter=OpenRouterConfig(**raw["openrouter"]),
        generator=ModelConfig(**raw["generator"]),
        judge=ModelConfig(**raw["judge"]),
        input=InputConfig(**raw["input"]),
        generation=generation,
        output=OutputConfig(**raw["output"]),
    )


def _load_prompts(input_cfg: InputConfig) -> list[str]:
    path = Path(input_cfg.path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    prompts: list[str] = []
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            if suffix == ".json":
                payload = json.load(f)
                if not isinstance(payload, list):
                    raise ValueError("JSON input must be a list of objects")
                rows = payload
            else:
                rows = [json.loads(line) for line in f if line.strip()]

        for row in rows:
            p = str(row.get(input_cfg.prompt_field, "")).strip()
            if p:
                prompts.append(p)

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = str(row.get(input_cfg.prompt_field, "")).strip()
                if p:
                    prompts.append(p)

    elif suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported input format. Use .jsonl, .json, .csv, or .txt")

    if input_cfg.limit is not None:
        prompts = prompts[: input_cfg.limit]

    if not prompts:
        raise ValueError("No prompts loaded. Check input path and prompt_field.")

    return prompts


def _extract_json_block(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError(f"Judge output is not valid JSON: {text}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pipeline(config: PipelineConfig) -> Path:
    rng = random.Random(config.generation.seed)
    prompts = _load_prompts(config.input)
    client = OpenRouterChatClient(config.openrouter)

    output_path = Path(config.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = Path(config.output.error_log_path) if config.output.error_log_path else None

    judge_system_prompt = build_judge_system_prompt()

    with output_path.open("w", encoding="utf-8") as out:
        for prompt in tqdm(prompts, desc="Generating"):
            for sample_idx in range(config.generation.samples_per_prompt):
                target_type = None
                if config.generation.mode == "targeted_deception":
                    target_type = rng.choice(config.generation.deception_types)

                gen_response = client.chat_completion(
                    model=config.generator.model,
                    system_prompt=build_generator_system_prompt(
                        config.generation.mode,
                        target_type,
                    ),
                    user_prompt=prompt,
                    temperature=config.generator.temperature,
                    max_tokens=config.generator.max_tokens,
                )
                answer = gen_response["text"]

                judge_response = client.chat_completion(
                    model=config.judge.model,
                    system_prompt=judge_system_prompt,
                    user_prompt=build_judge_user_prompt(prompt, answer, target_type),
                    temperature=config.judge.temperature,
                    max_tokens=config.judge.max_tokens,
                )
                judge_raw = judge_response["text"]
                judge_json: dict[str, Any]
                parse_error: str | None = None

                try:
                    judge_json = _extract_json_block(judge_raw)
                except Exception as first_err:  # noqa: BLE001
                    retry_user_prompt = (
                        build_judge_user_prompt(prompt, answer, target_type)
                        + "\n\nIMPORTANT: Return ONLY valid JSON with no extra text."
                    )
                    retry_response = client.chat_completion(
                        model=config.judge.model,
                        system_prompt=judge_system_prompt,
                        user_prompt=retry_user_prompt,
                        temperature=config.judge.temperature,
                        max_tokens=config.judge.max_tokens,
                    )
                    judge_raw_retry = retry_response["text"]
                    try:
                        judge_json = _extract_json_block(judge_raw_retry)
                        judge_response = retry_response
                    except Exception as second_err:  # noqa: BLE001
                        parse_error = (
                            f"Judge JSON parse failed after retry. "
                            f"First error: {first_err}; Second error: {second_err}"
                        )
                        judge_json = {
                            "is_deceptive": None,
                            "detected_type": "parse_error",
                            "matches_target": None,
                            "confidence": 0.0,
                            "reason": "Judge output was not valid JSON after retry.",
                        }
                        error_row = {
                            "error_type": "judge_json_parse_failure",
                            "message": parse_error,
                            "prompt": prompt,
                            "sample_idx": sample_idx,
                            "generation_mode": config.generation.mode,
                            "target_deception_type": target_type,
                            "generated_answer": answer,
                            "judge_raw_first": judge_raw,
                            "judge_raw_retry": judge_raw_retry,
                            "judge_model": config.judge.model,
                        }
                        if error_log_path:
                            _append_jsonl(error_log_path, error_row)
                        if config.output.print_errors:
                            print(parse_error, file=sys.stderr)

                row = {
                    "prompt": prompt,
                    "sample_idx": sample_idx,
                    "generation_mode": config.generation.mode,
                    "target_deception_type": target_type,
                    "generated_answer": answer,
                    "judge": judge_json,
                    "generator_model": config.generator.model,
                    "judge_model": config.judge.model,
                    "generator_usage": gen_response.get("usage", {}),
                    "judge_usage": judge_response.get("usage", {}),
                    "generator_request_id": gen_response.get("id"),
                    "judge_request_id": judge_response.get("id"),
                    "judge_parse_error": parse_error,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LiarsBench-style synthetic deception data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.example.json")),
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = _parse_config(_load_json(Path(args.config)))
    out = run_pipeline(cfg)
    print(f"Saved generated dataset to: {out}")


if __name__ == "__main__":
    main()
