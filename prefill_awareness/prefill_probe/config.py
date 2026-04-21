from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Experiment2Config:
    # Relative token positions within the assistant response (0-indexed)
    positions: List[int] = field(default_factory=lambda: [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 31
    ])

    # Subset of Llama's 32 layers to probe at each position
    layers: List[int] = field(default_factory=lambda: [0, 8, 16, 24, 30])

    # Responses shorter than this are excluded (we can't probe beyond their length)
    min_length_tokens: int = 32

    # Source of Experiment 1 generation artefacts (responses.json, prompts.json)
    ex1_generations_dir: Path = Path("outputs/experiment1/generations")

    # Experiment 2 output paths
    activations_dir_ex2: Path = Path("outputs/experiment2/activations")
    results_dir_ex2: Path = Path("outputs/experiment2/results")

    def __post_init__(self):
        for d in [self.activations_dir_ex2, self.results_dir_ex2]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # Models
    target_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    source_model_id: str = "google/gemma-2-9b-it"

    # Generation
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 42

    # Data
    n_prompts: int = 300
    dataset_name: str = "tatsu-lab/alpaca"
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    min_response_tokens: int = 20
    max_response_tokens: int = 256

    # Activation extraction
    # Llama-3.1-8B: 32 layers, hidden dim 4096
    extract_layers: List[int] = field(default_factory=lambda: list(range(32)))
    activation_dtype: str = "float16"

    # Probing
    probe_regularisation_grid: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )

    # Paths
    output_dir: Path = Path("outputs/experiment1")
    generations_dir: Path = Path("outputs/experiment1/generations")
    activations_dir: Path = Path("outputs/experiment1/activations")
    results_dir: Path = Path("outputs/experiment1/results")

    def __post_init__(self):
        for d in [self.output_dir, self.generations_dir,
                  self.activations_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment3Config:
    # Target model (same as Experiments 1 & 2)
    target_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Source models for cross-model conditions (Llama 70B excluded for single-GPU / Mac)
    gemma_model_id: str = "google/gemma-2-9b-it"
    mistral_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # Datasets
    datasets: List[str] = field(default_factory=lambda: ["alpaca", "oasst1", "mmlu"])
    n_prompts_per_dataset: int = 300

    # Altered-self condition: same model, different generation parameters
    altered_self_temperature: float = 1.2
    altered_self_system_prompt: str = (
        "You are a terse, no-nonsense assistant. Avoid filler words and "
        "unnecessary elaboration. Be direct."
    )

    # Style-imitation: Gemma mimics Llama's style via few-shot examples
    style_imitation_n_shot: int = 3

    # Standard generation parameters (match Experiment 1)
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 42

    # Extraction: layer 30, last content token (best from Experiments 1 & 2)
    layer: int = 30

    # Filtering: discard prompts where any condition is shorter than this
    min_response_tokens: int = 20

    # Probing
    probe_regularisation_grid: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )

    # Perplexity-controlled analysis
    perplexity_match_threshold: float = 0.5   # nats; pairs within this are "matched"
    outlier_perplexity_threshold: float = 10.0  # responses above this are "outliers"

    # Paths
    ex1_generations_dir: Path = Path("outputs/experiment1/generations")
    generations_dir_ex3: Path = Path("outputs/experiment3/generations")
    activations_dir_ex3: Path = Path("outputs/experiment3/activations")
    results_dir_ex3: Path = Path("outputs/experiment3/results")

    def __post_init__(self):
        for d in [self.generations_dir_ex3, self.activations_dir_ex3, self.results_dir_ex3]:
            d.mkdir(parents=True, exist_ok=True)
