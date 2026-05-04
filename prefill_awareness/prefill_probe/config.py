import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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


@dataclass
class Experiment4Config:
    # Target models for this experiment
    llama70b_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    gemma31b_model_id: str = "google/gemma-4-31b-it"

    # Source models whose responses are used as cross-model prefills
    llama8b_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    gemma9b_model_id: str = "google/gemma-2-9b-it"

    # Generation parameters (match Experiment 1)
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 256
    seed: int = 42

    # Data
    n_prompts: int = 300
    min_response_tokens: int = 20

    # Probe regularisation grid
    probe_regularisation_grid: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )

    # Incremental checkpoint: save activations to disk every N prompts
    checkpoint_interval: int = 20

    # Exp 1 artefact paths (activations + responses reused for scaling comparison)
    ex1_generations_dir: Path = Path("outputs/experiment1/generations")
    ex1_activations_dir: Path = Path("outputs/experiment1/activations")
    ex1_results_dir: Path = Path("outputs/experiment1/results")

    # Experiment 4 output paths
    output_dir_ex4: Path = Path("outputs/experiment4")
    generations_dir_ex4: Path = Path("outputs/experiment4/generations")
    activations_dir_llama70b: Path = Path("outputs/experiment4/activations/llama70b")
    activations_dir_gemma31b: Path = Path("outputs/experiment4/activations/gemma31b")
    results_dir_ex4: Path = Path("outputs/experiment4/results")

    def __post_init__(self):
        for d in [
            self.output_dir_ex4,
            self.generations_dir_ex4,
            self.activations_dir_llama70b,
            self.activations_dir_gemma31b,
            self.results_dir_ex4,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment5Config:
    # Target model for steering
    llama70b_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    gemma9b_model_id: str = "google/gemma-2-9b-it"
    llama8b_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Steering configuration
    # Layer 24 ≈ 30% depth — strong signal (AUROC ~0.85), 56 layers of headroom downstream
    steering_layer: int = 24
    steering_condition: str = "cross_gemma9b"

    # Layer sweep for steering layer selection
    sweep_layers: List[int] = field(
        default_factory=lambda: [16, 24, 32, 40, 48, 56, 64]
    )

    # Alpha values (populated from calibration JSON at runtime)
    alpha_conservative: Optional[float] = None
    alpha_moderate: Optional[float] = None
    alpha_aggressive: Optional[float] = None

    # Generation
    temperature: float = 0.0        # deterministic; steering is the variable
    max_new_tokens_5a: int = 30     # attribution answers are short
    max_new_tokens_5b: int = 200    # evaluation responses are longer
    seed: int = 42

    # Data
    n_prompts_5a: int = 100         # capped by available test+val prompts
    n_prompts_5b: int = 50          # capped by available prompts
    n_prompts_sweep: int = 10       # quick layer sweep
    n_prompts_calibrate: int = 20   # for residual norm estimation

    # Probe regularisation (retraining to extract probe direction)
    probe_regularisation_grid: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )

    # Checkpointing
    checkpoint_interval: int = 10

    # LLM judge (via OpenRouter)
    judge_model_id: str = "meta-llama/llama-3.3-70b-instruct"
    judge_temperature: float = 0.0

    # Input paths (from experiment 4)
    ex4_generations_dir: Path = Path("outputs/experiment4/generations")
    ex4_activations_dir_llama70b: Path = Path("outputs/experiment4/activations/llama70b")
    ex4_results_dir: Path = Path("outputs/experiment4/results")

    # Output paths
    output_dir_ex5: Path = Path("outputs/experiment5")
    results_dir_ex5: Path = Path("outputs/experiment5/results")
    generations_dir_ex5: Path = Path("outputs/experiment5/generations")

    def __post_init__(self):
        for d in [self.output_dir_ex5, self.results_dir_ex5, self.generations_dir_ex5]:
            d.mkdir(parents=True, exist_ok=True)

    def load_alphas_from_calibration(self) -> bool:
        """Populate alpha fields from calibration JSON. Returns True if found."""
        calib_path = self.results_dir_ex5 / "alpha_calibration.json"
        if not calib_path.exists():
            return False
        with open(calib_path) as f:
            calib = json.load(f)
        self.alpha_conservative = calib.get("alpha_conservative")
        self.alpha_moderate = calib.get("alpha_moderate")
        self.alpha_aggressive = calib.get("alpha_aggressive")
        return True

    def get_alpha_values(self) -> List[float]:
        """[alpha_pos, 0, alpha_neg] for the 3-condition steering sweep."""
        if self.alpha_moderate is None:
            raise ValueError("Alpha not calibrated. Run --from-step calibrate first.")
        return [self.alpha_moderate, 0.0, -self.alpha_moderate]


@dataclass
class Experiment6Config:
    # Model (same as Experiment 5)
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct"

    # Steering layers to test; denser sampling at 60-70% depth (layers 48-60 for 80-layer Llama)
    steering_layers: List[int] = field(default_factory=lambda: [
        16, 24, 32, 40, 48, 52, 56, 60
    ])

    # Alpha fractions: each alpha = fraction × (layer_norm / 100)
    alpha_fractions: List[float] = field(default_factory=lambda: [
        0.25, 0.5, 0.75, 1.0, 1.5
    ])

    # Analysis D: additional random-vector magnitudes to test
    alpha_fractions_d: List[float] = field(default_factory=lambda: [
        1.0, 1.5, 2.0, 2.5
    ])

    # Number of random vectors for Analysis A
    n_random_vectors: int = 10

    # Random seed for vector generation
    random_seed: int = 42

    # Number of prompts for steering evaluation
    n_prompts: int = 100

    # Number of prompts used to estimate per-layer residual norms
    n_prompts_norm: int = 20

    # Generation
    max_new_tokens: int = 30
    seed: int = 42

    # Probe condition used to train probe at each layer
    probe_condition: str = "cross_gemma9b"

    # Probe regularisation search grid
    probe_regularisation_grid: List[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )

    # Checkpoint: save partial results every N generations
    checkpoint_interval: int = 10

    # Input paths (from Experiment 4)
    ex4_generations_dir: Path = Path("outputs/experiment4/generations")
    ex4_activations_dir_llama70b: Path = Path("outputs/experiment4/activations/llama70b")

    # Experiment 5 results (for loading the moderate alpha used in Analysis A)
    ex5_results_dir: Path = Path("outputs/experiment5/results")

    # Output paths
    output_dir_ex6: Path = Path("outputs/experiment6")
    results_dir_ex6: Path = Path("outputs/experiment6/results")
    generations_dir_ex6: Path = Path("outputs/experiment6/generations")

    def __post_init__(self):
        for d in [self.output_dir_ex6, self.results_dir_ex6, self.generations_dir_ex6]:
            d.mkdir(parents=True, exist_ok=True)

    def load_exp5_alpha(self) -> Optional[float]:
        """Return the Experiment 5 moderate alpha, or None if not found."""
        calib_path = self.ex5_results_dir / "alpha_calibration.json"
        if not calib_path.exists():
            return None
        with open(calib_path) as f:
            return json.load(f).get("alpha_moderate")
