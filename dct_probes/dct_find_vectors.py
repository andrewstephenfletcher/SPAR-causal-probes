"""
andrewstephenfletcher March 2026

Finds and savessteering vectors using Deep Causal Transcoding (DCT).


"""

import gc
import torch
import dct
import json
from pathlib import Path
from torch import vmap
from tqdm import tqdm

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Drop any variables from a previous run
for _var in ["model", "tokenizer", "sliced_model", "delta_acts_single", "delta_acts",
             "steering_calibrator", "exp_dct", "X", "Y", "U", "V", "hidden_states"]:
    if _var in dir():
        del globals()[_var]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved()   / 1e9:.2f} GB")

gc.collect()

# Detect best available device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")
torch.set_default_dtype(torch.float32)

def load_dct_params():
    with open("dct_params.json", "r") as f:
        params = json.load(f)
    return params

def set_dct_params(params):
    global MODEL_NAME, TOKENIZER_NAME, INPUT_SCALE, NUM_SAMPLES, FORWARD_BATCH_SIZE, \
           BACKWARD_BATCH_SIZE, MAX_SEQ_LEN, CALIBRATION_SAMPLE_SIZE, \
           CALIBRATION_PROMPT_SAMPLE_SIZE, DIM_OUTPUT_PROJECTION, NUM_ITERS, \
           NUM_FACTORS, FACTOR_BATCH_SIZE, SOURCE_LAYER_IDX, TARGET_LAYER_IDX, SYSTEM_PROMPT, \
           TOKEN_IDXS
    MODEL_NAME = params["MODEL_NAME"]
    TOKENIZER_NAME = params["TOKENIZER_NAME"]
    INPUT_SCALE = params["INPUT_SCALE"]
    NUM_SAMPLES = params["NUM_SAMPLES"]
    FORWARD_BATCH_SIZE = params["FORWARD_BATCH_SIZE"]
    BACKWARD_BATCH_SIZE = params["BACKWARD_BATCH_SIZE"]
    MAX_SEQ_LEN = params["MAX_SEQ_LENGTH"]
    CALIBRATION_SAMPLE_SIZE = params["CALIBRATION_SAMPLE_SIZE"]
    CALIBRATION_PROMPT_SAMPLE_SIZE = params["CALIBRATION_PROMPT_SAMPLE_SIZE"]
    DIM_OUTPUT_PROJECTION = params["DIM_OUTPUT_PROJECTION"]
    NUM_ITERS = params["NUM_ITERS"]
    NUM_FACTORS = params["NUM_FACTORS"]
    FACTOR_BATCH_SIZE = params["FACTOR_BATCH_SIZE"]
    SOURCE_LAYER_IDX = params["SOURCE_LAYER_IDX"]
    TARGET_LAYER_IDX = params["TARGET_LAYER_IDX"]
    SYSTEM_PROMPT = params["SYSTEM_PROMPT"]
    TOKEN_IDXS = slice(params["TOKEN_IDXS_START"], params["TOKEN_IDXS_STOP"])

def load_model(MODEL_NAME, TOKENIZER_NAME):

    tokenizer = AutoTokenizer.from_pretrained(
                                            TOKENIZER_NAME,
                                            trust_remote_code=True,
                                            padding_side="left",
                                            truncation_side="left"
                                            )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        _attn_implementation="eager",
    )

    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Num layers:   {model.config.num_hidden_layers}")
    print(f"d_model:      {model.config.hidden_size}")
    print(f"Device:       {next(model.parameters()).device}")

    return model, tokenizer

def set_chat_template(tokenizer, SYSTEM_PROMPT, instructions):

    chat_init = ([{'content': SYSTEM_PROMPT, 'role': 'system'}]
             if SYSTEM_PROMPT is not None else [])
    chats = [chat_init + [{'content': c, 'role': 'user'}]
            for c in instructions[:NUM_SAMPLES]]
    EXAMPLES = [tokenizer.apply_chat_template(
        chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
        for chat in chats]

    test_chats = [chat_init + [{'content': c, 'role': 'user'}]
                for c in instructions[-32:]]
    TEST_EXAMPLES = [tokenizer.apply_chat_template(
        chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True)
        for chat in test_chats]

    print(f"Training examples: {len(EXAMPLES)}")
    print(f"Test examples:     {len(TEST_EXAMPLES)}")

    return EXAMPLES, TEST_EXAMPLES

def sliced_sanity_check(model, tokenizer):

    model_inputs = tokenizer(
        ["Is Paris the capital of France?"],
        return_tensors="pt", truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        hidden_states = model(
            model_inputs["input_ids"], output_hidden_states=True
        ).hidden_states

    sliced_test = dct.SlicedModel(
        model, start_layer=SOURCE_LAYER_IDX, end_layer=TARGET_LAYER_IDX, layers_name="model.layers"
    )
    with torch.no_grad():
        out = sliced_test(hidden_states[SOURCE_LAYER_IDX])
        assert torch.allclose(out, hidden_states[TARGET_LAYER_IDX], atol=1e-2), \
            f"SlicedModel mismatch! max_diff={( out - hidden_states[TARGET_LAYER_IDX]).abs().max()}"
    print("SlicedModel sanity check passed.")

def load_got_statements(dataset: str = "cities", label: int = 1) -> list[str]:
    path = Path(f"data/got_datasets/{dataset}.csv")
    df = pd.read_csv(path)
    return df[df["label"] == label]["statement"].tolist()

def load_calibration_texts() -> list[str]:
    path = Path("data/calibration_texts.jsonl")
    with open(path, "r") as f:
        return [json.loads(line)["text"] for line in f]

def create_sliced_model(model):
    return dct.SlicedModel(
        model,
        start_layer=SOURCE_LAYER_IDX,
        end_layer=TARGET_LAYER_IDX,
        layers_name="model.layers"
    )

def construct_unsteered_activations(model, tokenizer, EXAMPLES, sliced_model):

    d_model = model.config.hidden_size

    X = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu", dtype=torch.float32)
    Y = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu", dtype=torch.float32)

    for t in tqdm(range(0, NUM_SAMPLES, FORWARD_BATCH_SIZE)):
        with torch.no_grad():
            model_inputs = tokenizer(
                EXAMPLES[t:t + FORWARD_BATCH_SIZE],
                return_tensors="pt", truncation=True,
                padding="max_length", max_length=MAX_SEQ_LEN
            ).to(DEVICE)
            hidden_states = model(
                model_inputs["input_ids"], output_hidden_states=True
            ).hidden_states
            h_source        = hidden_states[SOURCE_LAYER_IDX]
            unsteered_target = sliced_model(h_source)

            X[t:t + FORWARD_BATCH_SIZE] = h_source.cpu()
            Y[t:t + FORWARD_BATCH_SIZE] = unsteered_target.cpu()

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    return X, Y

def find_delta_acts(sliced_model):

    delta_acts_single = dct.DeltaActivations(
        sliced_model, target_position_indices=TOKEN_IDXS
    )
    delta_acts = vmap(
        delta_acts_single, in_dims=(1, None, None), out_dims=2,
        chunk_size=FACTOR_BATCH_SIZE
    )

    return delta_acts_single, delta_acts

def calibrate_steering(delta_acts_single, X, Y):

    input_scale = INPUT_SCALE
    if input_scale is None:
        steering_calibrator = dct.SteeringCalibrator(target_ratio=0.5)
        input_scale = steering_calibrator.calibrate(
            delta_acts_single,
            X, Y,
            factor_batch_size=FACTOR_BATCH_SIZE,
            calibration_sample_size=CALIBRATION_SAMPLE_SIZE,
        )
    print(f"INPUT_SCALE: {input_scale}")

    return input_scale

def compute_exp_dct(delta_acts_single, X, Y, input_scale):

    exp_dct = dct.ExponentialDCT(num_factors=NUM_FACTORS)
    U, V = exp_dct.fit(
        delta_acts_single,
        X, Y,
        batch_size=BACKWARD_BATCH_SIZE,
        factor_batch_size=FACTOR_BATCH_SIZE,
        init="jacobian",
        d_proj=DIM_OUTPUT_PROJECTION,
        input_scale=input_scale,
        max_iters=NUM_ITERS,
        beta=1.0,
    )
    print(f"U shape: {U.shape}, V shape: {V.shape}")

    return exp_dct, U, V

def rank_vectors(exp_dct, model, tokenizer, delta_acts_single, X, Y) -> tuple[torch.Tensor, torch.Tensor]:
    yes_token = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_token  = tokenizer.encode(" No",  add_special_tokens=False)[0]
    with torch.no_grad():
        target_vec = (model.lm_head.weight.data[no_token] - model.lm_head.weight.data[yes_token]).float()

    scores, indices = exp_dct.rank(
        delta_acts_single, X, Y,
        target_vec=target_vec,
        batch_size=FORWARD_BATCH_SIZE,
        factor_batch_size=FACTOR_BATCH_SIZE,
    )
    print(f"Top-5 factor indices: {indices[:5].tolist()}")
    print(f"Top-5 scores:         {scores[:5].tolist()}")
    return scores, indices

def save_vectors(U, V, exp_dct, params, scores=None, indices=None, output_dir="vectors"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "U": U,  # [d_model, num_factors] — steering directions at source layer
        "V": V,  # [d_model, num_factors] — output directions at target layer
    }
    if scores is not None and indices is not None:
        payload["scores"]  = scores
        payload["indices"] = indices

    torch.save(payload, output_dir / "dct_vectors.pt")

    with open(output_dir / "dct_run_config.json", "w") as f:
        json.dump(params, f, indent=2)


def main():

    params = load_dct_params()
    set_dct_params(params)

    vectors_dir = Path("vectors")
    vectors_path = vectors_dir / "dct_vectors.pt"
    activations_cache = vectors_dir / "activations_cache.pt"

    if vectors_path.exists():
        print(f"Vectors already exist at {vectors_path}, skipping.")
        return

    model, tokenizer = load_model(MODEL_NAME, TOKENIZER_NAME)

    if NUM_SAMPLES == 1:
        instructions = ["Is Paris the capital of France?"]
    else:
        instructions = load_got_statements(dataset="cities", label=1)

    EXAMPLES, TEST_EXAMPLES = set_chat_template(tokenizer, SYSTEM_PROMPT, instructions)

    sliced_sanity_check(model, tokenizer)

    sliced_model = create_sliced_model(model)

    if activations_cache.exists():
        print(f"Loading cached activations from {activations_cache}")
        cache = torch.load(activations_cache, weights_only=True)
        X, Y = cache["X"], cache["Y"]
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    else:
        X, Y = construct_unsteered_activations(model, tokenizer, EXAMPLES, sliced_model)
        vectors_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "Y": Y}, activations_cache)
        print(f"Saved activations cache to {activations_cache}")

    delta_acts_single, delta_acts = find_delta_acts(sliced_model)

    INPUT_SCALE = calibrate_steering(delta_acts_single, X, Y)

    exp_dct, U, V = compute_exp_dct(delta_acts_single, X, Y, input_scale=INPUT_SCALE)

    if NUM_SAMPLES == 1:
        scores, indices = rank_vectors(exp_dct, model, tokenizer, delta_acts_single, X, Y)
        save_vectors(U, V, exp_dct, params, scores=scores, indices=indices)
    else:
        save_vectors(U, V, exp_dct, params)

    activations_cache.unlink(missing_ok=True)

if __name__ == "__main__":
    main()





