# RunPod Setup Guide — SPAR Causal Probes

A step-by-step guide to get up and running with LLM inference on RunPod for causal probing experiments.

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with credits loaded
- An SSH key pair (see [Generating Your SSH Public Key](https://git-scm.com/book/en/v2/Git-on-the-Server-Generating-Your-SSH-Public-Key))
- [VS Code](https://code.visualstudio.com/) installed locally with the **Remote - SSH** extension

## 1. Generate and Register Your SSH Key

If you don't already have an SSH key, generate one:

```bash
ssh-keygen -t ed25519
```

When prompted for a file location, you can accept the default (`~/.ssh/id_ed25519`) or choose a custom path like `~/.ssh/id_ed25519_runpod`.

Copy your **public** key to the clipboard:

```bash
# macOS
pbcopy < ~/.ssh/id_ed25519_runpod.pub

# Linux
xclip -selection clipboard < ~/.ssh/id_ed25519_runpod.pub

# Windows (WSL / PowerShell)
clip < ~/.ssh/id_ed25519_runpod.pub
```

Then go to **RunPod → Settings → SSH Public Keys**, paste it in, and save.

> **Never share your private key** (the file without `.pub`).

## 2. Launch a GPU Pod

1. Go to **RunPod → Pods → New Pod**
2. Select a GPU — an **A40 (48GB)** or **RTX 3090 (24GB)** is plenty for 7B models
3. Choose the **RunPod PyTorch 2.4.0** template
4. Select an **EU region** for lower latency (if based in the UK)
5. Confirm your SSH key is attached
6. Deploy the pod

Once running, note the **SSH connection details** (IP and port) from the pod's dashboard — click **Connect** on your pod and look for the SSH command.

## 3. Connect via VS Code

Add the pod to your local SSH config (`~/.ssh/config`):

```
Host runpod
    HostName <POD_IP>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519_runpod
```

Replace `<POD_IP>` and `<PORT>` with the values from RunPod.

Then in VS Code:

1. Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`)
2. Select **Remote-SSH: Connect to Host**
3. Choose `runpod`

You're now editing and running code directly on the pod.

## 4. Clone the Repo and Install Dependencies

In the VS Code terminal (now connected to the pod):

```bash
cd /workspace

git clone https://github.com/andrewstephenfletcher/SPAR-causal-probes.git
cd SPAR-causal-probes
```

Install `uv` and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
```

This will install all dependencies (including `transformers`, `accelerate`, and `torch`) from the lockfile.

## 5. Run Inference

Open or create a notebook in VS Code (install the Jupyter extension if prompted), then run:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen1.5-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI safety and why does it matter?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Select the Python kernel when prompted. The first run will download the model (~14GB) — this is cached and won't repeat.

## Tips

- **Use `/workspace`** for anything you want to persist across pod restarts. Other directories get wiped.
- **Stop your pod** when you're not using it to save credits. Don't delete it unless you want to lose cached models.
- **Monitor GPU usage** with `nvidia-smi` in the terminal.
- If you need to push changes back to the repo, set up Git credentials on the pod:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your@email.com"
  ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH connection refused | Double-check IP, port, and that your public key is saved in RunPod settings |
| `uv: command not found` | Run `source ~/.bashrc` after installing uv |
| Out of GPU memory | Try loading the model in 4-bit: add `load_in_4bit=True` to `from_pretrained` (requires `bitsandbytes`) |
| Model download is slow | RunPod has good bandwidth — just be patient on first download |
