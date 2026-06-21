# RunPod Guide — SPAR Causal Probes

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

Once running, click **Connect** on your pod dashboard and note the SSH connection details.

## 3. Update Your SSH Config

Add or update the RunPod entry in `~/.ssh/config`:

```
Host runpod
    HostName <POD_IP>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519_runpod
```

Replace `<POD_IP>` and `<PORT>` with the values from the pod dashboard.

Then run the following once to verify the connection and accept the host key:

```bash
ssh runpod
```

Type `yes` when prompted about the host fingerprint, then `exit`.

## 4. Run the Startup Script

From the repo root on your **local machine**, run:

```bash
./setup/startup.sh
```

This script (run against `Host runpod` by default) will:

1. Clone the repo into `/workspace/SPAR-causal-probes` on the pod (or pull latest if already there)
2. Install `uv` if missing and run `uv sync` to install all dependencies
3. Install `rsync` on the pod if missing
4. Sync local data directories (`data/`, `prefill_awareness/outputs/`) to the pod, skipping files that already exist
5. Set `HF_HOME` to the container disk (avoids network volume size limits)
6. Sync your local `.env` file to the pod

> To target a different SSH host: `./setup/startup.sh <SSH_HOST>`

## 5. Connect in VS Code

1. Open the Command Palette (`Cmd+Shift+P`)
2. Select **Remote-SSH: Connect to Host**
3. Choose `runpod`

You're now editing and running code directly on the pod.

## Tips

- **Use `/workspace`** for anything you want to persist across pod restarts. Other directories get wiped.
- **Stop your pod** when you're not using it to save credits. Don't delete it unless you want to lose cached models.
- **Monitor GPU usage** with `nvidia-smi` in the terminal.
- **Re-running startup.sh** on an existing pod is safe — it skips files that are already present and pulls the latest code.
- If you need to push changes back to the repo, set up Git credentials on the pod:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your@email.com"
  ```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| SSH connection refused | Double-check IP, port, and that your public key is saved in RunPod settings |
| Host key prompt hangs startup.sh | Run `ssh runpod` once manually and accept the fingerprint first |
| `uv: command not found` | Run `source ~/.bashrc` after installing uv |
| Out of GPU memory | Try loading the model in 4-bit: add `load_in_4bit=True` to `from_pretrained` (requires `bitsandbytes`) |
| Model download is slow | RunPod has good bandwidth — just be patient on first download |
