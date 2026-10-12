#!/bin/bash
# Creates .venv, installs the pinned requirements, and downloads the public NanoJev checkpoint and dataset
# that every command in this repository starts from. Run from the repository root.
set -euo pipefail
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  . .venv/bin/activate
  uv pip install -r requirements-toy.txt -r requirements-vizdoom.txt huggingface_hub
else
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements-toy.txt -r requirements-vizdoom.txt huggingface_hub
fi
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(repo_id="C-Tianyu/NanoJev", revision="unified-games-v1", local_dir="checkpoints/NanoJev-unified",
                  allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*",
                                  "TRAINING_RECIPE.md", "*MANIFEST.json", "README.md"])
snapshot_download(repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset", revision="unified-games-v1",
                  local_dir="data/NanoJev-unified")
PY
python -c "import torch, transformers, vizdoom; print(torch.__version__, transformers.__version__, vizdoom.__version__, 'cuda', torch.cuda.is_available())"
