#!/bin/bash
# =============================================================================
# CacheBlend Demo — Setup Script
# Tested on: vast.ai GPU instance, CUDA 12.8, Python 3.12
# GPU requirement: 16GB+ VRAM (Mistral-7B needs ~14GB)
# =============================================================================
set -e  # stop on first error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         CacheBlend Demo — Environment Setup          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: PyTorch (nightly, CUDA 12.8 wheels work on CUDA 12.8+) ──────────
echo " Step 1/6: Installing PyTorch (nightly cu128)..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 -q
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  OK ~ PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}')
"

# ── Step 2: vLLM ─────────────────────────────────────────────────────────────
echo " Step 2/6: Installing vLLM 0.19.1..."
pip install vllm==0.19.1 -q
python -c "import vllm; print(f'  OK ~ vLLM {vllm.__version__}')"

# ── Step 3: LMCache ──────────────────────────────────────────────────────────
echo " Step 3/6: Installing LMCache 0.4.3..."
pip install lmcache==0.4.3 -q
python -c "import lmcache; print('  OK ~ LMCache OK')"

# ── Step 4: Download Mistral-7B ──────────────────────────────────────────────
echo " Step 4/6: Downloading Mistral-7B-Instruct-v0.2 (~15GB)..."
pip install huggingface_hub -q
mkdir -p /workspace/models
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="/workspace/models/Mistral-7B-Instruct-v0.2",
    ignore_patterns=["*.pt", "*.bin"],
)
print("  OK ~ Model downloaded")
EOF

# ── Step 5: Apply LMCache patches ────────────────────────────────────────────
echo " Step 5/6: Applying LMCache compatibility patches..."

LMCACHE_SITE="/venv/main/lib/python3.12/site-packages/lmcache"

# Patch 1: blend/utils.py — defer blender creation if model not registered yet
cat > "$LMCACHE_SITE/v1/compute/blend/utils.py" << 'ENDOFFILE'
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Dict
from torch import nn
from lmcache.logging import init_logger
from lmcache.v1.compute.blend.blender import LMCBlender
from lmcache.v1.compute.models.utils import VLLMModelTracker
if TYPE_CHECKING:
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector import GPUConnectorInterface

logger = init_logger(__name__)

class LMCBlenderBuilder:
    _blenders: Dict[str, LMCBlender] = {}

    @classmethod
    def get_or_create(cls, instance_id, cache_engine, gpu_connector, config):
        if instance_id not in cls._blenders:
            logger.info(f"Creating blender for {instance_id}")
            try:
                vllm_model = VLLMModelTracker.get_model(instance_id)
            except ValueError:
                logger.warning(
                    f"vllm model for {instance_id} not registered yet — "
                    "blender creation deferred."
                )
                cls._blenders[instance_id] = None
                return None
            blender = LMCBlender(
                cache_engine=cache_engine,
                gpu_connector=gpu_connector,
                vllm_model=vllm_model,
                config=config,
            )
            cls._blenders[instance_id] = blender
        else:
            logger.info(f"Blender for {instance_id} already exists.")
        return cls._blenders[instance_id]

    @classmethod
    def get(cls, instance_id) -> nn.Module:
        if instance_id not in cls._blenders:
            raise ValueError(f"Blender for {instance_id} not found.")
        return cls._blenders[instance_id]
ENDOFFILE

# Patch 2: vllm_v1_adapter.py — guard blend() call against None blender
python3 - <<'PYEOF'
path = "/venv/main/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py"
with open(path) as f: content = f.read()
old = "                if self.enable_blending:\n                    # TODO(Jiayi): Need to make prefix caching and blending compatible\n                    self.blender.blend("
new = "                if self.enable_blending and self.blender is not None:\n                    # TODO(Jiayi): Need to make prefix caching and blending compatible\n                    self.blender.blend("
if old in content:
    with open(path, "w") as f: f.write(content.replace(old, new, 1))
    print("  OK ~ Patch 2 applied")
else:
    print("   Patch 2 already applied or not needed")
PYEOF

# Patch 3: zmq_transport.py — convert ConstantList to plain list, preserve str/bytes
python3 - <<'PYEOF'
path = "/venv/main/lib/python3.12/site-packages/lmcache/v1/rpc/zmq_transport.py"
with open(path) as f: content = f.read()
old = "        encoded = [self.encoder.encode(m) for m in msg]"
new = (
    "        def _to_plain(x):\n"
    "            if isinstance(x, (str, bytes, int, float, dict, bool)): return x\n"
    "            if isinstance(x, list): return x\n"
    "            try: return list(x)\n"
    "            except Exception: return x\n"
    "        encoded = [self.encoder.encode(_to_plain(m)) for m in msg]"
)
if old in content:
    with open(path, "w") as f: f.write(content.replace(old, new, 1))
    print("  OK ~ Patch 3 applied")
else:
    print("  ℹ Patch 3 already applied or not needed")
PYEOF

# Clear pyc caches
find "$LMCACHE_SITE" -name "*.pyc" -delete 2>/dev/null || true
find "$LMCACHE_SITE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# ── Step 6: Verify base vLLM works ──────────────────────────────────────────
echo " Step 6/6: Verifying base vLLM inference..."
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(
    model="/workspace/models/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
out = llm.generate(["What is RAG?"], SamplingParams(temperature=0.0, max_tokens=30))
print(f"  OK ~ vLLM inference OK: {out[0].outputs[0].text.strip()[:60]}...")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete! Run: python demo_cacheblend.py      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
