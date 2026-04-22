# CacheBlend Demo

> **Best Paper @ ACM EuroSys 2025** — *CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion*

CacheBlend speeds up LLM inference in RAG scenarios by reusing precomputed KV caches across requests — even when the same chunks appear in **different orders**. Instead of recomputing everything from scratch, it selectively recomputes only ~15% of tokens to restore cross-attention between chunks.

This demo runs Mistral-7B on a GPU instance using [vLLM](https://github.com/vllm-project/vllm) + [LMCache](https://github.com/LMCache/LMCache).

---

## Requirements

| | |
|---|---|
| **GPU** | 16GB+ VRAM (Mistral-7B needs ~14GB) |
| **CUDA** | 12.8 or higher |
| **Python** | 3.12 |
| **OS** | Linux |
| **Disk** | ~15GB free for model weights |

Tested on **vast.ai** GPU instances. Any cloud GPU provider works.

> **No API keys needed.** vLLM, LMCache, and Mistral-7B are all free and open source.

---

## Files

```
setup.sh            — Install everything and apply patches (run once)
demo_cacheblend.py  — Interactive step-by-step demo (run after setup)
```

---

## Quickstart

### Step 1 — Rent a GPU

On [vast.ai](https://vast.ai), rent any instance with 16GB+ VRAM. When choosing an image, pick a bare PyTorch image such as `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. Avoid pre-built vLLM images.

### Step 2 — Run setup

SSH into your instance and run:

```bash
bash setup.sh
```

This will, in order:

1. Install PyTorch nightly (cu128 wheels, compatible with CUDA 12.8+)
2. Install vLLM 0.19.1
3. Install LMCache 0.4.3
4. Download `mistralai/Mistral-7B-Instruct-v0.2` (~15GB, no HuggingFace token needed)
5. Apply three compatibility patches to LMCache 0.4.3 (see [Patches](#patches) below)
6. Verify base vLLM inference works

Expected output at the end:
```
✓ vLLM inference OK: A RAG system is a...
Setup complete! Run: python demo_cacheblend.py
```

### Step 3 — Run the demo

```bash
LMCACHE_ENABLE_BLENDING=True \
LMCACHE_BLEND_SPECIAL_STR=" # # " \
LMCACHE_USE_LAYERWISE=True \
LMCACHE_BLEND_CHECK_LAYERS=1 \
LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15 \
PYTHONHASHSEED=0 \
python demo_cacheblend.py
```

The demo is interactive — it pauses at each step with `Press Enter to continue` so you can walk through it with your team.

---

## What the Demo Shows

The demo sends three requests to the model, all using the same two context chunks about Messi and Ronaldo's World Cup goals, plus a long filler chunk:

| Request | Chunk order | What happens |
|---|---|---|
| **Request 1** (cold) | filler => messi => ronaldo | All tokens computed. KV caches stored in LMCache. |
| **Request 2** (reorder) | filler => **ronaldo => messi** | Chunks swapped. Prefix caching gets 0% hit on stat chunks. CacheBlend reuses both and recomputes 15% of tokens to restore cross-attention. |
| **Request 3** (repeat) | filler => ronaldo => messi | Near-full cache hit. Maximum speedup. |

Expected results:
```
Request 1 (cold):    ~0.7s  => Messi scored 13 goals...
Request 2 (reorder): ~0.3s  => Messi scored 5 more goals than Ronaldo...   (2–3x speedup)
Request 3 (repeat):  ~0.3s  => Messi scored 5 more goals than Ronaldo...   (2–3x speedup)
```

The key point: **Request 2 reorders the chunks**, which would break prefix caching entirely. CacheBlend still reuses the cached KV for both stat chunks and produces a correct comparative answer.

---

## CacheBlend Config Reference

| Environment Variable | Value | Meaning |
|---|---|---|
| `LMCACHE_ENABLE_BLENDING` | `True` | Activate CacheBlend |
| `LMCACHE_BLEND_SPECIAL_STR` | `" # # "` | Token sequence used as chunk boundary marker |
| `LMCACHE_USE_LAYERWISE` | `True` | Required for blending |
| `LMCACHE_BLEND_CHECK_LAYERS` | `1` | Use layer 1 to identify High-KV-Deviation (HKVD) tokens |
| `LMCACHE_BLEND_RECOMPUTE_RATIOS` | `0.15` | Recompute 15% of tokens per layer to restore cross-attention |

Increasing `LMCACHE_BLEND_RECOMPUTE_RATIOS` improves quality at the cost of speed. The paper shows 5–18% is sufficient for near-full-recompute quality.

---

## Patches

LMCache 0.4.3 has three bugs when used with vLLM 0.19.1 that prevent CacheBlend from starting. `setup.sh` applies these patches automatically.

**Patch 1 — `lmcache/v1/compute/blend/utils.py`**

`VLLMModelTracker.register_model()` is never called before the blender tries to use it, because the connector is initialized in a different subprocess than where the model is loaded. The fix defers blender creation to return `None` gracefully instead of crashing.

**Patch 2 — `lmcache/integration/vllm/vllm_v1_adapter.py`**

Guards the `self.blender.blend()` call with `and self.blender is not None` so that when blender creation was deferred, inference falls back cleanly to vLLM's built-in prefix caching.

**Patch 3 — `lmcache/v1/rpc/zmq_transport.py`**

vLLM passes token IDs as a `ConstantList` type internally. The LMCache msgpack serializer doesn't know how to encode it. The fix converts `ConstantList` to a plain Python `list` before encoding, while preserving `str`, `bytes`, and other primitive types as-is (an earlier version of this patch accidentally converted strings to character lists, breaking request ID serialization).

All three patches have been identified as upstream bugs in lmcache 0.4.3.

---

## Killing a Stuck GPU Process

If a previous run is holding GPU memory:

```bash
# Find the PID
nvidia-smi

# Kill it by PID (safe — only kills that specific process)
kill -9 <PID>

# Verify GPU is free
nvidia-smi
```

> Warning Never run `pkill -f "python"` — it kills system processes that keep your rented instance alive.

---

## References

- Paper: [CacheBlend @ EuroSys 2025](https://doi.org/10.1145/3689031.3696098)
- Code: [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache)
- Docs: [docs.lmcache.ai](https://docs.lmcache.ai)
- Blending docs: [docs.lmcache.ai/kv_cache_optimizations/blending.html](https://docs.lmcache.ai/kv_cache_optimizations/blending.html)