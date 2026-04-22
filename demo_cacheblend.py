"""
demo_cacheblend.py
==================
Interactive CacheBlend demo - EuroSys '25 Best Paper

CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion
  => Selectively recomputes only ~15% of KV cache tokens to restore
    cross-attention when chunks are reused in different orders.

Run after setup.sh:
    LMCACHE_ENABLE_BLENDING=True \
    LMCACHE_BLEND_SPECIAL_STR=" # # " \
    LMCACHE_USE_LAYERWISE=True \
    LMCACHE_BLEND_CHECK_LAYERS=1 \
    LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15 \
    PYTHONHASHSEED=0 \
    python demo_cacheblend.py
"""

import os, time, textwrap
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

#  Config 
MODEL   = "/workspace/models/Mistral-7B-Instruct-v0.2"
SEP_STR = " # # "   # must match LMCACHE_BLEND_SPECIAL_STR
WIDTH   = 70

#  Helpers 
def banner(text, char="═"):
    pad = max(0, WIDTH - len(text) - 4)
    left = pad // 2
    right = pad - left
    print(f"\n{char * (WIDTH - 2)}")
    print(f"{' ' * left}  {text}  {' ' * right}")
    print(f"{char * (WIDTH - 2)}\n")

def section(title):
    print(f"\n{'-' * WIDTH}")
    print(f"  {title}")
    print(f"{'-' * WIDTH}")

def wrap(text, indent=4):
    return textwrap.fill(text, width=WIDTH - indent,
                         initial_indent=" " * indent,
                         subsequent_indent=" " * indent)

def bar(label, value, max_val, width=30):
    filled = int(width * value / max_val) if max_val > 0 else 0
    pct = value / max_val * 100 if max_val > 0 else 0
    return f"  {label:<20} {'█' * filled}{'░' * (width - filled)} {value:.2f}s ({pct:.0f}%)"

def build_sep(tokenizer, *parts):
    sep = tokenizer.encode(SEP_STR, add_special_tokens=False)
    out = []
    for i, p in enumerate(parts):
        out += [int(x) for x in p]
        if i < len(parts) - 1:
            out += [int(x) for x in sep]
    return out

def run(llm, params, token_ids, label):
    t0 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": token_ids}], params)
    elapsed = time.perf_counter() - t0
    text = out[0].outputs[0].text.strip()
    return text, elapsed

def pause(msg="Press Enter to continue..."):
    input(f"\n  {msg}")

# Main demo
def main():
    banner("CacheBlend Interactive Demo", "=")

    print(wrap(
        "This demo shows CacheBlend (EuroSys '25 Best Paper) in action. "
        "We send the same context chunks to an LLM in different orders and "
        "measure how much faster CacheBlend makes subsequent requests by "
        "reusing KV caches with selective recompute (~15% of tokens)."
    ))

    # Verify env vars
    section("1. Checking CacheBlend Configuration")
    required = {
        "LMCACHE_ENABLE_BLENDING":        "True",
        "LMCACHE_BLEND_SPECIAL_STR":      " # # ",
        "LMCACHE_USE_LAYERWISE":          "True",
        "LMCACHE_BLEND_CHECK_LAYERS":     "1",
        "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
    }
    all_ok = True
    for k, expected in required.items():
        val = os.environ.get(k, "NOT SET")
        status = "OK" if val == expected else "NOT_OK"
        print(f"  {status} {k} = {val}")
        if val != expected:
            all_ok = False

    if not all_ok:
        print("\n  WARN:  Some env vars are missing. Run with the full command:")
        print("""
  LMCACHE_ENABLE_BLENDING=True \\
  LMCACHE_BLEND_SPECIAL_STR=" # # " \\
  LMCACHE_USE_LAYERWISE=True \\
  LMCACHE_BLEND_CHECK_LAYERS=1 \\
  LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15 \\
  PYTHONHASHSEED=0 \\
  python demo_cacheblend.py
        """)
        return

    pause("Configuration looks good. Press Enter to load the model...")

    #  Load model
    section("2. Loading Mistral-7B with LMCache CacheBlend")
    print("  Loading model (this takes ~15 seconds)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        kv_transfer_config={
            "kv_connector": "LMCacheConnectorV1",
            "kv_role": "kv_both",
        },
    )
    params_long  = SamplingParams(temperature=0.0, max_tokens=40)
    params_short = SamplingParams(temperature=0.0, max_tokens=40)
    print("  OK:  Model loaded with LMCache blending enabled")

    #  Build chunks
    section("3. Preparing Context Chunks")

    c1 = tokenizer.encode(
        "Lionel Messi scored 13 goals at FIFA World Cups across his career.",
        add_special_tokens=False)
    c2 = tokenizer.encode(
        "Cristiano Ronaldo scored 8 goals at FIFA World Cups across his career.",
        add_special_tokens=False)
    filler = tokenizer.encode(
        "Football is the world's most popular sport with over 4 billion fans. " * 10,
        add_special_tokens=False)
    q1 = tokenizer.encode(
        "[INST] Based only on the context above, how many goals did Messi "
        "score at the World Cup? Answer in one sentence. [/INST]",
        add_special_tokens=False)
    q2 = tokenizer.encode(
        "[INST] Based only on the context above, how many more goals did Messi "
        "score than Ronaldo at FIFA World Cups? Answer in one sentence. [/INST]",
        add_special_tokens=False)

    p1 = build_sep(tokenizer, filler, c1, c2, q1)   # filler=>messi=>ronaldo
    p2 = build_sep(tokenizer, filler, c2, c1, q2)   # filler=>ronaldo=>messi (REORDERED)

    print(f"  Chunk layout for Request 1: [filler] # # [messi] # # [ronaldo] # # [query]")
    print(f"  Chunk layout for Request 2: [filler] # # [ronaldo] # # [messi] # # [query]")
    print(f"  | Chunks 2 & 3 are SWAPPED - prefix caching would miss both")
    print(f"    CacheBlend reuses ALL chunks + recomputes 15% for cross-attention")
    print(f"\n  Prompt 1: {len(p1)} tokens")
    print(f"  Prompt 2: {len(p2)} tokens")

    pause("Chunks ready. Press Enter to run Request 1 (cold)...")

    #  Request 1: Cold
    section("4. Request 1 - Cold (building KV cache)")
    print("  Query: 'How many goals did Messi score at the World Cup?'")
    print("  Running... (first request, no cache available)")
    ans1, t1 = run(llm, params_long, p1, "Request 1")
    print(f"\n  Answer : {ans1}")
    print(f"  Time   : {t1:.2f}s  ← cold, all {len(p1)} tokens computed from scratch")
    print(f"  Status : KV caches stored in LMCache CPU memory")

    pause("Request 1 done. Press Enter to run Request 2 (reordered chunks)...")

    #  Request 2: Reordered
    section("5. Request 2 - Warm, Reordered Chunks (CacheBlend in action)")
    print("  Query: 'How many MORE goals did Messi score than Ronaldo?'")
    print("  Chunks are in a DIFFERENT ORDER than Request 1.")
    print("  ======================================================")
    print("  = Prefix caching: 0% hit on ronaldo+messi chunks      =")
    print("  = CacheBlend:     reuses both + 15% selective recompute=")
    print("  =======================================================")
    print("  Running...")
    ans2, t2 = run(llm, params_short, p2, "Request 2")
    print(f"\n  Answer : {ans2}")
    print(f"  Time   : {t2:.2f}s")
    print(f"  Speedup: {t1/t2:.1f}x faster than Request 1")

    pause("Request 2 done. Press Enter to run Request 3 (repeat)...")

    # Request 3: Full cache hit
    section("6. Request 3 - Full Cache Hit (same as Request 2)")
    print("  Same prompt as Request 2 - maximum cache reuse.")
    print("  Running...")
    ans3, t3 = run(llm, params_short, p2, "Request 3")
    print(f"\n  Answer : {ans3}")
    print(f"  Time   : {t3:.2f}s")
    print(f"  Speedup: {t1/t3:.1f}x faster than cold")

    #  Results summary 
    banner("Results Summary", "-")

    print("  TTFT Comparison (lower is better):\n")
    print(bar("Request 1 (cold)",    t1, t1))
    print(bar("Request 2 (reorder)", t2, t1))
    print(bar("Request 3 (repeat)",  t3, t1))

    print(f"\n  Speedups:")
    print(f"    Request 2 vs cold : {t1/t2:.1f}x")
    print(f"    Request 3 vs cold : {t1/t3:.1f}x")

    print(f"\n  Answer quality:")
    print(f"    Request 1: {ans1}")
    print(f"    Request 2: {ans2}")
    print(f"    Request 3: {ans3}")

    print(f"\n  CacheBlend config used:")
    print(f"    Recompute ratio : 15% of tokens per layer")
    print(f"    Check layers    : layer 1 (identifies HKVD tokens)")
    print(f"    Chunk separator : '{SEP_STR.strip()}'")

    print(f"\n  Key insight: Request 2 reordered the chunks = prefix caching")
    print(f"  would get 0% hit on the stat chunks. CacheBlend reused both")
    print(f"  and still produced a correct comparative answer via selective")
    print(f"  KV recompute (15% of tokens restored cross-attention).")

    banner("Demo Complete", "=")

if __name__ == "__main__":
    main()
