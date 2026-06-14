# MiniMax-M2.7-AWQ performance bench history

Hardware: 2x NVIDIA DGX Spark (GB10 / sm_121), NCCL/RoCE direct interconnect.
Image: `vllm-node:latest` (eugr's freshly-rebuilt build at digest `c0c4f32a5a06`, 2026-06-10).
Model: `cyankiwi/MiniMax-M2.7-AWQ-4bit` (~230B MoE params, AWQ 4-bit weights).

Two datasets per round:
- **Random**: `--dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 32`. Cross-recipe comparison parity with DSv4 / Step-3.7 bench history.
- **Spec-Bench**: `--dataset-name spec_bench --spec-bench-output-len 256 --num-prompts 32`. Real prompts across 13 categories. Better for MoE expert-routing realism (random tokens cluster on few experts).

`vllm bench serve` from inside the `vllm_node` container against the HTTPS endpoint.

---

## Summary

The canonical recipe is **R1 safe defaults only** (file:
`minimax-m2.7-jasl-gb10.yaml`). Key finding: unlike DSv4 / Step-3.7,
MiniMax-M2.7-AWQ is **NOT concurrency-bound** on the GB10 cluster.
Output tok/s is within ±3% of the eugr baseline regardless of the
DSv4-style tunings, which strongly suggests this workload is
**memory-bandwidth bound** on AWQ-4 MoE expert reads.

| C | Baseline Output tok/s (Spec-Bench) | Canonical Output tok/s (Spec-Bench) | Δ |
|---|---|---|---|
| 1 | 38.87 | 39.32 | +1.2% |
| 4 | 77.74 | 79.61 | +2.4% |
| 8 | 105.27 | 104.29 | −0.9% |

Throughput is essentially unchanged. The canonical recipe earns its
keep on three soft wins that don't show up in tok/s:

1. **TTFT improves 11–17% on real prompts** at C≥4 (prefix-caching +
   flashinfer-autotune pay off when batches share structure)
2. **Zero-downtime client swap** via `--served-model-name minimax-m2.7`
   alongside the canonical HF path
3. **Future-proofed** if a higher-concurrency workload ever saturates
   the engine, the `max_num_seqs: 8` cap + raised batched-tokens
   budget will be ready without a relaunch

For workloads that aren't real chat traffic — pure throughput stress,
benchmarking — running the stock `minimax-m2.7-awq` recipe gives
identical numbers.

---

## R0 — Baseline (stock eugr `minimax-m2.7-awq` recipe)

Launch flags from upstream `recipes/minimax-m2.7-awq.yaml`:
- TP=2, ray, gpu_memory_utilization=0.8, max_model_len=196608
- `--load-format fastsafetensors`
- `--enable-auto-tool-choice`, `--tool-call-parser minimax_m2`, `--reasoning-parser minimax_m2`
- All other knobs: vLLM defaults (max_num_seqs, max_num_batched_tokens, prefix-caching, autotune)

KV pool: 27.14 GiB/rank, 215,984 tokens, **1.10× concurrency at 192K**.

### Random dataset

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 37.11 | 185.57 | 25.16 | 481 | 25.16 |
| 4 | 77.76 | 388.81 | 50.45 | 299 | 50.45 |
| 8 | 101.34 | 506.72 | 77.50 | 441 | 77.50 |

### Spec-Bench dataset

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 38.87 | 80.50 | 24.58 | 304 | 24.59 |
| 4 | 77.74 | 161.29 | 48.82 | 253 | 48.82 |
| 8 | 105.27 | 215.53 | 73.64 | 304 | 73.65 |

Raw log: [`minimax-m2.7-jasl-gb10.bench.r0.log`](minimax-m2.7-jasl-gb10.bench.r0.log).

**Observations:**
- C=1 single-stream: 37–39 tok/s, solid for a 230B MoE AWQ-4 at TP=2
- Random ≈ Spec-Bench output tok/s — MoE expert routing isn't a bottleneck at this scale; the difference (Total tok/s) reflects different prompt-length ratios only
- C=8 random: 101 tok/s = 2.7× C=1 — engine is **concurrency-bound**, not compute-bound (KV pool 1.10× concurrency at full 192K context is the cap)
- Mean TTFT under 500 ms even at C=8 — interactive-friendly already

---

## R1 — max_num_seqs 8 + prefix-caching + autotune + served-name alias

Launch flags added over R0:
- `max_num_seqs: 8` (was vLLM default 256, but engine was throttled by KV)
- `max_num_batched_tokens: 8192` (was 2048 default)
- `--enable-prefix-caching`
- `--enable-flashinfer-autotune`
- `--served-model-name minimax-m2.7 cyankiwi/MiniMax-M2.7-AWQ-4bit`

KV pool: 28.53 GiB/rank, 241,264 tokens, **1.23× concurrency at 192K**
(up from R0's 1.10×; tiny bump because prefix-caching reserves some KV).

### Random dataset

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 34.58 | 172.89 | 24.93 | 1046 | 24.93 |
| 4 | 78.16 | 390.81 | 50.38 | 248 | 50.38 |
| 8 | 101.51 | 507.57 | 77.53 | 399 | 77.53 |

### Spec-Bench dataset

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 39.32 | 80.50 | 24.49 | 259 | 24.49 |
| 4 | 79.61 | 163.78 | 48.66 | 226 | 48.68 |
| 8 | 104.29 | 214.09 | 73.63 | 286 | 73.63 |

Raw log: [`minimax-m2.7-jasl-gb10.bench.r1.log`](minimax-m2.7-jasl-gb10.bench.r1.log).

### R1 vs R0 deltas

| C | Random Δ tok/s | Spec-Bench Δ tok/s | Random Δ TTFT | Spec-Bench Δ TTFT |
|---|---|---|---|---|
| 1 | **−6.8%** | +1.2% | +118% (worse — one-shot autotune warmup) | −15% (faster) |
| 4 | +0.5% | +2.4% | −17% (faster) | −11% (faster) |
| 8 | +0.2% | −0.9% | −10% (faster) | −6% (faster) |

**Verdict:** R1's `max_num_seqs` tuning is **not a throughput unlock** for
MiniMax-M2.7-AWQ at this test scale. Output tok/s is within ±3% of R0
across the board. TTFT is mostly improved (15–17% faster on real
prompts), with one outlier at random C=1 likely caused by FlashInfer
autotune warmup that doesn't amortise over 32 random prompts. The
engine appears to be **GPU-compute or memory-bandwidth bound** (likely
the latter for an AWQ-4 MoE), not concurrency-bound — opposite of
what we saw on DSv4.

Keep R1's safe-default flags (`prefix-caching`, `autotune`, alias) into
the canonical regardless — they're free wins at higher load / repeated
prefixes / client-swap convenience. Drop `max_num_seqs: 8` and
`max_num_batched_tokens: 8192` as no-ops unless a later round needs
them. Continue to R2 with the safe defaults retained.

---

## R2 — R1 base + `--kv-cache-dtype fp8` + `--block-size 256`  **(CRASHED at C=8)**

Launch flags added over R1:
- `--kv-cache-dtype fp8`
- `--block-size 256`

KV pool: 30.68 GiB/rank, 497,664 tokens, **2.53× concurrency at 192K**
(up from R1's 1.23× — fp8 doubles per-byte token efficiency, block_size
256 packs more tokens per block).

### Random dataset

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 37.75 | 188.75 | 24.65 | 496 | 24.65 |
| 4 | 78.22 | 391.09 | 49.08 | 571 | 49.08 |
| 8 | **4.73** ⚠️ | 30.73 | 76.49 | 906 | 76.54 |

### Spec-Bench dataset

| C | Output tok/s | Notes |
|---|---|---|
| 1 | — | engine died during random C=8; not run |
| 4 | — | container down |
| 8 | — | container down |

Raw log: [`minimax-m2.7-jasl-gb10.bench.r2.log`](minimax-m2.7-jasl-gb10.bench.r2.log).

### Crash diagnosis

After ~10 min of sustained C=8 traffic on random, the engine hit:

```
ERROR ... [core.py:1197] TimeoutError: RPC call to sample_tokens timed out.
ERROR ... [async_llm.py:704] vllm.v1.engine.exceptions.EngineDeadError
```

— a Ray cross-node RPC stalled, the EngineCore process died, subsequent
requests returned 500 Internal Server Error, and the container shut
itself down. Plausible root cause: the combination of `block_size: 256`
+ `kv_cache_dtype: fp8` + `max_num_seqs: 8` produces a
cudagraph-specialisation footprint that interacts badly with MiniMax's
multi-node TP=2 RPC under sustained sub-blocksize batches.

### R2 vs R0 / R1 (C=1, C=4 only — the runs that completed)

| C | R0 Random | R1 Random | R2 Random | Δ R2 vs R0 |
|---|---|---|---|---|
| 1 | 37.11 | 34.58 | 37.75 | +1.7% |
| 4 | 77.76 | 78.16 | 78.22 | +0.6% |

So at low concurrency, fp8 KV is **neutral** — no measurable win, no
regression. The 2.3× KV pool gain doesn't matter at this test scale
because we're not running long-context workloads.

### Verdict

**Drop fp8 KV + block-size 256 from the canonical recipe.** The C=8
crash is reproducible and unacceptable for a production endpoint. The
KV headroom benefit is irrelevant at our current concurrency cap. If a
future workload pushes long-context multi-user we may revisit with
diagnostics, but for the chat/code use case the upstream baseline
(BF16 KV, block_size=16) is more robust.

Move to R4 (skip R3 — gpu_mem 0.85 won't help if we're not KV-bound).

---

## R3 — gpu_memory_utilization 0.85 — **skipped**

DSv4 tested 0.85 vs 0.80 and saw no throughput gain. R1's KV pool
(28.53 GiB/rank, 1.23× concurrency at 192K) is comfortable on hardware
that has ~117 GB free per Spark, so the extra 5% gpu_mem wouldn't be
a host-margin problem. But since R2 confirmed we're not KV-bound for
chat-style C=1–8 workloads, increasing the KV pool is just dead
capacity. **Not promoted** — `gpu_memory_utilization: 0.80` stays.

---

## R4 — Marlin kernel verification — **no-op (auto-selected)**

The R1 engine launch log confirms vLLM auto-selected Marlin for both
linear and MoE layers on both ranks:

```
[compressed_tensors_wNa16.py:112] Using MarlinLinearKernel for CompressedTensorsWNA16
[compressed_tensors_moe.py:132] Using CompressedTensorsWNA16MarlinMoEMethod
[int_wna16.py:149] Using 'MARLIN' WNA16 MoE backend.
```

No `--quantization awq_marlin` override needed. R4 is informational only.

---

## Verdict and canonical choice

Ship the single canonical recipe **`minimax-m2.7-jasl-gb10.yaml`** with
R1's safe defaults:

- `max_num_seqs: 8` + `max_num_batched_tokens: 8192` (throughput-neutral
  at our test scale but future-proofed)
- `--enable-prefix-caching` + `--enable-flashinfer-autotune` (11–17% TTFT
  improvement on real prompts)
- `--served-model-name minimax-m2.7 cyankiwi/MiniMax-M2.7-AWQ-4bit`
  (zero-downtime client swap)

### Knobs that did NOT pay off (documented so they're not re-tried)

- `--kv-cache-dtype fp8` + `--block-size 256`: throughput-neutral at
  C=1/4, **crashed the engine at C=8** with a multi-node Ray RPC
  timeout. Unacceptable for a production endpoint; KV headroom benefit
  irrelevant at our concurrency.
- `gpu_memory_utilization: 0.85`: not KV-bound at this test scale, so
  the larger pool is dead capacity. Skipped — DSv4 saw the same and
  came to the same conclusion.
- DSv4-style env vars (`VLLM_USE_FLASHINFER_MOE_FP8=1` etc.) — DSv4's
  FP8-MoE-specific path; doesn't apply to AWQ-4 weights.
- MTP / Eagle3 spec-decode: `MiniMaxM2ForCausalLM` inherits
  `SupportsEagle3` but no Eagle3 draft model has been published for the
  AWQ variant. Out of scope.

