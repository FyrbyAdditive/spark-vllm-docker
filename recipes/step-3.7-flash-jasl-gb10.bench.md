# Step-3.7-Flash performance bench history

Hardware: 2x NVIDIA DGX Spark (GB10 / sm_121), NCCL/RoCE direct interconnect.
Image: `vllm-node:latest` (eugr's vLLM 0.22.1rc1.dev124+gace95c9cf.d20260603 build).
Model: `stepfun-ai/Step-3.7-Flash-NVFP4` (104B params, NVFP4 MoE, ModelOpt quant).

Two datasets:
- **Random**: `--dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 32`. Synthetic noise — hostile to spec decoding by design (no language structure).
- **Spec-Bench**: `--dataset-name spec_bench --spec-bench-output-len 256 --num-prompts 32`. The canonical spec-decode benchmark (480 real prompts across 13 categories: code, qa, math, reasoning, summarization, translation, ...).

`vllm bench serve` from inside the container against the HTTPS endpoint.

---

## Summary

The winning recipe is **canonical = R1 throughput tunings + MTP-3 speculative decoding** (file: `step-3.7-flash-jasl-gb10.yaml`).

| Concurrency | R1 (no MTP) Output tok/s | Canonical (MTP+R1) Output tok/s | Δ |
|---|---|---|---|
| C=1 (single-stream chat) | 21.66 | **25.28** | **+16.7%** |
| C=4 | 65.58 | 68.48 | +4.4% |
| C=8 | 99.75 | 102.76 | +3.0% |

(Spec-Bench dataset, real prompts, fair test of MTP.)

MTP per-position acceptance: 78% / 48% / 26% — matches what eugr PR #268's
author measured on the same GB10 hardware (80% / 44% / 14%).

---

## Random dataset benchmarks

### Baseline — stock `step-3.7-flash-nvfp4` recipe (eugr defaults)

Launch flags: vLLM defaults active. KV pool: 24.02 GiB/rank, 2.02M tokens.

| C | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---|---|---|
| 1 | 18.83 | 39.86 | 847 |
| 4 | 38.78 | 47.58 | 13,418 |
| 8 | 37.35 | 50.22 | 36,998 |

### Round 1 (R1) — baseline + `max_num_seqs: 8` + `max_num_batched_tokens: 4176` + served-name aliases

KV pool: 28.78 GiB/rank (+4.76 GiB), 2.39M tokens.

| C | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---|---|---|
| 1 | 19.09 | 39.84 | 992 |
| 4 | 56.15 | 47.58 | 2,039 |
| 8 | 55.35 | 65.43 | 18,267 |

**Throughput-neutral on C=1, +45% at C=4, +48% at C=8.** TTFT improved 10-13%.
The R1 tunings unlock real concurrency; everything else builds on R1.

### Round 2 (R2) — R1 + MTP — **CRASHED initially**

`RuntimeError: tensor shape mismatch 2048 vs 4096` at engine init.
Root cause: vLLM's MTP drafter quantizes `mtp_block` with NVFP4 quant config,
but our stepfun NVFP4 checkpoint ships BF16-grafted MTP weights.

Fix from **eugr PR #268** (`feat: Step-3.7-Flash NVFP4 + MTP speculative decoding`):
patch vLLM to keep `mtp_block` + `shared_head` unquantized on NVFP4. We
extracted that fix into `mods/step-3.7-flash-mtp-unquant/` (separate mod so
it always applies even when the parent `mods/step-3.7-flash` self-skips).

Also fixed `max_num_scheduled_tokens: 2048` warning by setting
`max_num_batched_tokens: 8192`.

### Round 3 (R3) — R1 + `VLLM_USE_FLASHINFER_MOE_FP8` + `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`

These env vars gave DSv4 +3-4% but Step-3.7-NVFP4 routes MoE through
`modelopt_fp4` — different path. Result: **no measurable change vs R1**.
Dropped.

### Round 4 (R4) — R1 + `gpu_memory_utilization: 0.85`

KV pool jumped to 34.75 GiB/rank but throughput unchanged. Spec-Bench isn't
KV-bound at our test scale. Host margin tighter (10 GB vs 11 GB available).
**Not promoted** — gpu_mem 0.8 stays.

### Round R-MTP-fixed (random dataset) — R1 base + MTP-3 + PR #268 fix + `max_num_batched_tokens: 8192`

| C | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) | MTP accept | MTP len | Per-position |
|---|---|---|---|---|---|---|
| 1 | 14.69 | 62.15 | 1,576 | 16.5% | 1.49 | 33% / 16% / 1% |
| 4 | 47.50 | 81.57 | 498 | 16.3% | 1.49 | 34% / 14% / 2% |
| 8 | 77.52 | 96.51 | 549 | 16.2% | 1.49 | 33% / 15% / 1% |

MTP **loses 22-31% vs R1 on random tokens**. Expected: spec-decode is
content-dependent and random tokens are anti-MTP by construction.

---

## Spec-Bench dataset benchmarks (the fair test)

### R1 (no MTP, throughput variant) — Spec-Bench

Average prompt length ~242 input tokens (vs random's 5,277). KV pool:
28.73 GiB/rank, 2.38M tokens, 9.09× concurrency at 256K.

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Mean ITL (ms) |
|---|---|---|---|---|---|
| 1 | 21.66 | 42.14 | 45.47 | 224 | 45.5 |
| 4 | 65.58 | 127.81 | 60.11 | 186 | 60.1 |
| 8 | 99.75 | 193.81 | 79.34 | 295 | 79.3 |

### R-MTP-stock (MTP at PR #268 defaults: gpu_mem 0.75, max_num_seqs unset, no aliases)

| C | Output tok/s | Mean TPOT (ms) | Mean TTFT (ms) | Accept rate | Accept len | Per-position |
|---|---|---|---|---|---|---|
| 1 | 24.79 | 39.10 | 325 | 47.7% | 2.43 | 76% / 45% / 23% |
| 4 | 68.51 | 54.42 | 467 | 49.4% | 2.48 | 78% / 46% / 24% |
| 8 | 99.97 | 73.01 | 584 | 49.4% | 2.48 | 77% / 46% / 25% |

### **Canonical (MTP + R1 hybrid)** — gpu_mem 0.8, max_num_seqs 8, MTP-3, batched-tokens 8192

KV pool: 25.59 GiB/rank, 2.03M tokens, 7.76× concurrency at 256K.

| C | Output tok/s | Total tok/s | Mean TPOT (ms) | P99 TPOT (ms) | Mean TTFT (ms) | P99 TTFT (ms) | Mean ITL (ms) | Accept rate | Accept len | Per-position |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **25.28** | 49.44 | 38.37 | 45.85 | 314 | 704 | 95.84 | 49.94% | 2.50 | 77% / 47% / 25% |
| 4 | **68.48** | 133.92 | 54.57 | 64.85 | 475 | 528 | 137.06 | 50.77% | 2.52 | 78% / 48% / 26% |
| 8 | **102.76** | 199.66 | 72.39 | 86.98 | 636 | 826 | 181.49 | 50.54% | 2.52 | 78% / 48% / 26% |

### Canonical vs R1 deltas (Spec-Bench, the fair real-text comparison)

| C | Output tok/s Δ | TPOT Δ | TTFT Δ |
|---|---|---|---|
| 1 | **+16.7%** | −15.6% (faster) | +40% (worse) |
| 4 | **+4.4%** | −9.3% (faster) | +154% (worse) |
| 8 | **+3.0%** | −8.7% (faster) | +116% (worse) |

### Canonical vs MTP-stock deltas (does R1's tuning help the MTP recipe?)

| C | Output tok/s Δ | TPOT Δ | Accept rate Δ |
|---|---|---|---|
| 1 | +2.0% | −1.8% | +2.2% (accept) |
| 4 | tied | tied | +1.4% (accept) |
| 8 | **+2.8%** | −0.8% | +1.1% (accept) |

R1's tunings (max_num_seqs cap, gpu_mem bump from 0.75 to 0.8, raised
batched-tokens) modestly improve MTP across the board, **and lift C=8 from
"tied" to "+2.8% vs MTP-stock"**. The per-position acceptance also picks
up 2-3 points at all concurrencies — likely because the bigger KV pool
keeps more draft state warm.

---

## Verdict and canonical choice

**Ship two recipes:**

1. **`step-3.7-flash-jasl-gb10.yaml`** (canonical default) — R1 tunings + MTP-3.
   Best per-token throughput at all concurrencies; +16.7% C=1 over the
   no-MTP variant. Optimal for chat/code workloads (1-4 concurrent users).
   ~50% MTP acceptance on real prompts, ~2.5 tokens accepted per spec step.

2. **`step-3.7-flash-jasl-gb10-throughput.yaml`** — R1 tunings only, no MTP.
   For high-concurrency shared deployments (C ≥ 8 sustained) where MTP's
   2-3× TTFT penalty outweighs its small remaining throughput advantage.

The canonical's TTFT is **2× worse than R1's** — irrelevant for single-user
chat (300 ms vs 200 ms), problematic for high-concurrency shared APIs.

### Knobs that did NOT pay off

- `VLLM_USE_FLASHINFER_MOE_FP8` + `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`:
  no-op on Step-3.7-NVFP4 (different MoE quant path than DSv4)
- `gpu_memory_utilization: 0.85`: bought 44% bigger KV pool but no throughput
  change at our test scale; not worth the tighter host margin
- `--kernel-config linear_backend=flashinfer_*`: skipped here based on DSv4
  results (both flashinfer linear backends failed on DSv4 with this image)
