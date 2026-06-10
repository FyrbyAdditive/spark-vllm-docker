#!/bin/bash
# Runtime patch: keep the Step-3.7-Flash MTP drafter's `mtp_block` + `shared_head`
# UNQUANTIZED on NVFP4 checkpoints so the BF16-grafted MTP weights load without
# a shape mismatch.
#
# Why this mod exists:
#   The eugr `mods/step-3.7-flash/` mod's run.sh self-skips when it detects
#   that vLLM "already has Step-3.7 support" — but the actual MTP-unquant fix
#   was added LATER to the same patch file (in eugr PR #268). The detector
#   doesn't know about the extension, so when our image's vLLM already has the
#   base Step-3.7 model classes, the MTP fix from PR #268 NEVER lands. We pull
#   the MTP-fix-only hunk into this standalone mod and always try to apply it.
#
#   The fix itself is from eugr PR #268 ("feat: Step-3.7-Flash NVFP4 + MTP
#   speculative decoding") by @choiceoh, who verified it on dual DGX Spark
#   GB10/sm_121a, TP=2, with mean MTP acceptance length ~2.4-2.6 tokens/step.
#   The upstream root cause is tracked in vllm-project/vllm#44087.
#
# This is a NO-OP when the patch is already applied or when the file doesn't
# look like the expected pre-patch state (so it stays idempotent across
# image rebuilds).

set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$MOD_DIR/mtp-unquant.patch"

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
    echo "[step-3.7-mtp-unquant] vLLM not found at $PYTHON_ROOT/vllm" >&2
    exit 1
fi

if [ ! -f "$PATCH_FILE" ]; then
    echo "[step-3.7-mtp-unquant] patch file missing: $PATCH_FILE" >&2
    exit 1
fi

target="$PYTHON_ROOT/vllm/model_executor/models/step3p5_mtp.py"
if [ ! -f "$target" ]; then
    echo "[step-3.7-mtp-unquant] step3p5_mtp.py not found; vLLM build lacks Step-3.5/3.7 MTP support entirely. Skipping." >&2
    exit 0
fi

cd "$PYTHON_ROOT"

# Marker-based idempotency check.
if grep -q "_mtp_unquant" "$target"; then
    echo "[step-3.7-mtp-unquant] Already applied; skipping."
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "[step-3.7-mtp-unquant] git is required to apply this mod." >&2
    exit 1
fi

if git apply --check "$PATCH_FILE" 2>/dev/null; then
    git apply "$PATCH_FILE"
    echo "[step-3.7-mtp-unquant] Applied MTP-unquant fix from eugr PR #268."
else
    echo "[step-3.7-mtp-unquant] Patch could not be applied. The vLLM step3p5_mtp.py may have diverged." >&2
    echo "[step-3.7-mtp-unquant] Regenerate the patch against the current vLLM source." >&2
    exit 1
fi
