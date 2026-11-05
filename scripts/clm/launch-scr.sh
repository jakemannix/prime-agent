#!/usr/bin/env bash
set -euo pipefail
# Pinned command template for a later authenticated CUDA host.
# It intentionally does not provision, deploy, or claim a successful model load.
: "${SCR_MODEL:=Qwen/Qwen3.6-27B}"
: "${SCR_MODEL_REVISION:=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
: "${SCR_SERVED_MODEL:=qwen36-27b}"
: "${SCR_HOST:=127.0.0.1}"
: "${SCR_PORT:=30000}"
: "${SCR_TP_SIZE:=1}"
: "${SCR_CONTEXT_LENGTH:=65536}"
: "${SCR_MEM_FRACTION_STATIC:=0.75}"
: "${SCR_CHUNKED_PREFILL_SIZE:=512}"
# Set these in the pinned suffix_cache_reuse environment; launcher names may vary by revision.
: "${KVREUSE_SIDE_SESSIONS:=1}"; : "${KVREUSE_MAX_SESSIONS:=1}"; : "${KVREUSE_SIDE_TOKENS:=40960}"
: "${KVREUSE_MAX_BLOCKS:=6}"; : "${KVREUSE_V6_MIN_EXTEND:=16}"; : "${KVREUSE_SSM_MODE:=fork}"; : "${KVREUSE_TRACE:=0}"
export KVREUSE_SIDE_SESSIONS KVREUSE_MAX_SESSIONS KVREUSE_SIDE_TOKENS KVREUSE_MAX_BLOCKS KVREUSE_V6_MIN_EXTEND KVREUSE_SSM_MODE KVREUSE_TRACE
exec python -m suffix_cache_reuse.serve \
  --model-path "$SCR_MODEL" --revision "$SCR_MODEL_REVISION" --served-model-name "$SCR_SERVED_MODEL" \
  --host "$SCR_HOST" --port "$SCR_PORT" --tp-size "$SCR_TP_SIZE" \
  --context-length "$SCR_CONTEXT_LENGTH" --mem-fraction-static "$SCR_MEM_FRACTION_STATIC" \
  --chunked-prefill-size "$SCR_CHUNKED_PREFILL_SIZE" --disable-cuda-graph \
  --dtype bfloat16 --tool-call-parser qwen3_coder --trust-remote-code
