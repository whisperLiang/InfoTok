#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# The upstream distributed initializer dlopens the unversioned libcudart.so.
# PyTorch's pip runtime ships the versioned library; keep the link in our venv.
CHP_CUDART=$(find "$(pwd)/.venv/lib" -path '*/nvidia/cuda_runtime/lib/libcudart.so.12' -print -quit)
if [ -n "$CHP_CUDART" ]; then
  mkdir -p .venv/chp_lib
  ln -sf "$CHP_CUDART" .venv/chp_lib/libcudart.so
  export LD_LIBRARY_PATH="$(pwd)/.venv/chp_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python -m torch.distributed.run --nproc_per_node=1 --rdzv_endpoint=localhost:29521 \
  -m cosmos_predict1.tokenizer.training.train --config=cosmos_predict1/tokenizer/training/configs/chp_vtok.py -- "$@"
