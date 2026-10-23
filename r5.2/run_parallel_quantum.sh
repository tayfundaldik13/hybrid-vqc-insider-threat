#!/usr/bin/env bash
set -u
SEEDS=${1:?seeds, e.g. "42 43 44"}
NPROC=${2:-32}
SMOKE=${3:-}
PY=${PY:-python}
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

LOGDIR=logs_parallel_quantum${SMOKE:+_smoke}
mkdir -p "$LOGDIR"
NST=$($PY -c "from src import config; print(len(config.DATA_SIZES))")

for st in $(seq 0 $((NST-1))); do for s in $SEEDS; do for f in 0 1 2 3 4; do echo "$s $st $f"; done; done; done |
  xargs -P "$NPROC" -L1 bash -c \
    "$PY parallel_quantum.py worker \$0 \$1 \$2 $SMOKE > $LOGDIR/seed\$0_stage\$1_fold\$2.log 2>&1"

$PY parallel_quantum.py aggregate $SEEDS $SMOKE 2>&1 | tee "$LOGDIR/aggregate.log"
