#!/usr/bin/env bash
set -u
SEEDS=${1:?seeds, e.g. "44 45 46"}
NPROC=${2:-32}
SMOKE=${3:-}
PY=${PY:-python}
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

LOGDIR=logs_parallel_quantum${SMOKE:+_smoke}
mkdir -p "$LOGDIR"

for s in $SEEDS; do for f in 0 1 2 3 4; do echo "$s $f"; done; done |
  xargs -P "$NPROC" -L1 bash -c \
    "$PY parallel_quantum_all.py worker \$0 \$1 $SMOKE > $LOGDIR/seed\$0_fold\$1.log 2>&1"

$PY parallel_quantum_all.py aggregate $SEEDS $SMOKE 2>&1 | tee "$LOGDIR/aggregate.log"
