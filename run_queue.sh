#!/bin/sh
# usage: run_queue.sh jobs.txt NPAR NTHREADS
f=$1; P=${2:-4}; T=${3:-1}
cd "$(dirname "$0")"
cat "$f" | xargs -P "$P" -I{} sh -c "OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T python3 -m triakis.run_idmrg {} > \"logs/\$(echo '{}' | tr -d ' -')\".out 2>&1"
