#!/bin/sh
# usage: launch.sh NTHREADS <run_idmrg args...>   (runs detached, log in logs/)
T=$1; shift
cd "$(dirname "$0")"
name=$(echo "$*" | tr -d ' -')
OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T nohup python3 -m triakis.run_idmrg "$@" > "logs/$name.out" 2>&1 &
