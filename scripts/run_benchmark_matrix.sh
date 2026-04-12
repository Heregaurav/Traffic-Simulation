#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/lpsim"
OUT_DIR="${ROOT_DIR}/benchmarks"
OUT_CSV="${OUT_DIR}/results.csv"

# Defaults tuned for quick local runs; override as needed.
GPU_LIST="${GPU_LIST:-1}"
STEPS="${STEPS:-600}"
DT="${DT:-1.0}"
REPEATS="${REPEATS:-3}"
DATASETS="${DATASETS:-basic partial_500}"

mkdir -p "${OUT_DIR}/logs"

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}"
  echo "Build first: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
  exit 1
fi

echo "timestamp,dataset,gpus,steps,dt,run,wall_time_s,status,log_file" > "${OUT_CSV}"

ts_now() { date +"%Y-%m-%dT%H:%M:%S%z"; }

for dataset in ${DATASETS}; do
  network="${ROOT_DIR}/data/processed/${dataset}/network.csv"
  demand="${ROOT_DIR}/data/processed/${dataset}/demand.csv"
  if [[ ! -f "${network}" || ! -f "${demand}" ]]; then
    echo "Skipping dataset '${dataset}' (missing ${network} or ${demand})"
    continue
  fi

  for gpus in ${GPU_LIST}; do
    for run_id in $(seq 1 "${REPEATS}"); do
      log_file="${OUT_DIR}/logs/${dataset}_g${gpus}_r${run_id}.log"
      status="ok"

      set +e
      "${BIN}" \
        --gpus "${gpus}" \
        --network "${network}" \
        --demand "${demand}" \
        --partition balanced \
        --steps "${STEPS}" \
        --dt "${DT}" \
        > "${log_file}" 2>&1
      rc=$?
      set -e

      if [[ ${rc} -ne 0 ]]; then
        status="failed(${rc})"
      fi

      wall_time="$(sed -n 's/.*Wall time (GPU)[[:space:]]*:[[:space:]]*\([0-9.eE+-]*\).*/\1/p' "${log_file}" | tail -n1)"
      if [[ -z "${wall_time}" ]]; then
        wall_time="NA"
      fi

      echo "$(ts_now),${dataset},${gpus},${STEPS},${DT},${run_id},${wall_time},${status},${log_file}" >> "${OUT_CSV}"
      echo "[${dataset}] gpus=${gpus} run=${run_id} wall=${wall_time}s status=${status}"
    done
  done
done

echo
echo "Benchmark summary written to: ${OUT_CSV}"
echo "Logs: ${OUT_DIR}/logs"

