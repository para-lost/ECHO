#!/usr/bin/env bash
# run all models in parallel (throttled)

MODELS=(anole bagel bagel_think LLM_DM gpt4o flux nano_banana gemini)
MAX_JOBS="${MAX_JOBS:-4}"          # set via env: MAX_JOBS=8 ./run.sh
PY="${PY:-python}"                 # set via env: PY=python3 ./run.sh
SCRIPT="${SCRIPT:-face_identity_metric.py}"
mkdir -p logs

for m in "${MODELS[@]}"; do
  # throttle: wait until running jobs < MAX_JOBS
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do wait -n; done
  $PY "$SCRIPT" --model "$m" >"logs/${m}.log" 2>&1 &
done
wait
echo "Done. Logs in ./logs/"
