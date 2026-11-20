#!/usr/bin/env bash
# Run from an allocation with four visible GPUs on the same machine.
set -euo pipefail
cd -- "$(dirname -- "$0")"
microbatch=${TRAIN_MICRO:-32}
eval_batch=${EVAL_BATCH:-32}
python -c 'import torch; n = torch.cuda.device_count(); print("Visible CUDA devices:", n); assert n >= 4, "Four visible CUDA devices are required"'

mkdir -p runs
logdir=$(mktemp -d "runs/launcher-$(date +%Y%m%d-%H%M%S)-XXXXXX")
run_on_gpu() {
    local gpu=$1
    shift
    python -u suite.py run --device "cuda:$gpu" --microbatch "$microbatch" --eval-batch "$eval_batch" "$@"
}
pids=()
for gpu in 0 1 2; do
    (
        for task in mnli sst2; do
            run_on_gpu "$gpu" --scope main --task "$task" --seed "$gpu"
        done
    ) >"$logdir/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
(
    for task in mrpc qnli cola; do
        run_on_gpu 3 --scope all --task "$task"
    done
) >"$logdir/gpu-3.log" 2>&1 &
pids+=("$!")

echo "Logs: $logdir"
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
python suite.py list
exit "$status"
