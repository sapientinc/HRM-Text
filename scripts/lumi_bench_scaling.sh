#!/bin/bash
# Multi-node scaling benchmark for the HRM-Text L model on LUMI MI250X (ROCm).
#
# For a given node count, runs the real L model for a few optimizer steps
# against the real sampled_original_sapient dataset and records per-step
# timing via pretrain.py's max_steps benchmark path (BENCH_OUTPUT json).
#
# Usage (from a login node):
#   scripts/lumi_bench_scaling.sh <NNODES>
#
# Strong scaling: global_batch_size is fixed, so per-rank work shrinks as nodes
# grow. gradient_accumulation_steps=1 so each optimizer step is one
# fwd/bwd + reduce-scatter, isolating compute + comms scaling.

set -euo pipefail

NNODES="${1:?usage: lumi_bench_scaling.sh <NNODES>}"

REPO=/project/project_465002606/HRM-Text
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
VENV=${REPO}/.venv-lumi
# Use a small synthetic dataset by default so all ranks read trivially and we
# isolate compute + communication scaling from Lustre I/O contention (which
# dominates at 8+ nodes when cold-reading the full 712 GB dataset). Override
# with DATA=... to benchmark against the real dataset.
DATA="${DATA:-/scratch/project_465002606/data/bench_synth}"
RESULTS=/scratch/project_465002606/bench/rocm_l_scaling
MIOPEN_CACHE=/scratch/project_465002606/cache/miopen

mkdir -p "${RESULTS}" "${MIOPEN_CACHE}"

GPUS_PER_NODE=8
WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
GLOBAL_BATCH_SIZE=172032
GRAD_ACCUM=1
MAX_STEPS=22
BENCH_OUT="${RESULTS}/nodes_${NNODES}.json"

echo "=== Benchmarking ${NNODES} node(s), world_size=${WORLD_SIZE}, per-rank batch=$(( GLOBAL_BATCH_SIZE / WORLD_SIZE )) tokens ==="

# Inner per-node launcher executed by srun on each node. Written once to scratch.
# Use a per-node-count name so concurrent runs do not clobber each other.
INNER="${RESULTS}/_inner_launch_${NNODES}.sh"
cat > "${INNER}" <<EOF
#!/bin/bash
set -euo pipefail
MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n1)
BIND="-B /project/project_465002606 -B /scratch/project_465002606 -B /flash/project_465002606"
singularity exec \${BIND} ${SIF} bash -c "
  cd ${REPO}
  source ${VENV}/bin/activate
  export PYTHONPATH=.
  export WANDB_MODE=disabled
  export MIOPEN_USER_DB_PATH=${MIOPEN_CACHE}
  export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CACHE}
  export MIOPEN_FIND_MODE=FAST
  export OMP_NUM_THREADS=7
  export NNODES=${NNODES}
  export BENCH_OUTPUT=${BENCH_OUT}
  python -m torch.distributed.run \\
    --nnodes=${NNODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --node_rank=\\\${SLURM_NODEID} \\
    --master_addr=\${MASTER_ADDR} \\
    --master_port=29500 \\
    pretrain.py \\
      data=original_sapient \\
      data.path=${DATA} \\
      arch/size@arch=L \\
      accelerator_type=rocm \\
      distributed_strategy=fsdp \\
      fwd_bwd_dtype=bfloat16 \\
      compile_train_batch=false \\
      lr=2.5e-4 \\
      global_batch_size=${GLOBAL_BATCH_SIZE} \\
      gradient_accumulation_steps=${GRAD_ACCUM} \\
      max_steps=${MAX_STEPS} \\
      log_interval=1 \\
      +project_name=bench \\
      +run_name=bench-nodes-${NNODES} \\
      +checkpoint_path=${RESULTS}/ckpt_nodes_${NNODES}
"
EOF
chmod +x "${INNER}"

srun --account=project_465002606 --partition=dev-g \
  --nodes="${NNODES}" --gpus-per-node="${GPUS_PER_NODE}" \
  --ntasks-per-node=1 --cpus-per-task=56 --mem=400G --time=00:30:00 \
  "${INNER}"

echo "=== Wrote ${BENCH_OUT} ==="
python3 -c "import json; d=json.load(open('${BENCH_OUT}')); print('nodes', d['nnodes'], 'world_size', d['world_size'], 'median_step_s', round(d['median_step_seconds'],4), 'mean', round(d['mean_step_seconds'],4))" 2>/dev/null || true
