#!/bin/bash
#SBATCH --job-name=hrm-original-sapient-L
#SBATCH --account=project_465002606
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/lumi/%x-%j.out
#SBATCH --error=logs/lumi/%x-%j.err

# HRM-Text "L" reproduction on LUMI-G (AMD MI250X / ROCm).
#
# Uses the prebuilt LUMI PyTorch-ROCm Singularity container (torch 2.7.1 +
# rocm 6.2.4, with FlashAttention 2 CK backend for gfx90a) plus a small venv
# layered on top for the few packages missing from the container
# (hydra-core, coolname).
#
# Backend: accelerator_type=rocm -> models/flash_attention_prefixlm_rocm.py,
# which calls flash_attn.flash_attn_varlen_func (FA2 CK varlen) and mirrors the
# FA4 PrefixLM split (bidirectional prefix + causal) attention structure.
#
# Verified on dev-g: single-GCD and 8-GCD FSDP smoke runs trained and
# checkpointed correctly. See wiki/pages/flashattention-b200.md.

set -euo pipefail

REPO=/project/project_465002606/HRM-Text
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
VENV=${REPO}/.venv-lumi
DATA=/scratch/project_465002606/data/sampled_original_sapient
CKPT=/scratch/project_465002606/checkpoints/original_sapient/L

# Per-project caches on scratch so they persist and are shared across nodes.
MIOPEN_CACHE=/scratch/project_465002606/cache/miopen
TRITON_CACHE=/scratch/project_465002606/cache/triton

mkdir -p "${REPO}/logs/lumi" "${CKPT}" "${MIOPEN_CACHE}" "${TRITON_CACHE}"

BIND="-B /project/project_465002606 -B /scratch/project_465002606 -B /flash/project_465002606"

# LUMI Slingshot / RCCL settings (AWS OFI plugin lives in the container env).
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export MPICH_GPU_SUPPORT_ENABLED=1

# torchrun rendezvous for single- or multi-node.
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500
GPUS_PER_NODE=8
NNODES=${SLURM_JOB_NUM_NODES:-1}

# Effective optimizer token batch. The per-rank microbatch is
# global_batch_size / world_size / gradient_accumulation_steps and must divide
# exactly (pretrain.py enforces this). With 8 GCDs and the values below the
# per-rank microbatch is 172032 / 8 / 4 = 5376 tokens. Increase
# GRAD_ACCUM if you hit OOM at 4097 max_seq_len on 64 GB/GCD.
GLOBAL_BATCH_SIZE=172032
GRAD_ACCUM=4
LR=2.5e-4

# Measured on a dev-g 8-GCD smoke: ~3.6 s/step, ~326336 total steps for this
# dataset => roughly 13-14 days of wall time on a single node. The standard-g
# walltime cap is 2 days, so this run MUST checkpoint by step and resume across
# multiple jobs (or scale to more nodes via --nodes / sbatch). Save a step
# checkpoint periodically so a requeued job can resume.
CKPT_STEP_INTERVAL=2000

# Auto-resume: find the newest step_* checkpoint under CKPT, if any.
RESUME_ARGS=""
LATEST_TAG=""
if [ -d "${CKPT}" ]; then
  LATEST_TAG=$(ls -1d "${CKPT}"/fsdp2_step_* 2>/dev/null \
    | sed 's#.*/fsdp2_##' \
    | sort -t_ -k2 -n \
    | tail -n1 || true)
fi
if [ -n "${LATEST_TAG}" ]; then
  echo "Resuming from checkpoint tag ${LATEST_TAG}"
  RESUME_ARGS="resume_checkpoint_path=${CKPT} resume_checkpoint_tag=${LATEST_TAG}"
else
  echo "No existing step checkpoint found; starting a fresh run."
fi

srun singularity exec ${BIND} "${SIF}" bash -c "
  cd ${REPO}
  source ${VENV}/bin/activate
  export PYTHONPATH=.
  export MIOPEN_USER_DB_PATH=${MIOPEN_CACHE}
  export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CACHE}
  export TRITON_CACHE_DIR=${TRITON_CACHE}
  export WANDB_MODE=\${WANDB_MODE:-offline}
  export OMP_NUM_THREADS=7

  python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --rdzv_id=\${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    pretrain.py \
      data=original_sapient \
      data.path=${DATA} \
      arch/size@arch=L \
      accelerator_type=rocm \
      distributed_strategy=fsdp \
      fwd_bwd_dtype=bfloat16 \
      lr=${LR} \
      global_batch_size=${GLOBAL_BATCH_SIZE} \
      gradient_accumulation_steps=${GRAD_ACCUM} \
      checkpoint_step_interval=${CKPT_STEP_INTERVAL} \
      +project_name='Original Sapient L HLM-torch (LUMI ROCm)' \
      +run_name=original-sapient-L-lumi \
      +checkpoint_path=${CKPT} \
      ${RESUME_ARGS}
"
