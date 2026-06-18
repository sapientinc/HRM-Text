# FlashAttention On B200

Last updated: 2026-06-17
Confidence: high  
Scope: Local adaptation of HRM-Text attention code for NVIDIA B200 plus accelerator backend selection.

## LUMI / AMD MI250X (ROCm) Findings, 2026-06-17

Confidence: high for live LUMI probes over `ssh lumi` (login node `uan01`,
user `frommoge`); medium for runtime GPU behavior not yet tested on a GPU node.

Context: a request to run the L training on LUMI "using FlashAttention 3"
is not viable. LUMI-G is AMD MI250X (CDNA2 / `gfx90a`, ROCm/HIP), and FA3 is
an NVIDIA Hopper (SM90) CUDA kernel. The user agreed to drop FA3 and build a
ROCm backend instead.

Live LUMI facts:

- Project `465002606` has `/scratch`, `/project`, `/flash` symlinks and GPU
  partition access to `dev-g`, `small-g`, `standard-g`, all `gpu:mi250:8`
  (8 GCDs/node). Account string for Slurm is `project_465002606`.
- Dataset path `/scratch/project_465002606/data/sampled_original_sapient/`
  is still being written: a temp `.tokens.npy.Omv9Zw` (~522 GB) was actively
  growing, `epoch_0..3` dirs exist, and `metadata.json` reports
  `max_seq_len=4097`, `total_length=14035178678`, `vocab_size=65536`. The
  final `tokens.npy` is not yet in place; training cannot start until the temp
  file is renamed/finalized.
- ROCm module: `rocm/6.3.4` (default) plus older versions.
- Prebuilt PyTorch-ROCm Singularity containers exist under
  `/appl/local/containers/sif-images/`. Newest tested:
  `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`.
- That container reports `torch 2.7.1+rocm6.2.4`, Python `3.12.11`,
  HIP `6.2.41134`, and ships `flash_attn 2.7.3` with
  `flash_attn.flash_attn_interface.flash_attn_varlen_func` available. The
  FA2 ROCm CK backend supports MI250X, so the CK varlen kernel is available
  in-container without a custom build.

Implication: add a new `rocm`/`gfx90a` accelerator backend that mirrors the
existing FA4 PrefixLM split-call structure but calls FA2 `flash_attn_varlen_func`.
PyTorch-ROCm exposes AMD GPUs as `torch.cuda` (`device.type == "cuda"`) and uses
RCCL via `backend="nccl"`, so the device/memory/FSDP plumbing mostly carries over;
the CUDA-capability checks in `models/accelerator.py` need a HIP-aware branch.

## LUMI ROCm Backend Implemented + Verified, 2026-06-17

Confidence: high for live GPU-node runs on LUMI dev-g (MI250X).

A new `rocm` accelerator backend was added and verified end-to-end on LUMI:

- `models/accelerator.py`: `AcceleratorType` now includes `"rocm"`; HIP-aware
  availability check (`torch.version.hip is not None and torch.cuda.is_available()`);
  treated as CUDA-like for device/memory/sync (`device.type == "cuda"` under ROCm).
- `models/flash_attention_prefixlm_rocm.py` (new): PrefixLM split (bidirectional
  prefix + causal) mirroring the FA4 module, calling FA2
  `flash_attn.flash_attn_varlen_func` (ROCm CK varlen kernel).
- `models/flash_attention_prefixlm_dispatch.py`: added `case "rocm"`.
- `pretrain.py`: `pin_memory` now includes `rocm`; the distributed guard already
  allows `rocm` (NCCL backend maps to RCCL on ROCm). `apply_fsdp` was made
  torch-version-adaptive: `set_gradient_divide_factor` (newer torch on B200) vs.
  `set_reduce_scatter_divide_factor` (torch 2.7.1 on LUMI), and
  `set_force_sum_reduction_for_comms` is now optional via `hasattr`. This was
  required because LUMI's torch 2.7.1 lacks the newer FSDP2 method names.

Verified on LUMI (container
`lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`, torch `2.7.1+rocm6.2.4`):

- `flash_attn_varlen_func` bf16 forward+backward on `AMD Instinct MI250X` is finite.
- Tiny `pretrain.py` epoch on 1 GCD with `accelerator_type=rocm` + FSDP + bf16
  completes and writes a sharded checkpoint.
- Tiny `pretrain.py` epoch on 8 GCDs (full node) completes; 8 RCCL ranks shard
  and write 8 `.distcp` shards + 8 carry files.
- Real L config (24 layers, hidden 1280) on 8 GCDs builds with no OOM at
  `global_batch_size=172032`, `gradient_accumulation_steps=4` (per-rank
  microbatch 5376 tokens). Observed `~3.6 s/step` over 308 steps; total schedule
  is `326336` steps for `data/sampled_original_sapient`
  (`total_length=14,035,178,678`, `max_seq_len=4097`). Estimated full run
  `~13-14 days` on one node -> must checkpoint-by-step and resume across jobs, or
  scale to more nodes.

Operational notes:

- Singularity needs explicit binds:
  `-B /project/project_465002606 -B /scratch/project_465002606 -B /flash/project_465002606`.
- The container lacks `hydra-core` and `coolname`; a venv with
  `--system-site-packages` at `${REPO}/.venv-lumi` supplies them while reusing the
  container's torch + flash_attn. Launch with `python -m torch.distributed.run`
  (the venv has no `torchrun` shim).
- Set `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` to a writable scratch path.
- `adam_atan2` is a vendored local module; no pip dependency needed.
- Data location: `/scratch/project_465002606/data/sampled_original_sapient`
  (finalized `tokens.npy` ~712 GB present as of 2026-06-17). Account
  `project_465002606`; GPU partitions `dev-g`, `small-g`, `standard-g` (mi250:8).

Runnable recipe: `scripts/lumi_train_original_sapient_l.sbatch` (auto-resumes
from the newest `step_*` checkpoint; `checkpoint_step_interval=2000`).

FlashAttention 3 was NOT used: it is NVIDIA Hopper-only and cannot run on MI250X.

## Multi-node Scaling, 2026-06-18

Confidence: high for live dev-g runs across 2/4/8/16 nodes.

Strong-scaling benchmark of the L model with a fixed `global_batch_size=172032`
tokens, `gradient_accumulation_steps=1`, bf16, FSDP2/RCCL, run via
`scripts/lumi_bench_scaling.sh` (adds an optional `max_steps` config that records
per-step times to `BENCH_OUTPUT`). A small synthetic dataset (1024-token rows)
was used to isolate compute + communication from Lustre I/O.

| Nodes | GCDs | tokens/GCD/step | median s/step | speedup vs 2-node | parallel eff |
|---|---|---|---|---|---|
| 2  | 16  | 10752 | 1.499 | 1.00x | 100% |
| 4  | 32  | 5376  | 0.887 | 1.69x | 85%  |
| 8  | 64  | 2688  | 0.660 | 2.27x | 57%  |
| 16 | 128 | 1344  | 0.645 | 2.32x | 29%  |

Findings:

- Scaling saturates by 8 nodes; 8->16 nodes gives almost no speedup because the
  fixed global batch leaves only ~1344 tokens/GCD at 16 nodes, so FSDP
  all-gather/reduce-scatter and fixed per-step overhead dominate. 16 nodes wastes
  roughly half the GCDs for this batch size.
- For real multi-node training, scale the GLOBAL batch with node count (weak
  scaling) and/or raise `gradient_accumulation_steps` so each GCD keeps a useful
  microbatch.
- Two benchmark gotchas learned: (1) synthetic benchmark rows must be SHORTER
  than the per-GCD token budget `global_batch_size / world_size`, or the sampler
  produces no batches at high node counts (1344 at 16 nodes); (2) keep all
  benchmark/checkpoint writes on `/scratch`, never `/projappl` (54 GB quota -
  a stray relative `checkpoints/` dir filled it and crashed an 8-node run).
- Results saved on LUMI:
  `/scratch/project_465002606/bench/rocm_l_scaling/SCALING_RESULTS.md` and
  `scaling_summary.json`.

## Decision

Use FlashAttention 4, not FlashAttention 3, for B200/SM100.

Update on 2026-05-25: training now has an explicit accelerator selector:

```text
accelerator_type: sm90 | sm100 | mps | cpu | none
```

- `sm90`: restores the original H100/Hopper FA3 PrefixLM implementation from git commit `00b4fe5`, using `flash_attn_3`, direct `torch.ops.flash_attn_3.fwd`, and the custom PrefixLM forward/backward torch library ops.
- `sm100`: uses the current FA4/CUTE path from `flash_attn.cute.flash_attn_varlen_func`.
- `mps`: uses a dense PyTorch SDPA PrefixLM path and single-process training without NCCL/FSDP.
- `cpu` / `none`: uses the same dense PyTorch SDPA PrefixLM path on CPU and single-process training without NCCL/FSDP.

Confidence: high for the inspected git history and local dense-path smoke tests.

## Why

- FA3/Hopper source and wheels were tried and did not produce a viable B200 runtime.
- Local experiments hit kernel/runtime issues and then CUTE/WGMMA architecture macro failures when trying to force SM100 behavior through the Hopper path.
- FA4 has explicit Blackwell/SM100 code paths under `flash_attn/cute`.

## Installed Dependency

Accelerator dependencies are now optional rather than unconditional base requirements:

```text
CPU:       pyproject extra cpu
MPS:       pyproject extra mps
SM90/H100: requirements-sm90.txt / pyproject extra sm90
SM100/B200: requirements-sm100.txt / pyproject extra sm100
```

Base `pyproject.toml` dependencies no longer install `torch`, FlashAttention, `vllm`, or server dependencies unconditionally. Use accelerator extras for the runtime backend and separate `eval` / `server` extras for evaluation and API serving. Base `requirements.txt` no longer installs a FlashAttention package, so Apple/MPS environments do not try to build CUDA extensions.

## Code Changes

- `models/accelerator.py`
  - owns the process-local accelerator selector.
- `models/flash_attention_prefixlm_v2.py`
  - keeps the original SM90/H100 FA3 PrefixLM implementation from commit `00b4fe5` in the historical public module.
  - adds only a thin accelerator check at the public `flash_attn_varlen_prefixlm` entrypoint; non-SM90 backends are delegated to `models/flash_attention_prefixlm_dispatch.py`.
  - keeps the `compute_aux_seq_tensors_scalars` entrypoint used by the dataset.
- `models/flash_attention_prefixlm_dispatch.py`
  - owns accelerator dispatch for SM100, MPS, CPU, and dense fallback implementations.
- `models/flash_attention_prefixlm_fa4.py`
  - owns the SM100/B200 FlashAttention 4/CUTE PrefixLM implementation.
- `models/flash_attention_prefixlm_dense.py`
  - owns the dense PyTorch SDPA PrefixLM fallback used by `mps`, `cpu`, and `none` when the Metal kernel does not support the current tensors.
- `models/flash_attention_prefixlm_common.py`
  - owns shared PrefixLM sequence metadata unpacking, active tensor slicing, shifted cu-seqlens construction, and sequence-index construction.

- `models/layers.py`
  - removed `flash_attn_with_kvcache`
  - cache path updates key/value tensors and uses `torch.nn.functional.scaled_dot_product_attention`
- `pretrain.py`
  - uses `accelerator_type` to select device, disable CUDA-only distributed/FSDP paths for MPS/CPU, and avoid torch.compile when requested.
  - supports `gradient_accumulation_steps`; `global_batch_size` remains the effective optimizer token batch, and the physical per-rank microbatch is `global_batch_size / world_size / gradient_accumulation_steps`. The code raises an error if this division is not exact.

## Verification

Earlier verified:

- `python -m py_compile models/flash_attention_prefixlm_v2.py models/layers.py`
- FA4 B200 forward smoke test
- PrefixLM forward/backward CUDA smoke test
- cache attention CUDA smoke test
- imports for `pretrain`, `simple_inference_engine`, and `evaluation.engines`

Refactor verification on 2026-05-26:

```bash
python -m py_compile models/flash_attention_prefixlm_common.py models/flash_attention_prefixlm_v2.py models/flash_attention_prefixlm_fa4.py models/flash_attention_prefixlm_dense.py models/flash_attention_prefixlm_mps.py
```

Also verified a tiny CPU dense PrefixLM forward/backward smoke through `models.flash_attention_prefixlm_v2.flash_attn_varlen_prefixlm`; outputs and gradients were finite. Confidence: high.

After regenerating `/private/tmp/hrm_tiny_sampled`, a one-step `scripts/debug_nan_training_step.py` CPU smoke also completed with finite loss, metric tensors, gradients, parameters, and post-optimizer parameters. Confidence: high.

Later on 2026-05-26, the SM100/FA4 implementation moved out of `models/flash_attention_prefixlm_v2.py` into `models/flash_attention_prefixlm_fa4.py`, and the dense fallback moved into `models/flash_attention_prefixlm_dense.py`.

Refactor update on 2026-05-29: accelerator selection moved from `models/flash_attention_prefixlm_v2.py` into `models/flash_attention_prefixlm_dispatch.py`. `v2` now remains a stable public compatibility module for `dataset_new.py`, `models/layers.py`, and debug scripts, while backend policy lives in the dispatcher. Verified with:

```bash
python -m py_compile models/flash_attention_prefixlm_v2.py models/flash_attention_prefixlm_dispatch.py models/flash_attention_prefixlm_fa4.py models/flash_attention_prefixlm_dense.py
```

Also verified a tiny CPU dense PrefixLM forward/backward smoke through `models.flash_attention_prefixlm_v2.flash_attn_varlen_prefixlm`; outputs and gradients were finite. Confidence: high.

SM90 layout update on 2026-05-29: the original H100/FA3 implementation was moved back into `models/flash_attention_prefixlm_v2.py` to minimize drift from commit `00b4fe5`. The separate `models/flash_attention_prefixlm_fa3.py` backend file was removed. `v2` now dispatches to `models.flash_attention_prefixlm_dispatch.flash_attn_varlen_prefixlm` only when `accelerator_type != "sm90"`. The FA3 import is lazy-safe so CPU/MPS/SM100 environments without `flash_attn_interface` can still import `v2`; attempting the SM90 path without FlashAttention 3 raises an explicit ImportError. Verified with:

```bash
python -m py_compile models/flash_attention_prefixlm_v2.py models/flash_attention_prefixlm_dispatch.py models/flash_attention_prefixlm_fa4.py models/flash_attention_prefixlm_dense.py models/flash_attention_prefixlm_mps.py
```

Also verified a tiny CPU dense PrefixLM forward/backward smoke through the `v2` public entrypoint; outputs and gradients were finite. Confidence: high.

MPS dispatch update on 2026-05-29: the Metal PrefixLM path no longer uses environment variables for opt-in or shape caps. For `accelerator_type=mps`, the dispatcher now chooses the Metal kernel dynamically when the actual attention tensors are on MPS, are `float32`, and have matching `q/k/v` shapes; otherwise it falls back to dense SDPA. The raw MPS kernel still validates dtype and shape internally. Debug scripts compare dense vs custom by temporarily switching the process-local `accelerator_type`, not by mutating environment variables. Confidence: high.

MPS bf16 update on 2026-05-26: `pretrain.py` no longer rewrites `fwd_bwd_dtype=bfloat16` to `float32` for `mps`/`cpu`/`none`. Instead, `TrainState` records the resolved forward/backward dtype and non-CUDA train steps use `torch.autocast(device_type=..., dtype=...)` when that dtype is not float32. This keeps parameters in fp32 while allowing bf16 forward/backward activations on supported CPU/MPS operations. Verified outside the sandbox on the M2 Max with `torch==2.13.0.dev20260524`: bf16 MPS matmul works, dense PrefixLM bf16 forward/backward on `mps:0` works, and a 3-step MPS training smoke with `fwd_bwd_dtype=bfloat16` completed with finite losses, metric tensors, gradients, parameters, and post-optimizer parameters. Confidence: high for the tiny smoke; larger XS/S/B runs still need memory/performance validation.

MPS memory diagnostic update on 2026-05-26: a user float32 run grew from about `18 GB` to `44 GB` Activity Monitor memory by step 33. Because this happened in float32, bf16/autocast caching is not the primary explanation for that run. `pretrain.py` initially supported `mps_memory_log_interval` and `mps_empty_cache_interval` config fields; those names are now superseded by backend-neutral `memory_log_interval` and `empty_cache_interval`. If MPS `allocated` stays bounded while `reserved` rises, this is likely MPS allocator/driver high-water or fragmentation from dense attention temporaries; if `allocated` rises monotonically after zero-grad, investigate real tensor retention. Use `empty_cache_interval=10` or `25` as a mitigation/experiment, expecting some throughput cost. Confidence: medium until a long real XS/S/B run reports allocated-vs-reserved numbers.

Memory diagnostic update on 2026-05-29: memory logging and cache clearing were generalized across supported backends. Use `memory_log_interval` and `empty_cache_interval`; the older `mps_memory_log_interval` and `mps_empty_cache_interval` aliases were removed. Backend behavior:

```text
CUDA / sm90 / sm100:
  memory: allocated, reserved, max_allocated, max_reserved from torch.cuda
  empty cache: torch.cuda.empty_cache()

MPS:
  memory: allocated/current and reserved/driver from torch.mps
  empty cache: torch.mps.empty_cache()

CPU / none:
  memory: process RSS
  empty cache: no-op
```

The shared helpers live in `models/accelerator.py` as `memory_stats_for_device`, `synchronize_device`, and `empty_accelerator_cache`. `scripts/debug_nan_training_step.py` now uses the same backend-neutral memory reporting. Verified with `py_compile`, Hydra override composition for the new interval names, and a one-step CPU diagnostic on `/private/tmp/hrm_tiny_sampled` showing sane RSS values around `402-485 MiB`. Confidence: high.

Verified locally on 2026-05-25:

```bash
python -m py_compile models/accelerator.py models/flash_attention_prefixlm_fa3.py models/flash_attention_prefixlm_v2.py models/layers.py models/lm_head.py models/baselines/hrm_nocarry_bp_warmup.py pretrain.py scripts/debug_nan_training_step.py scripts/create_tiny_sampled_dataset.py
python scripts/create_tiny_sampled_dataset.py /private/tmp/hrm_tiny_sampled --rows 96 --epochs 1 --vocab-size 512 --inst-len 5 --resp-len 11
python scripts/debug_nan_training_step.py --steps 3 --allow-mps-cpu-fallback --override data.path=/private/tmp/hrm_tiny_sampled --override accelerator_type=mps --override compile_train_batch=false --override fwd_bwd_dtype=float32 --override global_batch_size=64 --override epochs=1 --override lr_warmup_steps=1 --override ema=null --override arch.n_layers=2 --override arch.hidden_size=64 --override arch.num_heads=4 --override arch.expansion=2 --override arch.half_layers=false --override arch.H_cycles=1 --override arch.L_cycles=1 --override +arch.bp_min_steps=1 --override arch.bp_max_steps=1
```

The 3-step diagnostic produced finite losses, metric tensors, gradients, and post-optimizer parameters. Inside the normal command sandbox, PyTorch reported `torch.backends.mps.is_built() == True` but `torch.backends.mps.is_available() == False`, so the first diagnostic used the explicit CPU fallback while still selecting the `mps` attention backend. Confidence: high for dense backend execution.

Fresh conda env update, 2026-05-25:

```bash
conda env remove -y -n hrm
conda create -y -n hrm python=3.13
conda run -n hrm uv pip install torch
conda run -n hrm uv pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
conda run -n hrm uv pip install hydra-core einops numba coolname wandb pydantic numpy pyyaml tqdm
```

The fresh `hrm` env is Python `3.13.13` on `osx-arm64`. Stable PyTorch installed as `torch==2.12.0`; the nightly upgrade installed `torch==2.13.0.dev20260524`.

Inside the sandbox, both builds report:

```text
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == False
torch.mps.device_count() == 0
```

Outside the sandbox, the same fresh `hrm` env with nightly PyTorch reports:

```text
torch==2.13.0.dev20260524
torch.backends.mps.is_built() == True
torch.backends.mps.is_available() == True
torch.mps.device_count() == 1
torch.ones(2, device="mps") + 1 -> tensor([2., 2.], device='mps:0')
```

The machine itself reports Apple M2 Max graphics with Metal support via `system_profiler SPDisplaysDataType`. Conclusion: the earlier MPS blocker was sandbox device visibility, not repo code, absent hardware, or the conda env.

The new explicit CPU backend was verified in the fresh `hrm` env:

```bash
conda run -n hrm python -m py_compile models/accelerator.py models/flash_attention_prefixlm_fa3.py models/flash_attention_prefixlm_v2.py models/layers.py models/lm_head.py models/baselines/hrm_nocarry_bp_warmup.py pretrain.py scripts/debug_nan_training_step.py scripts/create_tiny_sampled_dataset.py
conda run -n hrm python scripts/create_tiny_sampled_dataset.py /private/tmp/hrm_tiny_sampled --rows 96 --epochs 1 --vocab-size 512 --inst-len 5 --resp-len 11
conda run -n hrm python scripts/debug_nan_training_step.py --steps 3 --override data.path=/private/tmp/hrm_tiny_sampled --override accelerator_type=cpu --override compile_train_batch=false --override fwd_bwd_dtype=float32 --override global_batch_size=64 --override epochs=1 --override lr_warmup_steps=1 --override ema=null --override arch.n_layers=2 --override arch.hidden_size=64 --override arch.num_heads=4 --override arch.expansion=2 --override arch.half_layers=false --override arch.H_cycles=1 --override arch.L_cycles=1 --override +arch.bp_min_steps=1 --override arch.bp_max_steps=1
```

Result: 3 training steps completed with finite losses, metrics, gradients, and post-optimizer parameters. Confidence: high.

Actual MPS verification, run outside the sandbox on 2026-05-25:

```bash
conda run -n hrm python scripts/debug_nan_training_step.py --steps 3 --override data.path=/private/tmp/hrm_tiny_sampled --override accelerator_type=mps --override compile_train_batch=false --override fwd_bwd_dtype=float32 --override global_batch_size=64 --override epochs=1 --override lr_warmup_steps=1 --override ema=null --override arch.n_layers=2 --override arch.hidden_size=64 --override arch.num_heads=4 --override arch.expansion=2 --override arch.half_layers=false --override arch.H_cycles=1 --override arch.L_cycles=1 --override +arch.bp_min_steps=1 --override arch.bp_max_steps=1
```

Result: 3 training steps completed on `mps:0` with finite losses, metric tensors, gradients, and post-optimizer parameters. Confidence: high.

Gradient accumulation verification, run outside the sandbox on 2026-05-25:

```bash
conda run -n hrm python scripts/debug_nan_training_step.py \
  --steps 1 \
  --override data.path=data/sampled_original_sapient_partial_smoke \
  --override arch/size@arch=B \
  --override accelerator_type=mps \
  --override compile_train_batch=false \
  --override fwd_bwd_dtype=float32 \
  --override global_batch_size=131072 \
  --override gradient_accumulation_steps=8 \
  --override epochs=4 \
  --override lr=2.5e-4 \
  --override lr_warmup_steps=50 \
  --override ema=null
```

Result: one effective optimizer step completed with `local_microbatch_size=16384`, eight accumulated microbatches, finite loss, finite metrics, finite gradients, and finite post-optimizer parameters. This avoids the MPS dense-attention OOM seen when trying to run `global_batch_size=131072` as one physical dense-attention batch. Confidence: high.

Small model-size configs added on 2026-05-25:

```text
S:          n_layers=8, hidden_size=768, num_heads=6
XS:         n_layers=6, hidden_size=512, num_heads=4
XXS:        n_layers=6, hidden_size=256, num_heads=2
XXS_wide:   n_layers=4, hidden_size=384, num_heads=3
```

All new configs keep 128-dimensional attention heads and even layer counts for HRM `half_layers: true`. Hydra composition was verified for all four sizes, and a one-step MPS `XXS` diagnostic on `data/sampled_original_sapient_partial_smoke` completed with finite loss, metrics, gradients, and post-optimizer parameters. User-reported follow-up XXS run used `global_batch_size=16384` with `gradient_accumulation_steps=4`. Confidence: high for local diagnostic verification; medium for the user-reported follow-up setting.

## Residual Risk

The MPS path uses dense SDPA, so large physical microbatches can still OOM even when the effective batch size is valid. Use gradient accumulation to keep the physical microbatch within the tested range.

Update on 2026-05-26: `models/accelerator.py` now owns runtime accelerator availability validation, not just name-to-device mapping. `sm90`/`sm100` require `torch.cuda.is_available()`, a valid local rank, and matching CUDA major capability `9.x`/`10.x`; `mps` requires `torch.backends.mps.is_available()`; `cpu` and `none` resolve to CPU and are always valid. The debug training script can still bypass validation for its explicit `--allow-mps-cpu-fallback` development mode. Verified by `py_compile` on `models/accelerator.py`, `pretrain.py`, and `scripts/debug_nan_training_step.py`, plus CPU/none helper checks. Confidence: high.

Update later on 2026-05-25: a first custom Metal/MPS PrefixLM attention backend was added at `models/flash_attention_prefixlm_mps.py`. It is now opt-in only via:

```bash
HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1
```

Default `accelerator_type=mps` uses the safer dense PyTorch fallback. The prototype streams exact PrefixLM attention without materializing dense attention matrices or masks. It is a correctness/memory prototype, not yet a tiled FlashAttention-class kernel.

Verified numeric parity against the dense fallback on a tiny hand-built PrefixLM batch:

```text
forward max abs diff: 4.77e-7
dq max abs diff:      2.21e-6
dk max abs diff:      2.38e-6
dv max abs diff:      1.91e-6
```

Verified model-level smoke:

```bash
/Users/petersk/Nobackup/miniconda3/bin/conda run -n hrm python scripts/debug_nan_training_step.py \
  --steps 1 \
  --override data.path=data/sampled_original_sapient_partial_smoke \
  --override arch/size@arch=XXS \
  --override accelerator_type=mps \
  --override compile_train_batch=false \
  --override fwd_bwd_dtype=float32 \
  --override global_batch_size=1024 \
  --override gradient_accumulation_steps=1 \
  --override epochs=1 \
  --override lr=2.5e-4 \
  --override lr_warmup_steps=10 \
  --override ema=null \
  --override arch.H_cycles=1 \
  --override arch.L_cycles=1 \
  --override +arch.bp_min_steps=1 \
  --override arch.bp_max_steps=1
```

Result: one MPS training step completed with finite loss, metrics, gradients, and post-optimizer parameters. The larger default-cycle `XXS` probe at `global_batch_size=4096` did not finish quickly; likely cause is the prototype kernel's untiled per-query/per-key loops. A follow-up attempt to vectorize the kernel with threadgroup reductions was unstable enough to lock up the machine during testing, so the experimental kernel must remain opt-in until it is redesigned and tested in much smaller standalone kernels. Confidence: high for tiny correctness probes, low for performance readiness.

Safety update on 2026-05-25: the experimental MPS kernel now refuses large shapes even when `HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1` is set. Default caps are:

```text
HRM_EXPERIMENTAL_MPS_MAX_TOKENS=256
HRM_EXPERIMENTAL_MPS_MAX_SEQS=8
HRM_EXPERIMENTAL_MPS_MAX_HEADS=4
HRM_EXPERIMENTAL_MPS_MAX_HEAD_DIM=64
```

Only raise these caps in isolated kernel tests, not full training runs. A tiny standalone parity harness was added:

Superseded on 2026-05-29: this environment-variable cap mechanism was removed. Current MPS dispatch is tensor-driven as described in the 2026-05-29 MPS dispatch update above. Confidence: high.

```bash
/Users/petersk/Nobackup/miniconda3/bin/conda run -n hrm python scripts/debug_mps_prefixlm_kernel.py --seqs 2 --prefix-len 4 --causal-len 4 --heads 2 --head-dim 32
```

Verified outside the sandbox on MPS:

```text
shape: tokens=4 seqs=1 heads=1 head_dim=16 causal=False
forward max abs diff: 2.38419e-07
dq max abs diff: 8.84756e-09
dk max abs diff: 6.51926e-09
dv max abs diff: 1.49012e-08

shape: tokens=16 seqs=2 heads=2 head_dim=32 causal=False
forward max abs diff: 4.76837e-07
dq max abs diff: 1.39698e-09
dk max abs diff: 1.39698e-09
dv max abs diff: 1.39698e-09
```

The tiny parity harness now also reports best-of-N forward+backward timings and one-pass memory deltas, using MPS synchronization around each timed/measured iteration. Default timing is best of 10 after 2 warmup iterations. This PyTorch MPS build exposes `current_allocated_memory()` and `driver_allocated_memory()`, but not CUDA-style peak memory stats, so memory output is a synchronized before/after delta rather than a true peak. The 16-token run above measured:

```text
dense forward+backward best of 10: 4.081 ms
kernel forward+backward best of 10: 0.944 ms
kernel/dense ratio: 0.231x
dense forward+backward memory: current_delta=0.027 MiB driver_delta=8.016 MiB current_after=0.096 MiB driver_after=18.656 MiB
kernel forward+backward memory: current_delta=0.027 MiB driver_delta=0.000 MiB current_after=0.096 MiB driver_after=10.641 MiB
kernel/dense current-memory ratio: 1.000x
kernel/dense driver-memory ratio: 0.000x
```

At this tiny shape, current-memory deltas are dominated by the retained tensors returned by the harness. Driver deltas are useful as an early signal of extra backend workspace, but larger isolated shapes are needed before drawing performance or memory conclusions. Confidence: high for the new tiny-kernel guard, parity harness, and tiny-shape timing/memory output.

First cautious XXS-geometry ramp test, also run outside the sandbox on 2026-05-25:

```bash
HRM_EXPERIMENTAL_MPS_MAX_TOKENS=512 \
HRM_EXPERIMENTAL_MPS_MAX_SEQS=4 \
HRM_EXPERIMENTAL_MPS_MAX_HEADS=2 \
HRM_EXPERIMENTAL_MPS_MAX_HEAD_DIM=128 \
/Users/petersk/Nobackup/miniconda3/bin/conda run -n hrm python scripts/debug_mps_prefixlm_kernel.py \
  --seqs 4 \
  --prefix-len 32 \
  --causal-len 96 \
  --heads 2 \
  --head-dim 128 \
  --timing-iterations 3 \
  --warmup-iterations 1
```

Result:

```text
shape: tokens=512 seqs=4 heads=2 head_dim=128 causal=False
forward max abs diff: 1.2219e-06
dq max abs diff: 5.00222e-11
dk max abs diff: 7.27596e-11
dv max abs diff: 1.09139e-11
dense forward+backward best of 3: 5.482 ms
kernel forward+backward best of 3: 3.144 ms
kernel/dense ratio: 0.573x
dense forward+backward memory: current_delta=3.500 MiB driver_delta=8.016 MiB current_after=12.002 MiB driver_after=26.406 MiB
kernel forward+backward memory: current_delta=3.500 MiB driver_delta=0.000 MiB current_after=12.002 MiB driver_after=18.391 MiB
```

Confidence: high for the 512-token isolated benchmark result.

MPS kernel benchmark harness update on 2026-05-25:

- `scripts/debug_mps_prefixlm_kernel.py` now reports granular timings:
  - dense forward-only
  - kernel forward-only
  - tiled `head_dim=128` forward-only
  - dense backward-only
  - kernel backward-only
  - dense forward+backward
  - kernel forward+backward
- `models/flash_attention_prefixlm_mps.py` has a benchmark-only tiled forward kernel specialized for `head_dim=128`, processing 4 queries per threadgroup. This is not used in the autograd/training path.
- A q8 variant was briefly tested and was slower on the 512-token shape, so the retained tiled benchmark path is q4.

Best-of-10 XXS-geometry result:

```text
shape: tokens=512 seqs=4 heads=2 head_dim=128 causal=False
forward max abs diff: 1.2219e-06
dq max abs diff: 5.00222e-11
dk max abs diff: 7.27596e-11
dv max abs diff: 1.09139e-11
forward-only kernel max abs diff: 1.43051e-06
forward-only tiled_q4_hdim128 max abs diff: 1.43051e-06
dense forward-only best of 10: 4.427 ms
kernel forward-only best of 10: 2.456 ms
tiled_q4_hdim128 forward-only best of 10: 2.195 ms
tiled/dense forward-only ratio: 0.496x
tiled/kernel forward-only ratio: 0.894x
kernel/dense forward-only ratio: 0.555x
dense backward-only best of 10: 1.668 ms
kernel backward-only best of 10: 1.698 ms
kernel/dense backward-only ratio: 1.018x
dense forward+backward best of 10: 6.927 ms
kernel forward+backward best of 10: 3.424 ms
kernel/dense ratio: 0.494x
dense forward+backward memory: current_delta=3.500 MiB driver_delta=16.016 MiB current_after=13.502 MiB driver_after=34.656 MiB
kernel forward+backward memory: current_delta=3.500 MiB driver_delta=0.000 MiB current_after=13.502 MiB driver_after=18.641 MiB
```

Interpretation: q4 tiling improves forward-only time by roughly 11% versus the existing one-query custom forward on this shape, but the full training-path gap is currently constrained by backward. The custom backward is about equal to dense backward at 512 tokens but does not yet use tiled work sharing. Confidence: high for this isolated benchmark.

Additional ramp on 2026-05-25:

`scripts/debug_mps_prefixlm_kernel.py` now has `--forward-only`, which skips backward parity/timing so larger forward candidates can be tested without the known slow custom backward dominating runtime.

```text
1024 tokens, 8 seqs x 128 tokens, heads=2, head_dim=128:
dense forward-only best of 5: 7.666 ms
kernel forward-only best of 5: 2.277 ms
tiled_q4_hdim128 forward-only best of 5: 2.692 ms
dense backward-only best of 5: 2.346 ms
kernel backward-only best of 5: 2.594 ms
dense forward+backward best of 5: 11.340 ms
kernel forward+backward best of 5: 4.986 ms

2048 tokens, 16 seqs x 128 tokens, heads=2, head_dim=128:
dense forward-only best of 3: 15.469 ms
kernel forward-only best of 3: 3.671 ms
tiled_q4_hdim128 forward-only best of 3: 4.826 ms
dense backward-only best of 3: 4.142 ms
kernel backward-only best of 3: 4.487 ms
dense forward+backward best of 3: 23.458 ms
kernel forward+backward best of 3: 9.056 ms

4096 tokens, 16 seqs x 256 tokens, heads=2, head_dim=128:
dense forward-only best of 5: 16.973 ms
kernel forward-only best of 5: 15.398 ms
tiled_q4_hdim128 forward-only best of 5: 20.431 ms

8192 tokens, 16 seqs x 512 tokens, heads=2, head_dim=128:
dense forward-only best of 3: 20.453 ms
kernel forward-only best of 3: 67.972 ms
tiled_q4_hdim128 forward-only best of 3: 83.531 ms
```

Conclusion: the current custom kernel is useful only as a correctness/memory prototype for short sequences. The q4 tiled-forward experiment shares K loads across query rows, but it is not a real FlashAttention-style key/value block tile; the extra barriers outweigh the reuse at realistic context lengths. The next viable MPS kernel must tile over key/value blocks and maintain online softmax state (`m`, `l`, and output accumulators) per query, then redesign backward around the same block structure. Confidence: high for the measured ramp; confidence: medium for the specific next-kernel design direction.

Online-softmax forward update on 2026-05-25:

- Added `prefixlm_forward_online_hdim128`, a forward-only MPS kernel specialized for `head_dim=128`.
- The experimental autograd path now uses this online forward for `head_dim=128`; backward still uses the previous custom backward kernels.
- The harness also reports an estimated `online-forward + dense-backward` best case by adding separately timed online forward and dense backward. This is not an implemented autograd path; it is a performance ceiling for prioritization.

Forward-only results:

```text
512 tokens, 4 seqs x 128:
dense forward-only best of 3: 4.425 ms
old kernel forward-only best of 3: 1.570 ms
online_hdim128 forward-only best of 3: 0.842 ms

2048 tokens, 16 seqs x 128:
dense forward-only best of 3: 19.049 ms
old kernel forward-only best of 3: 4.732 ms
online_hdim128 forward-only best of 3: 1.570 ms

4096 tokens, 16 seqs x 256:
dense forward-only best of 3: 17.525 ms
old kernel forward-only best of 3: 17.122 ms
online_hdim128 forward-only best of 3: 4.591 ms

8192 tokens, 16 seqs x 512, online-only because the old streaming forward produced a bad outlier:
dense forward-only best of 3: 19.746 ms
online_hdim128 forward-only best of 3: 16.319 ms
```

Full 4096-token result after using online forward inside the experimental autograd path:

```text
shape: tokens=4096 seqs=16 heads=2 head_dim=128 causal=False
forward max abs diff: 1.07288e-06
dq max abs diff: 7.7307e-12
dk max abs diff: 1.18234e-11
dv max abs diff: 1.19371e-12
dense forward-only best of 3: 16.330 ms
kernel forward-only best of 3: 4.420 ms
online_hdim128 forward-only best of 3: 4.325 ms
dense backward-only best of 3: 4.595 ms
kernel backward-only best of 3: 16.600 ms
estimated online-forward+dense-backward best-case: 8.920 ms
dense forward+backward best of 3: 22.918 ms
kernel forward+backward best of 3: 21.482 ms
kernel/dense ratio: 0.937x
```

Interpretation: online forward is now the right forward design and is substantially faster at realistic 4096-token XXS geometry. The remaining blocker is backward; the existing custom backward is about 3.6x slower than dense backward at 4096. Confidence: high.

Backward update on 2026-05-25:

- Added `head_dim=128` specialized backward kernels for the experimental MPS path.
- The specialized kernels reduce the QK dot product and softmax-gradient dot product in one reduction loop, reducing threadgroup barrier overhead.
- Added a precomputed per-query `sum(dO * O)` scalar, then use `sum(dO * V_j) - sum(dO * O)` inside `dq` and `dk/dv`.
- For PrefixLM, the pre-dot kernels also compute exact query/key loop bounds instead of scanning the full sequence and checking the mask for every pair.

512-token result:

```text
dense forward-only best of 5: 3.840 ms
kernel forward-only best of 5: 0.886 ms
online_hdim128 forward-only best of 5: 0.857 ms
dense backward-only best of 5: 1.216 ms
kernel backward-only best of 5: 1.132 ms
dense forward+backward best of 5: 5.535 ms
kernel forward+backward best of 5: 1.951 ms
kernel/dense ratio: 0.353x
```

4096-token XXS-geometry result:

```text
dense forward-only best of 3: 15.485 ms
kernel forward-only best of 3: 4.406 ms
online_hdim128 forward-only best of 3: 4.423 ms
dense backward-only best of 3: 4.327 ms
kernel backward-only best of 3: 10.206 ms
dense forward+backward best of 3: 22.020 ms
kernel forward+backward best of 3: 14.641 ms
kernel/dense ratio: 0.665x
```

Interpretation: the specialized backward improved the realistic 4096-token full path from roughly parity/slight win (`0.937x`) to `0.665x`, or about 1.5x faster than dense. Backward is still the bottleneck and remains about 2.4x slower than dense backward at this shape. Confidence: high for the measured isolated benchmark.

XL-geometry experiment on 2026-05-25:

XL uses `heads=12`, `head_dim=128`. A 4096-token isolated benchmark with `16` sequences of length `256` showed the current custom MPS kernel is not competitive for XL-width attention:

```text
shape: tokens=4096 seqs=16 heads=12 head_dim=128 causal=False
dense forward-only best of 3: 20.089 ms
kernel forward-only best of 3: 26.239 ms
online_hdim128 forward-only best of 3: 25.518 ms
dense backward-only best of 3: 19.987 ms
kernel backward-only best of 3: 58.600 ms
dense forward+backward best of 3: 43.072 ms
kernel forward+backward best of 3: 84.394 ms
kernel/dense ratio: 1.959x
```

A forward-only head-grouping experiment was then added: `prefixlm_forward_online_hdim128_headblock4`, which processes 4 heads for the same query in one threadgroup. It was correct but slower:

```text
512 tokens, 4 heads:
dense forward-only best of 3: 4.528 ms
online_hdim128 forward-only best of 3: 1.647 ms
headblock4_hdim128 forward-only best of 3: 1.918 ms

4096 tokens, 12 heads:
dense forward-only best of 3: 17.713 ms
online_hdim128 forward-only best of 3: 24.035 ms
headblock4_hdim128 forward-only best of 3: 26.564 ms
```

Conclusion: grouping heads inside one threadgroup adds synchronization overhead without meaningful reuse, because each head has independent Q/K/V. It should remain benchmark-only. For XL and wider models on MPS, use the dense PyTorch SDPA path unless a fundamentally different block kernel is developed. Confidence: high for this benchmark result.

True Q/K block experiments on 2026-05-25:

Two forward-only benchmark kernels were tried:

- `prefixlm_forward_matmulblock_hdim128_q4_k8`: 4 query rows x 8 key columns, 16 lanes per dot product.
- `prefixlm_forward_matmulblock_hdim128_q2_k8_l32`: 2 query rows x 8 key columns, 32 lanes per dot product.

Both compute a real Q/K score tile in threadgroup memory and perform a tile-level online softmax update. Both were numerically correct but slower than the simpler one-query online kernel and slower than dense SDPA at XL geometry.

Small smoke:

```text
512 tokens, 4 heads:
dense forward-only best of 3: 3.953 ms
online_hdim128 forward-only best of 3: 1.070 ms
headblock4_hdim128 forward-only best of 3: 1.217 ms
matmulblock_q2_k8_l32_hdim128 forward-only best of 3: 1.873 ms
```

XL geometry:

```text
4096 tokens, 12 heads:
dense forward-only best of 3: 18.028 ms
online_hdim128 forward-only best of 3: 23.176 ms
headblock4_hdim128 forward-only best of 3: 26.471 ms
matmulblock_q2_k8_l32_hdim128 forward-only best of 3: 52.895 ms
```

An earlier q4/k8/l16 XL run measured:

```text
matmulblock_q4_k8_hdim128 forward-only best of 3: 31.741 ms
```

Conclusion: hand-rolled threadgroup Q/K tiling without Apple matrix/SIMDgroup matrix instructions is not competitive for XL-width MPS attention. The barrier/reduction overhead dominates. The dense PyTorch MPS SDPA path remains the practical XL path. A future custom XL kernel would need to target Apple GPU matrix instructions or another lower-level acceleration path, not scalar threadgroup reductions. Confidence: high for the benchmark result; medium for the hardware-specific next-step assessment.

MPS SDPA implementation inspection on 2026-05-25:

- In this `torch==2.13.0.dev20260524` wheel, `torch.nn.functional.scaled_dot_product_attention` is a built-in op.
- Dispatch inspection shows an MPS-registered internal op:
  - `aten::_scaled_dot_product_attention_math_for_mps`
  - schema returns `(Tensor, Tensor)`
  - dispatch table has an `MPS` registration.
- The installed headers include MPS decode-attention Metal kernels at:
  - `/Users/petersk/Nobackup/miniconda3/envs/hrm/lib/python3.13/site-packages/torch/include/ATen/native/mps/kernels/DecodeAttention.h`
  - It defines `sdpa_vector`, `sdpa_vector_2pass_1`, and `sdpa_vector_2pass_2`, adapted from MLX.
  - The meta registration comments say `sdpa_vector_2pass_mps` and `sdpa_vector_fast_mps` are intentionally left out of meta handling, pointing to PyTorch issue `177603`.
- Profiling plain `F.scaled_dot_product_attention(q, k, v)` and a boolean-mask call on MPS showed `aten::_scaled_dot_product_attention_math`, `aten::bmm`, `_softmax`, and `_softmax_backward_data`, not `aten::_scaled_dot_product_attention_math_for_mps`.
- Directly calling `torch.ops.aten._scaled_dot_product_attention_math_for_mps(...)` works for a small MPS tensor and returns `(out, None)`.

Interpretation: for the dense PrefixLM fallback we use today, the observed speed likely comes from optimized MPS `bmm`/softmax primitives and PyTorch's generic math SDPA decomposition, not necessarily the dedicated MPS decode-attention vector kernels. Confidence: high for local dispatch/profiler observations; medium for exactly when PyTorch chooses the internal MPS SDPA op.

Direct internal MPS SDPA test on 2026-05-25:

```python
torch.ops.aten._scaled_dot_product_attention_math_for_mps(
    q, k, v, mask, 0.0, False, None, scale=None, enable_gqa=False
)
```

Forward-only worked and matched `F.scaled_dot_product_attention` exactly on a small masked MPS test:

```text
out diff: 0.0
F.sdpa forward:        0.909 ms
mps_internal forward:  0.435 ms
```

However, training backward is not implemented:

```text
RuntimeError: derivative for aten::_scaled_dot_product_attention_math_for_mps is not implemented
```

Conclusion: do not use `_scaled_dot_product_attention_math_for_mps` in the training path. It may be useful for forward-only inference/eval experiments, but it is a private/internal aten op with no autograd support in this build. Confidence: high.

Backward-part timing update on 2026-05-25:

Added benchmark-only helpers to time the custom MPS backward pieces separately:

- `flash_attn_varlen_prefixlm_mps_backward_context`
- `flash_attn_varlen_prefixlm_mps_backward_dq_part`
- `flash_attn_varlen_prefixlm_mps_backward_dk_dv_part`

4096-token XXS-geometry result:

```text
dense backward-only best of 3: 4.450 ms
kernel backward-only best of 3: 10.441 ms
kernel backward context (lse+query_dot) best of 3: 4.941 ms
kernel backward dq-part best of 3: 4.881 ms
kernel backward dk/dv-part best of 3: 5.782 ms
kernel backward parts sum: 15.604 ms
dense forward+backward best of 3: 21.651 ms
kernel forward+backward best of 3: 15.148 ms
kernel/dense ratio: 0.700x
```

The `context` timing includes an `lse` recomputation for the benchmark helper; real autograd saves `lse` from forward, so it is not part of the training backward in the same way. Among actual gradient kernels, `dk/dv` is the larger piece (`5.782 ms`) but `dq` is close (`4.881 ms`). Confidence: high.

Dense-math backward experiment on 2026-05-25:

Added a benchmark-only explicit backward helper:

```python
flash_attn_varlen_prefixlm_mps_backward_dense_math
```

It keeps the packed PrefixLM sequence loop but computes per-sequence dense attention matrices with MPS tensor ops:

```text
P = softmax(QK^T * scale + mask)
dV = P^T @ dO
dP = dO @ V^T
dS = P * (dP - sum(dO * O))
dQ = dS @ K
dK = dS^T @ Q
```

4096-token XXS-geometry result:

```text
dense-math dq max abs diff: 1.96695e-06
dense-math dk max abs diff: 1.90735e-06
dense-math dv max abs diff: 1.43051e-06
dense backward-only best of 3: 4.530 ms
kernel backward-only best of 3: 9.966 ms
dense-math backward explicit best of 3: 18.008 ms
estimated online-forward+dense-math-backward: 22.540 ms
dense forward+backward best of 3: 20.962 ms
kernel forward+backward best of 3: 14.892 ms
```

Conclusion: explicit Python-level per-sequence dense-math backward is correct but too slow. PyTorch's native dense SDPA autograd remains much faster for dense backward than reproducing the formulas manually in Python tensor ops. Keep this helper benchmark-only. Confidence: high.

SIMD32 forward update on 2026-05-26:

Added `prefixlm_forward_online_hdim128_simd32`, following the shape of PyTorch's MPS decode kernels:

- 32 lanes per query/head.
- For `head_dim=128`, each lane owns 4 Q/K/V elements.
- Dot products use `simd_sum` instead of 128-lane threadgroup reductions.
- Threadgroup memory is effectively avoided for the main online softmax state.
- The experimental autograd path now uses SIMD32 forward for `head_dim=128`; backward remains the specialized scalar MPS backward.

Sequential 4096-token XXS-geometry result:

```text
shape: tokens=4096 seqs=16 heads=2 head_dim=128 causal=False
dense forward-only best of 3: 16.741 ms
kernel forward-only best of 3: 1.377 ms
simd32_hdim128 forward-only best of 3: 1.417 ms
dense backward-only best of 3: 4.924 ms
kernel backward-only best of 3: 10.018 ms
dense forward+backward best of 3: 22.944 ms
kernel forward+backward best of 3: 11.828 ms
kernel/dense ratio: 0.516x
```

Sequential 4096-token XL-geometry result:

```text
shape: tokens=4096 seqs=16 heads=12 head_dim=128 causal=False
dense forward-only best of 3: 18.270 ms
kernel forward-only best of 3: 5.475 ms
simd32_hdim128 forward-only best of 3: 5.520 ms
dense backward-only best of 3: 20.497 ms
kernel backward-only best of 3: 68.820 ms
dense forward+backward best of 3: 41.135 ms
kernel forward+backward best of 3: 67.990 ms
kernel/dense ratio: 1.653x
```

Interpretation: SIMD32 fixes the forward path. It is about 12x faster than dense forward for the XXS geometry and about 3.3x faster for the XL geometry. End-to-end XXS is now about 1.9x faster than dense. XL remains slower end-to-end because the scalar custom backward scales poorly with head count. Confidence: high for the sequential isolated benchmark results.

SIMD32 backward update on 2026-05-26:

Added SIMD32 versions of the `head_dim=128` backward kernels:

- `prefixlm_backward_query_dot_hdim128_simd32`
- `prefixlm_backward_dq_hdim128_predot_simd32`
- `prefixlm_backward_dk_dv_hdim128_predot_simd32`

The experimental autograd path now uses SIMD32 forward plus SIMD32 backward for `head_dim=128`. The kernels keep the same exact PrefixLM mask semantics as the dense reference. Each 32-lane group handles one `(token, head)` item, and each lane owns four head-dimension elements for 128-dimensional heads.

Sequential 4096-token XXS-geometry result:

```text
shape: tokens=4096 seqs=16 heads=2 head_dim=128 causal=False
forward max abs diff: 1.72853e-06
dq max abs diff: 1.31877e-11
dk max abs diff: 1.86446e-11
dv max abs diff: 2.10321e-12
dense forward-only best of 5: 15.633 ms
kernel forward-only best of 5: 1.297 ms
dense backward-only best of 5: 4.190 ms
kernel backward-only best of 5: 2.889 ms
dense forward+backward best of 5: 21.274 ms
kernel forward+backward best of 5: 4.213 ms
kernel/dense ratio: 0.198x
```

Sequential 4096-token XL-geometry result:

```text
shape: tokens=4096 seqs=16 heads=12 head_dim=128 causal=False
forward max abs diff: 2.20537e-06
dq max abs diff: 3.35376e-12
dk max abs diff: 3.97904e-12
dv max abs diff: 5.11591e-13
dense forward-only best of 3: 17.862 ms
kernel forward-only best of 3: 5.598 ms
dense backward-only best of 3: 19.346 ms
kernel backward-only best of 3: 16.519 ms
dense forward+backward best of 3: 40.504 ms
kernel forward+backward best of 3: 22.758 ms
kernel/dense ratio: 0.562x
```

Interpretation: SIMD32 backward changed the custom path from forward-only useful to end-to-end useful for the tested XXS and XL geometries. XXS is about 5.0x faster than dense end-to-end; XL is about 1.8x faster. These are isolated kernel benchmarks, not full-model training throughput measurements. Confidence: high.

Model-path diagnostic after SIMD32 backward, 2026-05-26:

```bash
HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1 \
HRM_EXPERIMENTAL_MPS_MAX_TOKENS=4096 \
HRM_EXPERIMENTAL_MPS_MAX_SEQS=4096 \
HRM_EXPERIMENTAL_MPS_MAX_HEADS=2 \
HRM_EXPERIMENTAL_MPS_MAX_HEAD_DIM=128 \
/Users/petersk/Nobackup/miniconda3/bin/conda run -n hrm python scripts/debug_nan_training_step.py \
  --steps 1 \
  --override data.path=data/sampled_original_sapient_partial_smoke \
  --override arch/size@arch=XXS \
  --override accelerator_type=mps \
  --override compile_train_batch=false \
  --override fwd_bwd_dtype=float32 \
  --override global_batch_size=4096 \
  --override gradient_accumulation_steps=1 \
  --override epochs=1 \
  --override lr=2.5e-4 \
  --override lr_warmup_steps=10 \
  --override ema=null
```

Result: one real XXS model training diagnostic step completed on MPS with the experimental kernel enabled. The run reported finite loss (`11.602989196777344`), finite metrics, finite gradients, finite parameters, and finite post-optimizer parameters. Confidence: high.

Memory readout for the same diagnostic, rerun on 2026-05-26 after adding MPS memory logging:

```text
mps_memory startup: current=0.000 MiB driver=0.375 MiB
mps_memory after_init: current=456.016 MiB driver=1104.438 MiB
mps_memory step_1_before_train: current=456.109 MiB driver=1104.422 MiB
mps_memory step_1_after_train: current=552.192 MiB driver=6376.703 MiB
mps_memory step_1_after_zero_grad: current=456.192 MiB driver=6378.859 MiB
```

Interpretation: live tensor allocation for the one-step XXS diagnostic is about `552 MiB` immediately after train/optimizer work, and returns to about `456 MiB` after gradients are zeroed. The Metal/MPS driver allocator retained about `6.38 GiB`; this is retained driver memory, not live tensor memory. Confidence: high.

XL model-path diagnostic on 2026-05-26:

```bash
HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1 \
HRM_EXPERIMENTAL_MPS_MAX_TOKENS=1024 \
HRM_EXPERIMENTAL_MPS_MAX_SEQS=1024 \
HRM_EXPERIMENTAL_MPS_MAX_HEADS=12 \
HRM_EXPERIMENTAL_MPS_MAX_HEAD_DIM=128 \
/Users/petersk/Nobackup/miniconda3/bin/conda run -n hrm python scripts/debug_nan_training_step.py \
  --steps 1 \
  --override data.path=data/sampled_original_sapient_partial_smoke \
  --override arch/size@arch=XL \
  --override accelerator_type=mps \
  --override compile_train_batch=false \
  --override fwd_bwd_dtype=float32 \
  --override global_batch_size=1024 \
  --override gradient_accumulation_steps=1 \
  --override epochs=1 \
  --override lr=2.5e-4 \
  --override lr_warmup_steps=10 \
  --override ema=null
```

Result: one real XL model diagnostic step completed on MPS with the experimental kernel enabled. The run reported finite loss (`11.610648155212402`), finite metrics, finite gradients, finite parameters, and finite post-optimizer parameters.

Memory readout:

```text
mps_memory startup: current=0.000 MiB driver=0.375 MiB
mps_memory after_init: current=13544.016 MiB driver=14416.438 MiB
mps_memory step_1_before_train: current=13545.035 MiB driver=14416.422 MiB
mps_memory step_1_after_train: current=18533.056 MiB driver=24064.703 MiB
mps_memory step_1_after_zero_grad: current=13545.056 MiB driver=24066.859 MiB
```

Interpretation: at `global_batch_size=1024`, the XL diagnostic needs about `13.5 GiB` live MPS allocation after model/optimizer init and peaks at about `18.5 GiB` live allocation after the train/optimizer step. The Metal/MPS driver allocator retained about `24.1 GiB`. Confidence: high.

XL full-model dense-vs-custom timing on 2026-05-26:

Same one-step diagnostic configuration as above, run once with `HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1` and once without it:

```text
custom experimental MPS kernel:
  train_step_wall_ms=2133.971
  after_train current=18533.056 MiB driver=24064.703 MiB

dense MPS fallback:
  train_step_wall_ms=3073.054
  after_train current=18117.512 MiB driver=24016.750 MiB
```

Interpretation: for full XL model training at `global_batch_size=1024`, the experimental MPS kernel is about `1.44x` faster for the measured train step (`3073.054 / 2133.971`). Live MPS memory after the step is similar: custom is about `18.53 GiB`, dense is about `18.12 GiB`. Confidence: high for this single-step diagnostic comparison.

Five-step XL dense-vs-custom timing on 2026-05-26:

Same XL diagnostic configuration as above, `global_batch_size=1024`, `gradient_accumulation_steps=1`, `float32`, `epochs=1`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 2372.040
  step 2: 1950.961
  step 3: 12726.719
  step 4: 2261.985
  step 5: 2267.201

dense MPS fallback train_step_wall_ms:
  step 1: 2889.071
  step 2: 2383.261
  step 3: 6622.240
  step 4: 3043.306
  step 5: 2945.985
```

Summary:

```text
all-step mean:
  custom: 4315.781 ms
  dense:  3576.773 ms
  custom/dense: 1.207x

median:
  custom: 2267.201 ms
  dense:  2945.985 ms
  dense/custom speedup: 1.299x

excluding step 3 outlier:
  custom: 2213.047 ms
  dense:  2815.406 ms
  dense/custom speedup: 1.272x
```

Interpretation: the custom kernel is consistently faster on the non-outlier steps, by about `1.27x` to `1.30x` for this 5-step run. However, custom step 3 had a larger allocation/shape outlier than dense (`12.7 s` vs `6.6 s`), making the all-step mean worse. The step-3 batch was also the one that exceeded the original 1024-token experimental cap, so packed attention shape variability matters for full-model timing. Confidence: high for the recorded run; medium for extrapolating steady-state throughput from only five steps.

Ten-step XL dense-vs-custom timing on 2026-05-26:

Same XL diagnostic configuration as above, `global_batch_size=1024`, `gradient_accumulation_steps=1`, `float32`, `epochs=1`, with the experimental custom run capped at 4096 packed attention tokens.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1:  2404.737
  step 2:  1932.350
  step 3: 12825.552
  step 4:  2273.799
  step 5:  2035.316
  step 6:  2039.646
  step 7:  1971.645
  step 8:  1980.948
  step 9:  2038.020
  step 10: 1958.623

dense MPS fallback train_step_wall_ms:
  step 1: 2921.895
  step 2: 2391.747
  step 3: 6603.023
  step 4: 2910.397
  step 5: 2810.847
  step 6: 2949.683
  step 7: 2750.941
  step 8: 2995.389
  step 9: 2729.211
  step 10: 2890.358
```

Summary:

```text
all-step mean:
  custom: 3146.064 ms
  dense:  3195.349 ms
  dense/custom speedup: 1.016x

median:
  custom: 2036.668 ms
  dense:  2900.378 ms
  dense/custom speedup: 1.424x

excluding step 3 outlier:
  custom: 2070.565 ms
  dense:  2816.719 ms
  dense/custom speedup: 1.360x

steps 4-10:
  custom: 2042.571 ms
  dense:  2862.404 ms
  dense/custom speedup: 1.401x
```

Interpretation: the custom kernel still has a severe step-3 outlier on this sampled-data prefix shape, but after that allocation/shape event it stabilizes tightly around `2.0 s/step`. Dense stabilizes closer to `2.8-3.0 s/step`. The best current steady-state estimate from this diagnostic is about `1.4x` full-model speedup for XL at `global_batch_size=1024`, while all-step mean over only 10 steps is nearly tied because the one custom outlier dominates. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.

Ten-step L dense-vs-custom timing on 2026-05-26:

Same diagnostic setup as the XL comparison, but with `arch/size@arch=L`. The custom run used the experimental MPS kernel with caps `tokens=4096`, `heads=10`, `head_dim=128`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 1373.142
  step 2: 1139.257
  step 3: 7711.535
  step 4: 1277.349
  step 5: 1336.466
  step 6: 1278.594
  step 7: 1309.203
  step 8: 1328.550
  step 9: 1322.252
  step 10: 1165.187

dense MPS fallback train_step_wall_ms:
  step 1: 1854.250
  step 2: 1250.963
  step 3: 3961.176
  step 4: 1772.268
  step 5: 1637.370
  step 6: 1671.016
  step 7: 1744.570
  step 8: 1755.720
  step 9: 1700.678
  step 10: 1497.062
```

Summary:

```text
all-step mean:
  custom: 1924.154 ms
  dense:  1884.507 ms
  dense/custom speedup: 0.979x

median:
  custom: 1315.727 ms
  dense:  1722.624 ms
  dense/custom speedup: 1.309x

excluding step 3 outlier:
  custom: 1281.111 ms
  dense:  1653.766 ms
  dense/custom speedup: 1.291x

steps 4-10:
  custom: 1288.229 ms
  dense:  1682.669 ms
  dense/custom speedup: 1.306x
```

Memory:

```text
custom after_init current=7958.016 MiB driver=8272.438 MiB
custom highest after_train current=11267.556 MiB driver=19938.891 MiB
dense after_init current=7958.016 MiB driver=8272.438 MiB
dense highest after_train current=10997.335 MiB driver=26652.312 MiB
```

Interpretation: L shows the same shape as XL. The custom kernel is about `1.29x-1.31x` faster on median/steady steps, but the custom step-3 outlier makes the 10-step all-step mean slightly worse than dense. Live memory after model init is about `8.0 GiB`; peak live after-train memory was about `11.3 GiB` for custom and about `11.0 GiB` for dense. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.

Ten-step B dense-vs-custom timing on 2026-05-26:

Same diagnostic setup as the L/XL comparisons, but with `arch/size@arch=B`. The custom run used the experimental MPS kernel with caps `tokens=4096`, `heads=8`, `head_dim=128`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 566.046
  step 2: 446.681
  step 3: 2830.671
  step 4: 464.581
  step 5: 461.750
  step 6: 461.621
  step 7: 446.265
  step 8: 447.065
  step 9: 463.905
  step 10: 442.330

dense MPS fallback train_step_wall_ms:
  step 1: 819.578
  step 2: 500.184
  step 3: 1606.054
  step 4: 698.965
  step 5: 674.202
  step 6: 682.212
  step 7: 696.910
  step 8: 805.104
  step 9: 617.073
  step 10: 665.653
```

Summary:

```text
all-step mean:
  custom: 703.091 ms
  dense:  776.594 ms
  dense/custom speedup: 1.105x

median:
  custom: 461.685 ms
  dense:  689.561 ms
  dense/custom speedup: 1.494x

excluding step 3 outlier:
  custom: 466.694 ms
  dense:  684.431 ms
  dense/custom speedup: 1.467x

steps 4-10:
  custom: 455.360 ms
  dense:  691.446 ms
  dense/custom speedup: 1.518x
```

Memory:

```text
custom after_init current=3452.016 MiB driver=4144.438 MiB
custom highest after_train current=4547.792 MiB driver=9922.891 MiB
dense after_init current=3452.016 MiB driver=4144.438 MiB
dense highest after_train current=4474.851 MiB driver=13068.297 MiB
```

Interpretation: B is the cleanest full-model comparison so far. The custom kernel wins even in the all-step mean despite the same step-3 packed-shape spike. Steady-state speedup is about `1.5x`. Live memory after model init is about `3.45 GiB`; peak live after-train memory was about `4.55 GiB` for custom and about `4.47 GiB` for dense. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.

Ten-step S dense-vs-custom timing on 2026-05-26:

Same diagnostic setup as the B/L/XL comparisons, but with `arch/size@arch=S`. The custom run used the experimental MPS kernel with caps `tokens=4096`, `heads=6`, `head_dim=128`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 300.132
  step 2: 207.104
  step 3: 1492.006
  step 4: 264.668
  step 5: 261.283
  step 6: 234.830
  step 7: 217.919
  step 8: 221.069
  step 9: 226.032
  step 10: 217.958

dense MPS fallback train_step_wall_ms:
  step 1: 618.271
  step 2: 291.430
  step 3: 798.322
  step 4: 388.066
  step 5: 381.017
  step 6: 447.687
  step 7: 414.812
  step 8: 380.657
  step 9: 332.358
  step 10: 374.163
```

Summary:

```text
all-step mean:
  custom: 364.300 ms
  dense:  442.678 ms
  dense/custom speedup: 1.215x

median:
  custom: 230.431 ms
  dense:  384.541 ms
  dense/custom speedup: 1.669x

excluding step 3 outlier:
  custom: 238.999 ms
  dense:  403.162 ms
  dense/custom speedup: 1.687x

steps 4-10:
  custom: 234.823 ms
  dense:  388.394 ms
  dense/custom speedup: 1.654x
```

Memory:

```text
custom after_init current=1862.016 MiB driver=2352.438 MiB
custom highest after_train current=2429.056 MiB driver=7090.891 MiB
dense after_init current=1862.016 MiB driver=2352.438 MiB
dense highest after_train current=2428.710 MiB driver=8276.281 MiB
```

Interpretation: S shows the strongest steady-state gain so far: about `1.65x-1.69x` faster for custom on median/steady-step comparisons, and still `1.21x` faster even in the all-step mean with the step-3 spike included. Live memory after model init is about `1.86 GiB`; peak live after-train memory is about `2.43 GiB` for both custom and dense. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.

Ten-step XS dense-vs-custom timing on 2026-05-26:

Same diagnostic setup as the S/B/L/XL comparisons, but with `arch/size@arch=XS`. The custom run used the experimental MPS kernel with caps `tokens=4096`, `heads=4`, `head_dim=128`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 183.148
  step 2: 136.713
  step 3: 791.673
  step 4: 136.787
  step 5: 131.993
  step 6: 128.659
  step 7: 122.648
  step 8: 124.118
  step 9: 128.996
  step 10: 124.114

dense MPS fallback train_step_wall_ms:
  step 1: 358.637
  step 2: 185.095
  step 3: 448.623
  step 4: 253.336
  step 5: 288.552
  step 6: 289.228
  step 7: 289.653
  step 8: 256.429
  step 9: 204.237
  step 10: 236.410
```

Summary:

```text
all-step mean:
  custom: 200.885 ms
  dense:  281.020 ms
  dense/custom speedup: 1.399x

median:
  custom: 130.495 ms
  dense:  272.490 ms
  dense/custom speedup: 2.088x

excluding step 3 outlier:
  custom: 135.242 ms
  dense:  262.397 ms
  dense/custom speedup: 1.940x

steps 4-10:
  custom: 128.188 ms
  dense:  259.692 ms
  dense/custom speedup: 2.026x
```

Memory:

```text
custom after_init current=1028.016 MiB driver=1152.438 MiB
custom highest after_train current=1530.060 MiB driver=5994.891 MiB
dense after_init current=1028.016 MiB driver=1152.438 MiB
dense highest after_train current=1403.718 MiB driver=5940.297 MiB
```

Interpretation: XS shows the largest steady-state full-model speedup so far: about `2.0x` on median/steady-step comparisons, and still about `1.4x` faster in the all-step mean with the step-3 spike included. Live memory after model init is about `1.03 GiB`; peak live after-train memory was about `1.53 GiB` for custom and about `1.40 GiB` for dense. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.

XS target-batch dense diagnostic on 2026-05-26:

Ran `arch/size@arch=XS` with the target effective batch and dense MPS fallback:

```text
global_batch_size=32768
gradient_accumulation_steps=8
local_microbatch_size=4096
```

Ten optimizer-step timing:

```text
dense MPS fallback train_step_wall_ms:
  step 1: 10154.409
  step 2: 8064.698
  step 3: 10417.585
  step 4: 8831.811
  step 5: 10437.219
  step 6: 9719.472
  step 7: 9236.608
  step 8: 9654.086
  step 9: 8904.924
  step 10: 9089.388
```

Summary:

```text
mean: 9451.020 ms/optimizer-step
median: 9445.347 ms/optimizer-step
min: 8064.698 ms
max: 10437.219 ms
steps 4-10 mean: 9410.501 ms/optimizer-step
mean per physical microbatch: 1181.378 ms
steps 4-10 per physical microbatch: 1176.313 ms
```

Memory:

```text
after_init current=1028.016 MiB driver=1152.438 MiB
highest after_train current=2165.414 MiB driver=13767.875 MiB
after_zero_grad current≈1029 MiB
```

Result: all ten optimizer steps completed with finite loss, metrics, gradients, parameters, and post-optimizer parameters. Confidence: high.

XS target-batch custom-kernel diagnostic on 2026-05-26:

Same target effective batch as the dense diagnostic:

```text
global_batch_size=32768
gradient_accumulation_steps=8
local_microbatch_size=4096
HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1
HRM_EXPERIMENTAL_MPS_MAX_TOKENS=4096
HRM_EXPERIMENTAL_MPS_MAX_HEADS=4
```

Ten optimizer-step timing:

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 4182.757
  step 2: 5032.271
  step 3: 3788.423
  step 4: 4666.824
  step 5: 3445.798
  step 6: 3577.579
  step 7: 4407.849
  step 8: 3931.713
  step 9: 4276.995
  step 10: 4725.162
```

Paired dense-vs-custom summary:

```text
all-step mean:
  custom: 4203.537 ms/optimizer-step
  dense:  9451.020 ms/optimizer-step
  dense/custom speedup: 2.248x

median:
  custom: 4229.876 ms/optimizer-step
  dense:  9445.347 ms/optimizer-step
  dense/custom speedup: 2.233x

steps 4-10 mean:
  custom: 4147.417 ms/optimizer-step
  dense:  9410.501 ms/optimizer-step
  dense/custom speedup: 2.269x

per physical microbatch:
  custom mean: 525.442 ms
  dense mean:  1181.378 ms
```

Memory:

```text
custom after_init current=1028.016 MiB driver=1152.438 MiB
custom highest after_train current=1298.934 MiB driver=7482.859 MiB
dense after_init current=1028.016 MiB driver=1152.438 MiB
dense highest after_train current=2165.414 MiB driver=13767.875 MiB
```

Result: all ten custom optimizer steps completed with finite loss, metrics, gradients, parameters, and post-optimizer parameters. Interpretation: at the target XS effective batch (`32768` with `gas=8`), the custom kernel is about `2.25x` faster than dense and uses substantially less live and retained MPS memory during the step. Confidence: high.

XXS full smoke run metric dip on 2026-05-26:

Run directory:

```text
wandb/run-20260526_010620-pe62hvad
```

Config:

```text
arch/size@arch=XXS
global_batch_size=16384
gradient_accumulation_steps=4
local_microbatch_size=4096
epochs=1
log_interval=1
HRM_ENABLE_EXPERIMENTAL_MPS_KERNEL=1
```

Observed event:

```text
steps 1120-1160:
  mean train/loss:     4.867869
  mean train/accuracy: 0.309116

steps 1161-1200:
  mean train/loss:     7.881740
  mean train/accuracy: 0.174512

steps 1201-1220:
  mean train/loss:     7.367802
  mean train/accuracy: 0.168310
```

The change starts gradually at step `1161`: loss rises from about `4.85-5.08` to `5.23`, then reaches `7-8.7` by steps `1167-1180`; token accuracy falls from about `0.30-0.32` to `0.14-0.18`. `train/exact_accuracy` stayed `0`, `bp_steps` stayed `5`, and `train/lr` stayed `2.5e-4`, so this was not caused by LR scheduling or recurrence warmup. Step runtime stayed in the same general range, with no local W&B error/warning at the transition.

Dataset reconstruction from `data/sampled_original_sapient_partial_smoke` showed no top-level source switch across the dip. The windows are all dominated by the same SYNTH smoke subset:

```text
steps 1120-1160:
  SYNTH rows: 1681
  UNKNOWN rows from reconstructed range mapping: 68
  top tasks: SYNTH__synth_230.parquet, SYNTH__synth_175.parquet, SYNTH__synth_176.parquet

steps 1161-1200:
  SYNTH rows: 1663
  UNKNOWN rows from reconstructed range mapping: 62
  top tasks: SYNTH__synth_176.parquet, SYNTH__synth_230.parquet, SYNTH__synth_175.parquet
```

Decoded rows around the dip include harder long-form synthetic writing/legal/history examples, including near-max-length responses such as a Hellenistic papyrus-style creative fragment (`resp_len=3917`) at step `1168`. Superseded interpretation: this initially looked like a data/batch difficulty region in the smoke sample, not a kernel, optimizer, LR, or W&B failure. Confidence: high for the metric extraction and config; low for the original data-difficulty interpretation after dense-vs-custom comparison below.

Correction on 2026-05-26: the earlier dense XXS smoke run `smoke-xxs-mps-bs16384-ga4` in `wandb/run-20260525_173337-apkt56ui` used the same seed and same target batch settings but did not show the step `1161-1200` spike. Direct local W&B history comparison:

```text
dense run, steps 1120-1160:
  mean train/loss:     4.922174
  mean train/accuracy: 0.304512
dense run, steps 1161-1200:
  mean train/loss:     4.877770
  mean train/accuracy: 0.307725
dense run, steps 1201-1220:
  mean train/loss:     4.881029
  mean train/accuracy: 0.303363

custom run, steps 1120-1160:
  mean train/loss:     4.867869
  mean train/accuracy: 0.309116
custom run, steps 1161-1200:
  mean train/loss:     7.881740
  mean train/accuracy: 0.174512
custom run, steps 1201-1220:
  mean train/loss:     7.367802
  mean train/accuracy: 0.168310
```

This makes the experimental custom MPS attention kernel the leading suspect. The failure is probably cumulative numerical or backward-gradient error rather than an immediate forward mismatch, because early training matches dense closely and the divergence appears only after more than one thousand optimizer steps. Confidence: high for the dense-vs-custom metric comparison; medium for the suspected mechanism.

Follow-up diagnostic on 2026-05-26: added `scripts/compare_dense_custom_mps_attention.py` to compare dense MPS attention and the experimental custom MPS attention with identical model weights, carry, batches, and gradient accumulation. The script reports logits, microbatch losses, normalized metrics, and per-parameter gradient differences.

Results with current on-disk code:

```text
initial weights, optimizer_step=1:
  logits max_abs diff on first 512 tokens: 5.48362732e-06
  all four microbatch losses exactly matched at printed precision
  metric loss/accuracy/exact_accuracy matched
  largest gradient max_abs diff: 7.45058060e-09

initial weights, optimizer_step=1161:
  logits max_abs diff on first 512 tokens: 5.06639481e-06
  metric loss diff: 1.90734863e-06
  largest gradient max_abs diff: 1.86264515e-08

dense epoch-1 checkpoint, optimizer_step=1:
  metric loss/accuracy/exact_accuracy matched
  largest gradient max_abs diff: 6.05359674e-09

custom epoch-1 checkpoint, optimizer_step=1:
  metric loss diff: -4.76837158e-07
  metric accuracy/exact_accuracy matched
  largest gradient max_abs diff: 9.77888703e-09
```

Interpretation: the current on-disk custom kernel passes single-step dense equivalence at initial weights, at the bad batch index with initial weights, and at both dense-trained and custom-trained epoch-1 checkpoints. This does not clear the custom kernel used by the bad run, because that run imported `models/flash_attention_prefixlm_mps.py` at startup (`01:06`) and the file was edited later (`03:43`). Current diagnostics therefore test the latest file contents, not necessarily the exact in-memory Metal code that produced `smoke-xxs-custom-mps-bs16384-ga4`. Confidence: high for the diagnostic results; medium for the conclusion that the current file is single-step-correct but the bad run may have used an older in-memory kernel.

Full rerun on 2026-05-26: `smoke-xxs-custom-mps-bs16384-ga4-rerun` in `wandb/run-20260526_041126-tbtcaqxu` used the same XXS custom MPS settings after the on-disk kernel diagnostics. It did not reproduce the loss/accuracy spike:

```text
rerun custom, steps 1120-1160:
  mean train/loss:     4.862339
  mean train/accuracy: 0.309331
rerun custom, steps 1161-1200:
  mean train/loss:     4.816304
  mean train/accuracy: 0.312102
rerun custom, steps 1201-1220:
  mean train/loss:     4.818233
  mean train/accuracy: 0.308371
rerun custom, steps 1300-1410:
  mean train/loss:     4.769467
  mean train/accuracy: 0.313416
```

Conclusion: the spike in `smoke-xxs-custom-mps-bs16384-ga4` is not reproducible with the current custom MPS kernel. The most likely explanation is that the bad run used an older in-memory version of the experimental MPS kernel from before the later edits, or some other transient run-specific state, rather than a persistent issue in the current on-disk kernel. Confidence: high for the rerun comparison; medium for the causal explanation.

XXS_wide custom rerun on 2026-05-26: `smoke-xxs-wide-custom-mps-bs16384-ga4-rerun` in `wandb/run-20260526_065719-k1sjdsw2` used `arch/size@arch=XXS_wide` (`hidden_size=384`, `num_heads=3`, `head_dim=128`) with `HRM_EXPERIMENTAL_MPS_MAX_HEADS=3`. The run was still alive when inspected, but had already diverged by step `153`; no epoch checkpoint had been written yet.

```text
custom XXS_wide, steps 1-20:
  mean train/loss:     10.853374
  mean train/accuracy: 0.061030
custom XXS_wide, steps 100-140:
  mean train/loss:     6.313171
  mean train/accuracy: 0.213356
custom XXS_wide, last logged rows 153-172:
  loss range: roughly 6.47 -> 7.73 -> 7.36
  accuracy range: roughly 0.18-0.23
```

The earlier dense `XXS_wide` run `smoke-xxs-wide-mps-bs16384-ga4` in `wandb/run-20260525_201730-e75a2o1u` had identical metrics through early training (`1-20` and `100-140`) but remained stable later (`steps 1120-1200` loss about `4.64-4.68`, accuracy about `0.317-0.321`). Interpretation: the current custom MPS kernel appears stable for `XXS` with 2 heads but unstable for `XXS_wide` with 3 heads, or for some interaction exposed by the wider geometry. Confidence: high for metric comparison; medium for attributing specifically to `num_heads=3`.

Tighter dense-vs-custom comparison for `XXS_wide`: the dense run `wandb/run-20260525_201730-e75a2o1u` and custom run `wandb/run-20260526_065719-k1sjdsw2` are identical through step `149`; custom begins to diverge at steps `150-152` and then clearly fails from step `153`.

```text
dense XXS_wide:
  steps 141-152: loss 6.093142, accuracy 0.234157
  steps 153-172: loss 6.061670, accuracy 0.234757
  steps 173-192: loss 5.967895, accuracy 0.242694

custom XXS_wide:
  steps 141-152: loss 6.122311, accuracy 0.233062
  steps 153-172: loss 7.309322, accuracy 0.197355
  steps 173-192: loss 7.846729, accuracy 0.165831
```

Confidence: high.

Single-step `XXS_wide` dense-vs-custom diagnostic on 2026-05-26: `scripts/compare_dense_custom_mps_attention.py --optimizer-step 153` with `arch/size@arch=XXS_wide`, `global_batch_size=16384`, `gradient_accumulation_steps=4`, and custom caps `tokens=4096`, `heads=3`, `head_dim=128` showed no immediate per-batch kernel mismatch from initial weights:

```text
logits first 512 tokens max_abs diff: 5.48362732e-06
all four microbatch losses matched at printed precision
metric loss/accuracy/exact_accuracy matched
largest gradient max_abs diff: 1.49011612e-08
```

Interpretation: the visible `XXS_wide` training divergence is not explained by a simple single-batch forward/backward indexing failure at step `153` from initial weights. The next likely causes are cumulative trajectory sensitivity from small numerical differences, a stateful/runtime issue across repeated MPS kernel launches, or a bug that only appears after weights/optimizer state have evolved. Confidence: high for the diagnostic result; medium for interpretation.

XXS FLOP estimate on 2026-05-29: with `arch/size@arch=XXS`, HRM uses `hidden_size=256`, `num_heads=2`, `head_dim=128`, `n_layers=6`, and `half_layers=true`, so the H and L modules each have 3 transformer layers. `H_cycles=2`, `L_cycles=3`, for 8 recurrent transformer-stack calls per forward pass. At `global_batch_size=16384`, linear/LM-head FLOPs are approximately:

```text
bp_steps=2: 2.73e12 FLOPs before attention
bp_steps=5: 3.27e12 FLOPs before attention
```

For the first smoke optimizer step (`gradient_accumulation_steps=4`), the actual packed microbatches contained 15,882 tokens and 7,345,463 allowed PrefixLM query-key pairs, about 462 attended keys/token. Scaling to the nominal 16,384-token step gives an attention term of roughly `0.28e12` FLOPs at `bp_steps=2` and `0.42e12` FLOPs at `bp_steps=5`. Total expected forward+backward FLOPs per XXS optimizer step are therefore roughly:

```text
bp_steps=2: ~3.0e12 FLOPs
bp_steps=3: ~3.24e12 FLOPs
bp_steps=4: ~3.47e12 FLOPs
bp_steps=5: ~3.69e12 FLOPs
```

The simpler dense-transformer back-of-envelope `6 * num_params * tokens` with `num_params=39,059,456` and `tokens=16,384` gives `3.84e12` FLOPs/step, but it overcounts embedding lookup work and does not model HRM recurrence explicitly. Confidence: medium; exact attention FLOPs vary with packing/sequence lengths.

General HRM FLOP formula used for this estimate:

```text
T       = supervised/packed tokens per optimizer step, usually global_batch_size
A       = allowed PrefixLM query-key pairs per optimizer step, before multiplying by heads
D       = hidden_size
h       = num_heads
d       = head_dim = D / h
hk      = num_key_value_heads, currently h
I       = intermediate_size
V       = vocab_size
NH, NL  = number of transformer layers in the H and L stacks
HC, LC  = H_cycles and L_cycles
b       = bp_steps
Hbp     = min(HC, max(0, b - 1))
Lbp     = min(HC * LC, max(0, b - Hbp))

QKVout          = (2h + 2hk) * d
F_linear_fwd    = T * (2D * QKVout + 2D^2 + 6DI)
F_linear_bwd    = 2 * F_linear_fwd
F_attn_fwd      = 4h d A
F_attn_bwd      = 10h d A + 2h d T
F_layer_fwd     = F_linear_fwd + F_attn_fwd
F_layer_bwd     = F_linear_bwd + F_attn_bwd
F_recurrent     = (HC * NH + HC * LC * NL) * F_layer_fwd
                + (Hbp * NH + Lbp * NL) * F_layer_bwd
F_lm_head       = 6T D V
F_step          = F_recurrent + F_lm_head
```

This counts multiply-add matmul/attention FLOPs and intentionally excludes most elementwise ops; using `A = T * max_seq_len` gives a conservative attention upper bound. Confidence: medium-high for matmul/attention accounting; low for exact hardware instruction count.

XL full-run worked example on 2026-05-29: for `arch/size@arch=XL`, use `T=196608`, `D=1536`, `h=12`, `d=128`, `I=4096`, `V=65536`, `NH=NL=16` after `half_layers=true`, `HC=2`, `LC=3`, `bp_min=2`, `bp_max=5`, `bp_warmup_ratio=0.2`, and `steps=325925`. The HRM bp schedule gives:

```text
bp_steps=2: 21,728 steps
bp_steps=3: 21,728 steps
bp_steps=4: 21,728 steps
bp_steps=5: 260,741 steps
```

Using conservative worst-case attention `A = T * max_seq_len = 196608 * 4096 = 805,306,368` allowed query-key pairs per step:

```text
bp_steps=2: 3.463 PFLOPs/step
bp_steps=3: 4.047 PFLOPs/step
bp_steps=4: 4.631 PFLOPs/step
bp_steps=5: 5.215 PFLOPs/step
total:      1.624e21 FLOPs = 1.624 ZFLOPs
```

Using the first-smoke-batch observed average allowed attention density (`~462.5` keys/token) scaled to XL gives `A ~= 90,931,670` and a less conservative estimate:

```text
bp_steps=2: 2.551 PFLOPs/step
bp_steps=3: 2.959 PFLOPs/step
bp_steps=4: 3.367 PFLOPs/step
bp_steps=5: 3.775 PFLOPs/step
total:      1.177e21 FLOPs = 1.177 ZFLOPs
```

Confidence: medium. The conservative number intentionally overestimates attention; exact attention cost depends on the original Sapient packing distribution.

XXL doubled-dataset worked example on 2026-05-29: for `arch/size@arch=XXL`, use `T=262144`, `D=1792`, `h=14`, `d=128`, `I=4864`, `V=65536`, `NH=NL=36` after `half_layers=true`, `HC=2`, `LC=3`, and the same HRM bp schedule. Starting from the XL reference `325925` steps at `196608` tokens/step, doubling the dataset and using `262144` tokens/step gives:

```text
steps = floor(2 * 325925 * 196608 / 262144) = 488887

bp_steps=2:  32,592 steps
bp_steps=3:  32,592 steps
bp_steps=4:  32,593 steps
bp_steps=5: 391,110 steps
```

Using conservative worst-case attention `A = T * max_seq_len = 262144 * 4096 = 1,073,741,824`:

```text
bp_steps=2: 13.346 PFLOPs/step
bp_steps=3: 15.632 PFLOPs/step
bp_steps=4: 17.918 PFLOPs/step
bp_steps=5: 20.204 PFLOPs/step
total:      9.430e21 FLOPs = 9.430 ZFLOPs
```

Using the first-smoke-batch observed average allowed attention density (`~462.5` keys/token) scaled to `T=262144` gives `A ~= 121,242,227`:

```text
bp_steps=2: 10.151 PFLOPs/step
bp_steps=3: 11.822 PFLOPs/step
bp_steps=4: 13.494 PFLOPs/step
bp_steps=5: 15.165 PFLOPs/step
total:      7.087e21 FLOPs = 7.087 ZFLOPs
```

Confidence: medium. The conservative number is the planning upper estimate; exact attention cost depends on packing.

Size-ladder update on 2026-05-29: added only the non-wide branch above `XXL`:

```text
XXXL:  n_layers=96,  hidden_size=2048, num_heads=16, head_dim=128
XXXXL: n_layers=128, hidden_size=2560, num_heads=20, head_dim=128
```

No new `*_wide` configs were kept. Hydra config composition was checked successfully for both `arch/size@arch=XXXL` and `arch/size@arch=XXXXL`. Confidence: high.

Parameter and FLOP estimates for the new non-wide sizes on 2026-05-29, assuming HRM `half_layers=true`, `H_cycles=2`, `L_cycles=3`, `bp_warmup_ratio=0.2`, and a 50B-token dataset iterated for 4 epochs (`200B` token presentations):

```text
XXXL:
  physical params:          5.604B
  effective forward params: 21.609B
  smoke-density training:   19.223 ZFLOPs
  worst-attention training: 24.796 ZFLOPs

XXXXL:
  physical params:          11.325B
  effective forward params: 44.292B
  smoke-density training:   39.186 ZFLOPs
  worst-attention training: 48.473 ZFLOPs
```

Here “effective forward params” means transformer-stack parameters counted once per recurrent stack call plus embeddings/LM head once; it is not additional stored parameters. “Smoke-density” uses the observed first-smoke-batch average of about `462.5` allowed attention keys per token; “worst-attention” uses `max_seq_len=4096`. Confidence: medium.

Follow-up on 2026-05-26: several attention/model support files were edited during the wall-clock window of this run, overlapping the metric dip. The run started at `2026-05-26 01:06:20` local time; the dip started around step `1161`, roughly `03:40`. File mtimes showed `models/flash_attention_prefixlm_common.py` at `03:41:09`, `models/flash_attention_prefixlm_fa3.py` at `03:42:16`, `models/flash_attention_prefixlm_mps.py` at `03:43:54`, `models/flash_attention_prefixlm_fa4.py` at `03:46:53`, `models/flash_attention_prefixlm_dense.py` at `03:47:15`, and `models/flash_attention_prefixlm_v2.py` at `03:47:59`; core model files such as `models/layers.py`, `models/transformer.py`, `models/lm_head.py`, and `models/baselines/hrm_nocarry_bp_warmup.py` had older mtimes from 2026-05-24/25. A normal running Python process does not pick up `.py` file changes after import unless code explicitly reloads modules, and this training path does not do that. The attention dispatcher lazily imports the backend on first use, but by step `1161` the XXS run had already executed many attention calls, so the active module/function objects and Metal shader source were already resident in memory. Conclusion: the overlapping file rewrites can affect later runs, but they are not a plausible cause of the active run's loss/accuracy dip unless some external process forced dynamic module reloads, for which there is no evidence. Confidence: high for file timestamps; medium-high for non-causality based on Python import semantics and inspected dispatcher code.

Ten-step XXS dense-vs-custom timing on 2026-05-26:

Same diagnostic setup as the XS/S/B/L/XL comparisons, but with `arch/size@arch=XXS`. The custom run used the experimental MPS kernel with caps `tokens=4096`, `heads=2`, `head_dim=128`.

```text
custom experimental MPS kernel train_step_wall_ms:
  step 1: 125.844
  step 2: 62.802
  step 3: 406.887
  step 4: 74.021
  step 5: 72.748
  step 6: 72.820
  step 7: 69.706
  step 8: 69.932
  step 9: 72.769
  step 10: 69.829

dense MPS fallback train_step_wall_ms:
  step 1: 322.335
  step 2: 108.761
  step 3: 297.280
  step 4: 183.558
  step 5: 205.398
  step 6: 211.766
  step 7: 198.437
  step 8: 216.849
  step 9: 175.935
  step 10: 204.939
```

Summary:

```text
all-step mean:
  custom: 109.736 ms
  dense:  212.526 ms
  dense/custom speedup: 1.937x

median:
  custom: 72.758 ms
  dense:  205.168 ms
  dense/custom speedup: 2.820x

excluding step 3 outlier:
  custom: 76.719 ms
  dense:  203.109 ms
  dense/custom speedup: 2.647x

steps 4-10:
  custom: 71.689 ms
  dense:  199.555 ms
  dense/custom speedup: 2.784x
```

Memory:

```text
custom after_init current=456.016 MiB driver=1104.438 MiB
custom highest after_train current=556.573 MiB driver=4874.891 MiB
dense after_init current=456.016 MiB driver=1104.438 MiB
dense highest after_train current=556.143 MiB driver=4908.312 MiB
```

Interpretation: XXS is the strongest result so far. Custom is about `1.94x` faster even in the all-step mean with the spike included, and about `2.65x-2.82x` faster on non-outlier/steady-step comparisons. Live memory after model init is about `456 MiB`; peak live after-train memory is about `557 MiB` for both custom and dense. Confidence: high for the recorded run; medium for longer-run throughput because only 10 steps were measured.
