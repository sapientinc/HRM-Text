from dataclasses import dataclass, fields
from typing import Optional, Any
import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.common import IGNORE_LABEL_ID
from models.flash_attention_prefixlm_v2 import compute_aux_seq_tensors_scalars
from models.layers import find_multiple
from multipack_sampler import MultipackDistributedBatchSampler


class V1DatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    batch_max_length: int
    drop_last_batch: bool

    target_only: bool

    rank: int
    num_replicas: int


class V1DatasetMeta(pydantic.BaseModel):
    tokenizer_info: dict[str, Any] = {}
    vocab_size: Optional[int] = None
    max_seq_len: int
    total_length: int


@dataclass
class V1DatasetIndices:
    inst_start: np.ndarray
    inst_len: np.ndarray
    resp_start: np.ndarray
    resp_len: np.ndarray


class V1Dataset(IterableDataset):
    def __init__(self, config: V1DatasetConfig):
        super().__init__()
        self.config = config
        self.metadata = self._load_metadata()

        # State
        self._data: Optional[np.ndarray] = None
        self._data_indices: Optional[V1DatasetIndices] = None
        self._sampler: Optional[MultipackDistributedBatchSampler] = None
        self._epoch = 0

    def _load_metadata(self) -> V1DatasetMeta:
        with open(os.path.join(self.config.dataset_path, "metadata.json"), "r") as f:
            metadata = V1DatasetMeta(**json.load(f))
            # Account for autoregressive shift
            metadata.max_seq_len -= 1
            # Compute vocab size from tokenizer info
            assert metadata.vocab_size is None
            metadata.vocab_size = find_multiple(metadata.tokenizer_info.pop("vocab_size"), 256)

            return metadata

    def _load_dataset_before_epoch_begin(self):
        # Load tokens (only if not loaded)
        if self._data is None:
            self._data = np.load(os.path.join(self.config.dataset_path, "tokens.npy"), mmap_mode="r")

        # Load indices
        self._data_indices = V1DatasetIndices(**{f.name: np.load(os.path.join(self.config.dataset_path, f"epoch_{self._epoch}", f"{f.name}.npy"), mmap_mode="r")
                                                 for f in fields(V1DatasetIndices)})
        self._epoch += 1

        # Re-create sampler
        self._sampler = MultipackDistributedBatchSampler(
            lengths=self._data_indices.inst_len + self._data_indices.resp_len - 1,  # Account for autoregressive shift
            batch_max_length=self.config.batch_max_length,
            drop_last_batch=self.config.drop_last_batch,

            rank=self.config.rank,
            num_replicas=self.config.num_replicas
        )

    def _load_batch(self, indices: np.ndarray):
        # Load instructions and responses
        assert self._data is not None and self._data_indices is not None
        raw = {k: [] for k in ["inst", "resp"]}
        for k, v in raw.items():
            start = getattr(self._data_indices, f"{k}_start")[indices]
            end = start + getattr(self._data_indices, f"{k}_len")[indices]
            for i in range(len(start)):
                v.append(self._data[start[i]: end[i]].astype(np.int32))

        # Form batch
        batch = {k: [] for k in ["inputs", "labels", "position_ids"]}
        seqlens_i, seqlens_o = [], []
        for i, o in zip(raw["inst"], raw["resp"]):
            # PrefixLM: last input token predict first output token, for compatibility with Causal LM frameworks (prefill-phase)
            # Inputs
            batch["inputs"].append(i)
            batch["inputs"].append(o[:-1])
            # Labels
            batch["labels"].append(np.full(len(i) - 1, dtype=i.dtype, fill_value=IGNORE_LABEL_ID) if self.config.target_only else i[1:])
            batch["labels"].append(o)
            # Position IDs
            batch["position_ids"].append(np.arange(len(i), dtype=np.int32))
            batch["position_ids"].append(np.arange(len(i), len(i) + len(o) - 1, dtype=np.int32))
            # Seqlens
            seqlens_i.append(len(i))
            seqlens_o.append(len(o) - 1)

        # Concat
        batch = {k: np.concatenate(v, dtype=np.int32) for k, v in batch.items()}
        # pad to fixed len
        pad_len = self.config.batch_max_length - batch["inputs"].shape[0]
        if pad_len > 0:
            pad_values = {
                "inputs": 0,  # FIXME: Pad with an arbitary token.
                "labels": IGNORE_LABEL_ID,
                "position_ids": 0,
            }
            for k in pad_values.keys():
                batch[k] = np.pad(batch[k], (0, pad_len), mode="constant", constant_values=pad_values[k])

        # Compute cu_seqlens
        seqlen_i, seqlen_o = np.array(seqlens_i, dtype=np.int32), np.array(seqlens_o, dtype=np.int32)
        tensors, scalars = compute_aux_seq_tensors_scalars(seqlen_i, seqlen_o, self.config.batch_max_length)

        # to tensor
        return {k: torch.from_numpy(v) for k, v in (batch | tensors).items()}, scalars

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1
        # TODO: Feature (Low Priority): Multithreaded data loading

        self._load_dataset_before_epoch_begin()

        assert self._sampler is not None
        for indices in self._sampler.iter():
            yield self._load_batch(indices)
