import os
import re
import argparse
import multiprocessing

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

from tokenizer_utils import Tokenizer


def clean_text(text: str) -> str | None:
    """
    Remove all non-ascii characters.
    Remove all samples where 20%+ of characters were non-ascii.
    """
    ascii_only = ''.join(char for char in text if ord(char) < 128)
    single_spaced = re.sub(r'\s+', ' ', ascii_only)
    cleaned_text = single_spaced.strip()
    if len(cleaned_text) / len(text) > 0.8:
        return cleaned_text
    return None


def download_data(
    data_samples: int | None = None,
    train_fraction: float = 0.8,
    remove_non_ascii: bool = True,
) -> tuple[list[str], list[str]]:
    print("Loading data")
    split = "train" if data_samples is None else f"train[:{data_samples}]"
    dataset = [
        sample["text"]
        for sample in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split=split)
    ]
    if remove_non_ascii:
        bar = tqdm(desc="Cleaning text", total=len(dataset))
        nproc = max(1, os.cpu_count() // 2)
        cleaned_dataset = []
        with multiprocessing.Pool(nproc) as pool:
            for text in pool.imap(clean_text, dataset, chunksize=16):
                if text is not None:
                    cleaned_dataset.append(text)
                bar.update(1)
        dataset = cleaned_dataset

    train_size = int(len(dataset) * train_fraction)
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]
    return train_data, val_data


class ShardsBuilder:
    def __init__(self, data_samples: int | None = None, train_fraction: float = 0.8, shard_size: int = int(1e8)):
        self.data_samples = data_samples
        self.train_fraction = train_fraction
        self.shard_size = shard_size

        self.shards_saved = 0

    def _build_shards(self, tokenizer: Tokenizer, dataset: list[str], shard_name: str):
        vocab_size = tokenizer.vocab_size
        if vocab_size < 65536:
            # enough to save token ids from 0 up to 65535
            # for shard_size = 1e8 each shard ~190MB
            np_dtype = np.uint16
        else:
            # from 0 up to ~4e9
            # ~380MB
            np_dtype = np.uint32

        curr_shard = np.full(self.shard_size, fill_value=vocab_size, dtype=np_dtype)
        buffer_offset = 0
        shards_saved = 0
        bar = tqdm(desc=f"Building {shard_name} shards", total=len(dataset))
        nproc = max(1, os.cpu_count() // 2)
        with multiprocessing.Pool(nproc) as pool:
            for sample_idx, tokens in enumerate(pool.imap(tokenizer.encode, dataset, chunksize=16)):
                if len(tokens) > self.shard_size - buffer_offset or sample_idx + 1 == len(dataset):
                    truncated_size = min(len(tokens), self.shard_size - buffer_offset)
                    leftover_size = max(0, len(tokens) - truncated_size)
                    curr_shard[buffer_offset:buffer_offset + truncated_size] = tokens[:truncated_size]
                    curr_shard = curr_shard[:buffer_offset + truncated_size]  # truncate shard if it is last

                    assert sum(curr_shard == vocab_size) == 0
                    np.save(f"shards/{shard_name}_{shards_saved}.npy", curr_shard)
                    shards_saved += 1
                    curr_shard = np.full(self.shard_size, fill_value=vocab_size, dtype=np_dtype)
                    curr_shard[:leftover_size] = tokens[truncated_size: truncated_size + leftover_size]
                    buffer_offset = leftover_size
                else:
                    curr_shard[buffer_offset: buffer_offset + len(tokens)] = tokens
                    buffer_offset += len(tokens)
                bar.update(1)
    
    def build_shards(self, tokenizer: Tokenizer, train_dataset: list[str], val_dataset: list[str]):
        os.makedirs("shards", exist_ok=True)
        self._build_shards(tokenizer, train_dataset, "train")
        self._build_shards(tokenizer, val_dataset, "val")


class DataLoader:
    def __init__(self, shards_name: str, batch_size: int, max_len: int):
        assert shards_name in ("train", "val")
        self.shards = [shard_name for shard_name in os.listdir("shards") if shards_name in shard_name]
        assert self.shards
        self.shard_idx = self.current_offset = self.n_batches = 0
        self.batch_size = batch_size
        self.max_len = max_len
        for shard_idx in range(len(self.shards)):
            shard = self._get_shard(shard_idx)
            self.n_batches += len(shard) // (self.batch_size * self.max_len + 1)
        self.shard = self._get_shard(self.shard_idx)

    def _inc_shard_idx(self):
        # cyclic iteration over [0, 1, ..., len(self.shards) - 1]
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)

    def _get_shard(self, shard_idx: int):
        shard = np.load(f"shards/{self.shards[shard_idx]}").astype(np.int32)
        return torch.tensor(shard, dtype=torch.long)

    def reset(self):
        if self.shard_idx != 0:
            self.shard = self._get_shard(0)
        self.shard_idx = 0
        self.current_offset = 0

    def get_batch(self):
        b, l = self.batch_size, self.max_len
        # drop all tokens from shard that don't fit into the batch
        if self.current_offset + b * l + 1 > len(self.shard):
            self._inc_shard_idx()
            self.shard = self._get_shard(self.shard_idx)
        batch = self.shard[self.current_offset: self.current_offset + b * l + 1]
        self.current_offset += b * l + 1
        inputs = batch[:-1].view(b, l)
        outputs = batch[1:].view(b, l)
        return inputs, outputs


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset for training or evaluation"
    )
    parser.add_argument(
        "--data_samples",
        type=int,
        default=1_000_000,
        help="Number of samples to download from the dataset (default: 1_000_000)",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Size of BPE tokenizer vocabulary (default: 32768)",
    )
    parser.add_argument(
        "--use_non_ascii",
        action="store_true",
        help="Use non ascii characters in dataset (removed by default)",
    )
    args = parser.parse_args()
    train_dataset, val_dataset = download_data(
        args.data_samples,
        args.train_fraction,
        not args.use_non_ascii,
    )
    tokenizer_path = "tokenizer.json"
    if os.path.isfile(tokenizer_path):
        tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_data(train_dataset, args.vocab_size)
        tokenizer.save_pretrained(tokenizer_path)

    shard_builder = ShardsBuilder(args.data_samples, args.train_fraction)
    shard_builder.build_shards(tokenizer, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
