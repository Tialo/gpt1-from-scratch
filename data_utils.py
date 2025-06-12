import os
import json
import argparse
from typing import TYPE_CHECKING

import torch
from datasets import load_dataset

if TYPE_CHECKING:
    from tokenizers import Tokenizer


def download_data(save_path: str, data_fraction: float, train_fraction: float = 0.8) -> None:
    data = load_dataset("JeanKaddour/minipile", split="train")["text"]
    data_size = int(len(data) * data_fraction)
    data = data[:data_size]
    train_size = int(len(data) * train_fraction)
    train_data = data[:train_size]
    val_data = data[train_size:]

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train.json"), "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_path, "val.json"), "w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)


def get_data_batch_iterator(
    data: list[str], tokenizer: "Tokenizer", batch_size: int = 16
):
    for i in range(0, len(data), batch_size):
        batch_data = data[i : i + batch_size]
        encoding = tokenizer.encode_batch(batch_data)

        yield (
            torch.tensor([b.ids for b in encoding]),
            torch.tensor([b.attention_mask for b in encoding]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset for training or evaluation"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data",
        help="Path to save the dataset files",
    )
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.5,
        help="Fraction of the dataset to download (default: '0.5')",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )
    args = parser.parse_args()
    download_data(args.save_path, args.data_fraction, args.train_fraction)
