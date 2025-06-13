import os
import re
import json
import argparse
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
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


def download_data(save_path: str, data_samples: int | None = None, train_fraction: float = 0.8, remove_non_ascii: bool = True) -> None:
    print("Loading data")
    split = "train" if data_samples is None else f"train[:{data_samples}]"
    dataset = load_dataset("m-a-p/FineFineWeb-test", split=split)
    if remove_non_ascii:
        clean_text_f = clean_text
        print("Cleaning data")
    else:
        clean_text_f = lambda x: x

    dataset = [cleaned_text for sample in dataset if (cleaned_text := clean_text_f(sample["text"])) is not None]
    train_size = int(len(dataset) * train_fraction)
    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    print("Saving data")
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
        yield tokenizer.encode(batch_data)


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
        "--data_samples",
        type=int,
        default=None,
        help="Number of samples to download from the dataset (default: all)",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--remove_non_ascii",
        action="store_true",
        help="Remove non-ascii characters from the dataset (default: True)",
    )
    args = parser.parse_args()
    download_data(
        args.save_path,
        args.data_samples,
        args.train_fraction,
        args.remove_non_ascii,
    )
