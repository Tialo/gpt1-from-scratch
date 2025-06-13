import os
import json
import math
import random
import argparse
import time
from functools import partial
from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch
import numpy as np

from generator import Generator
from gpt import GPT
from loss import LabelSmoothingLoss
from tokenizer_utils import Tokenizer
from data_utils import get_data_batch_iterator


if TYPE_CHECKING:
    from gpt import GPTConfig


@dataclass
class TrainConfig:
    train_path: str = "data/train.json"
    val_path: str = "data/val.json"
    tokenizer_path: str = "tokenizer.json"
    data_fraction: float = 1.0
    batch_size: int = 64
    epochs: int = 1
    base_lr: float = 1.5e-4
    warmup_fraction: float = 0.06
    accumulation_steps: int = 4
    label_smoothing: float = 0.1
    seed: int = 42


def set_seed(seed: int | None = 42):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_training(config: TrainConfig, gpt_config: "GPTConfig"):
    set_seed(config.seed)
    if not os.path.isfile(config.train_path):
        raise FileNotFoundError(f"Training data file {config.train_path} not found.")
    if not os.path.isfile(config.val_path):
        raise FileNotFoundError(f"Validation data file {config.val_path} not found.")
    with open(config.train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(config.val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_data = train_data[:int(len(train_data) * config.data_fraction)]
    val_data = val_data[:int(len(val_data) * config.data_fraction)]
    print(f"{len(train_data):,} rows in train dataset")
    print(f"{len(val_data):,} rows in val dataset")
    train_epoch_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
    train_epoch_steps = (
        train_epoch_batches + config.accumulation_steps - 1
    ) // config.accumulation_steps
    train_steps = train_epoch_steps * config.epochs
    warmup_steps = int(train_steps * config.warmup_fraction)

    if not os.path.isfile(config.tokenizer_path):
        tokenizer = Tokenizer.from_data(train_data, vocab_size=gpt_config.vocab_size, max_len=gpt_config.max_len)
        tokenizer.save_pretrained(config.tokenizer_path)
    else:
        tokenizer = Tokenizer.from_pretrained(config.tokenizer_path)
        tokenizer.change_max_len(gpt_config.max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(gpt_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model with {n_params:,} parameters loaded.")
    generator = Generator(
        model,
        tokenizer.token_to_id("[START]"),
        tokenizer.token_to_id("[END]"),
    )
    criterion = LabelSmoothingLoss(
        ignore_index=tokenizer.token_to_id("[PAD]"),
        smoothing=config.label_smoothing,
    )
    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_epoch_batches": train_epoch_batches,
        "train_epoch_steps": train_epoch_steps,
        "warmup_steps": warmup_steps,
        "tokenizer": tokenizer,
        "device": device,
        "model": model,
        "generator": generator,
        "criterion": criterion,
    }


# https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/optimization.py#L132
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def train_one_epoch(
    model,
    data_iterator,
    criterion,
    opt,
    scheduler,
    device,
    train_epoch_batches,
    train_epoch_steps,
    accumulation_steps,
    global_step,
    epoch_index,
    n_epochs,
):
    model.train()
    epoch_loss_history = []
    accumulated_loss = 0
    step_indices = []
    step_losses = []
    
    epoch_start_time = time.time()
    steps_digits = len(str(train_epoch_steps))
    print(f"Epoch: [{epoch_index + 1}/{n_epochs}]")
    
    for batch_index, tokens in enumerate(data_iterator, 1):
        inputs = tokens[:, :-1].to(device)
        labels = tokens[:, 1:].to(device)

        logits = model(inputs)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ) / accumulation_steps

        epoch_loss_history.append(loss.item())
        accumulated_loss += loss.item()
        loss.backward()

        if batch_index % accumulation_steps == 0 or batch_index == train_epoch_batches:
            # log all steps
            step_index = batch_index // accumulation_steps
            current_lr = scheduler.get_last_lr()[0]
            elapsed_time = time.time() - epoch_start_time
            print(
                f"Step: [{str(step_index).rjust(steps_digits)}/{train_epoch_steps}]",
                f"Step loss: {accumulated_loss:.4f}",
                f"lr: {current_lr:.2e}",
                f"elapsed: {format_elapsed_time(elapsed_time)}",
                sep=" | ",
            )
            opt.step()
            opt.zero_grad()
            scheduler.step()
            step_indices.append(global_step)
            step_losses.append(accumulated_loss)
            accumulated_loss = 0
            global_step += 1

    return (
        sum(epoch_loss_history) / len(epoch_loss_history),
        global_step,
        step_indices,
        step_losses,
    )


@torch.no_grad
def validate_one_epoch(model, data_iterator, criterion, device):
    model.eval()
    epoch_val_loss_history = []
    for tokens in data_iterator:
        inputs = tokens[:, :-1].to(device)
        labels = tokens[:, 1:].to(device)

        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        epoch_val_loss_history.append(loss.item())
    return sum(epoch_val_loss_history) / len(epoch_val_loss_history)


def cherry_pick_generation(val_data, tokenizer, generator, n_picks, device):
    print("Cherry picked generations:")
    for i in range(n_picks):
        tokens = tokenizer.encode(val_data[i], include_eos=False)
        generated = generator.generate(
            tokens.to(device),
            max_tokens=len(tokens) + 50,
        )
        print(
            f"Input:     {tokenizer.decode(tokens)}",
            f"Generated: {tokenizer.decode(generated)}\n",
            sep="\n",
        )


def train_main(
    config: TrainConfig, gpt_config: "GPTConfig", save_path: str
):
    if os.path.isdir(save_path):
        raise RuntimeError(
            f"Directory {save_path} already exists, can't train model and save it there."
        )
    prep = prepare_training(config, gpt_config)
    train_data = prep["train_data"]
    val_data = prep["val_data"]
    train_epoch_batches = prep["train_epoch_batches"]
    train_epoch_steps = prep["train_epoch_steps"]
    warmup_steps = prep["warmup_steps"]
    tokenizer = prep["tokenizer"]
    device = prep["device"]
    model = prep["model"]
    generator = prep["generator"]
    criterion = prep["criterion"]

    # The learning rate was increased linearly from zero over the
    # first 2000 updates and annealed to 0 using a cosine schedule
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_epoch_steps * config.epochs,
        num_cycles=0.5,
    )
    # We also employed a modified version of L2 regularization
    # proposed in https://arxiv.org/abs/1711.05101
    # with w = 0.01 on all non bias or gain weights.
    no_decay_parameters, decay_parameters = model.get_splitted_params_for_opt()
    opt = torch.optim.AdamW(
        [
            {"params": no_decay_parameters},
            {"params": decay_parameters, "weight_decay": 0.01},
        ],
        lr=config.base_lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    global_step = 0
    epoch_indices = []
    epoch_train_losses = []
    epoch_val_losses = []
    step_indices = []
    step_train_losses = []

    for e in range(config.epochs):
        train_iterator = get_data_batch_iterator(
            train_data,
            tokenizer,
            batch_size=config.batch_size,
        )
        epoch_train_loss_avg, global_step, step_x, step_losses = train_one_epoch(
            model,
            train_iterator,
            criterion,
            opt,
            scheduler,
            device,
            train_epoch_batches,
            train_epoch_steps,
            config.accumulation_steps,
            global_step,
            e,
            config.epochs,
        )
        epoch_indices.append(e)
        epoch_train_losses.append(epoch_train_loss_avg)
        step_indices.extend(step_x)
        step_train_losses.extend(step_losses)
        torch.cuda.empty_cache()

        cherry_pick_generation(val_data, tokenizer, generator, 4, device)
        torch.cuda.empty_cache()
        print(f"\nTrain loss: {epoch_train_loss_avg:.4f}", end=" | ", flush=True)
        val_iterator = get_data_batch_iterator(
            val_data,
            tokenizer,
            batch_size=2 * config.batch_size,
        )
        epoch_val_loss_avg = validate_one_epoch(
            model,
            val_iterator,
            criterion,
            device,
        )
        epoch_val_losses.append(epoch_val_loss_avg)
        torch.cuda.empty_cache()
        print(f"Valid loss: {epoch_val_loss_avg:.4f}\n")
        print()

    os.mkdir(save_path)
    model.save_pretrained(save_path)
    print(f"Model successfully saved at {save_path}!")

    return {
        "epoch_train_loss": (epoch_indices, epoch_train_losses),
        "epoch_val_loss": (epoch_indices, epoch_val_losses),
        "step_train_loss": (step_indices, step_train_losses),
    }


def create_train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer_path=args.tokenizer_path,
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_lr=args.base_lr,
        warmup_fraction=args.warmup_fraction,
        accumulation_steps=args.accumulation_steps,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )


def create_gpt_config_from_args(args) -> "GPTConfig":
    from gpt import GPTConfig

    return GPTConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_len=args.max_len,
        dropout=args.dropout,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model")

    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--train_path",
        type=str,
        default="data/train.json",
        help="Path to training data file (default: data/train.json)",
    )
    train_group.add_argument(
        "--val_path",
        type=str,
        default="data/val.json",
        help="Path to validation data file (default: data/val.json)",
    )
    train_group.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer file. If not exists it will be built using train data (default: tokenizer.json)",
    )
    train_group.add_argument(
        "--data_fraction",
        type=float,
        default=0.4,
        help="Fraction of data used (default: 0.4)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Batch size for training (default: 48)",
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    train_group.add_argument(
        "--base_lr", type=float, default=1.5e-4, help="Base learning rate (default: 1.5e-4)"
    )
    train_group.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.06,
        help="Fraction of training steps for warmup (default: 0.06)",
    )
    train_group.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    train_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (default: 0.1)",
    )
    train_group.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    model_group = parser.add_argument_group("GPT Model Configuration")
    model_group.add_argument(
        "--vocab_size", type=int, default=8192, help="Vocabulary size (default: 8192)"
    )
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Embedding dimension (default: 512)"
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of decoder layers (default: 8)",
    )
    model_group.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads in each decoder layers (default: 8)",
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=2048,
        help="Feed-forward network dimension (default: 2048)",
    )
    model_group.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )
    train_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout value (default: 0.1)",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="model",
        help="Path to save the trained model (default: model)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()

    _train_config = create_train_config_from_args(_args)
    _gpt_config = create_gpt_config_from_args(_args)

    train_main(
        _train_config,
        _gpt_config,
        _args.save_path,
    )
