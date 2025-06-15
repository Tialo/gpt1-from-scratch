import os
import math
import random
import argparse
import time
from functools import partial
from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from generator import Generator
from gpt import GPT
from tokenizer_utils import Tokenizer
from data_utils import DataLoader


if TYPE_CHECKING:
    from gpt import GPTConfig


@dataclass
class TrainConfig:
    data_fraction: float = 1.0
    batch_size: int = 8
    epochs: int = 1
    base_lr: float = 2.5e-4
    warmup_fraction: float = 0.06
    accumulation_steps: int = 8
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
    if not os.path.isdir("shards"):
        raise FileNotFoundError("Shards not found, please run python data_utils.py first")

    train_loader = DataLoader("train", batch_size=config.batch_size, max_len=gpt_config.max_len)
    val_loader = DataLoader("val", batch_size=config.batch_size, max_len=gpt_config.max_len)
    # we just drop those last batches that didn't fit into accumulation_steps 
    train_epoch_steps = train_loader.n_batches // config.accumulation_steps
    # just make less steps if data_fraction is not 1.0
    orig_train_epochs_steps = train_epoch_steps
    train_epoch_steps = int(train_epoch_steps * config.data_fraction)
    train_steps = train_epoch_steps * config.epochs
    warmup_steps = int(train_steps * config.warmup_fraction)
    print(f"{train_loader.n_batches:,} batches in train dataset")
    print(f"{orig_train_epochs_steps:,} steps will could done during train")
    print(f"{train_epoch_steps:,} steps will be done during train (after using data_fraction={config.data_fraction})")
    print(f"{val_loader.n_batches:,} batches in val dataset")

    tokenizer = Tokenizer.from_pretrained("tokenizer.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(gpt_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model with {n_params:,} parameters loaded.")
    generator = Generator(
        model,
        tokenizer.token_to_id("[START]"),
        tokenizer.token_to_id("[END]"),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id("[PAD]"),
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
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


@torch.no_grad
def calculate_eval_loss(
    model,
    val_loader: DataLoader,
    val_batches: int,
    criterion,
    device,
):
    val_loss = 0
    for _ in range(val_batches):
        inputs, labels = val_loader.get_batch()
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        val_loss += loss.item()
    return val_loss / val_batches
    

def train_one_epoch(
    model,
    train_loader: DataLoader,
    train_epoch_steps: int,
    val_loader: DataLoader,
    val_batches: int,
    accumulation_steps: int,
    criterion,
    opt,
    scheduler,
    generator: Generator,
    tokenizer: Tokenizer,
    device,
):
    step_losses = []
    val_losses = []

    epoch_start_time = time.time()
    steps_digits = len(str(train_epoch_steps))

    for step_index in range(train_epoch_steps):
        if step_index % 500 == 0 or step_index + 1 == train_epoch_steps:
            model.eval()
            torch.cuda.empty_cache()
            val_loss = calculate_eval_loss(
                model=model,
                val_loader=val_loader,
                val_batches=val_batches,
                criterion=criterion,
                device=device,
            )
            val_losses.append(val_loss)
            print(
                f"Step: [{str(step_index + 1).rjust(steps_digits)}/{train_epoch_steps}]",
                f"Validation loss: {val_loss:.4f}",
                sep=" | ",
            )
        if step_index > 0 and step_index % 1000 == 0 or step_index + 1 == train_epoch_steps:
            print(
                f"Step: [{str(step_index + 1).rjust(steps_digits)}/{train_epoch_steps}]",
                "Generations:",
                sep=" | "
            )
            for inputs in [
                "Hello, my name is",
                "I am a language model"
            ]:
                encoded = tokenizer.encode(inputs, add_end_token=False).to(device)
                generation = generator.generate(encoded, max_tokens=128)
                decoded = tokenizer.decode(generation, skip_special_tokens=False)
                print(
                    f"Inputs: {inputs}",
                    f"Generated: {decoded}",
                    sep='\n'
                )

        model.train()
        accumulated_loss = 0
        accumulation_start_time = time.time()

        for _ in range(accumulation_steps):
            inputs, labels = train_loader.get_batch()
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ) / accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        accumulation_end_time = time.time()
        accumulation_elapsed_time = accumulation_end_time - accumulation_start_time
        tokens_processed = accumulation_steps * train_loader.batch_size * train_loader.max_len
        tokens_in_a_second = tokens_processed / accumulation_elapsed_time
        current_lr = scheduler.get_last_lr()[0]
        elapsed_time = accumulation_end_time - epoch_start_time

        print(
            f"Step: [{str(step_index + 1).rjust(steps_digits)}/{train_epoch_steps}]",
            f"Step loss: {accumulated_loss:.4f}",
            f"lr: {current_lr:.3e}",
            f"elapsed: {format_elapsed_time(elapsed_time)}",
            f"tokens/sec: {tokens_in_a_second:.3f}",
            sep=" | ",
        )

        opt.step()
        opt.zero_grad()
        scheduler.step()
        step_losses.append(accumulated_loss)

    return step_losses, val_losses


def train_main(
    config: TrainConfig, gpt_config: "GPTConfig", save_path: str
):
    if os.path.isdir(save_path):
        raise RuntimeError(
            f"Directory {save_path} already exists, can't train model and save it there."
        )
    prep = prepare_training(config, gpt_config)
    train_loader = prep["train_loader"]
    val_loader = prep["val_loader"]
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
            {"params": no_decay_parameters, "weight_decay": 0.0},
            {"params": decay_parameters, "weight_decay": 0.01},
        ],
        lr=config.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    step_losses = []
    val_losses = []

    for e in range(config.epochs):
        print(f"Epoch: [{e + 1}/{config.epochs}]")
        e_step_losses, e_val_losses = train_one_epoch(
            model=model,
            train_loader=train_loader,
            train_epoch_steps=train_epoch_steps,
            val_loader=val_loader,
            val_batches=30,
            accumulation_steps=config.accumulation_steps,
            criterion=criterion,
            opt=opt,
            scheduler=scheduler,
            generator=generator,
            tokenizer=tokenizer,
            device=device,
        )
        step_losses.extend(e_step_losses)
        val_losses.extend(e_val_losses)
        torch.cuda.empty_cache()

    os.mkdir(save_path)
    model.save_pretrained(save_path)
    print(f"Model successfully saved at {save_path}!")

    return {
        "train_losses": step_losses,
        "val_losses": val_losses,
    }

def create_train_config_from_args(args) -> TrainConfig:
    return TrainConfig(
        data_fraction=args.data_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_lr=args.base_lr,
        warmup_fraction=args.warmup_fraction,
        accumulation_steps=args.accumulation_steps,
        seed=args.seed,
    )


def create_gpt_config_from_args(args, tokenizer) -> "GPTConfig":
    from gpt import GPTConfig

    return GPTConfig(
        vocab_size=tokenizer.vocab_size,
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
        "--data_fraction",
        type=float,
        default=1,
        help="Fraction of data used (default: 1)",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    train_group.add_argument(
        "--base_lr", type=float, default=2.5e-4, help="Base learning rate (default: 2.5e-4)"
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
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    train_group.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    model_group = parser.add_argument_group("GPT Model Configuration")
    model_group.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)"
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of decoder layers (default: 12)",
    )
    model_group.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads in each decoder layers (default: 12)",
    )
    model_group.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="Feed-forward network dimension (default: 3072)",
    )
    model_group.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
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
    if not os.path.isfile("tokenizer.json"):
        raise FileNotFoundError("tokenizer.json not found, please run python data_utils.py first")
    _tokenizer = Tokenizer.from_pretrained("tokenizer.json")
    _train_config = create_train_config_from_args(_args)
    _gpt_config = create_gpt_config_from_args(_args, _tokenizer)

    train_main(
        _train_config,
        _gpt_config,
        _args.save_path,
    )
