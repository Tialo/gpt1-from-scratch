import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder


def build_tokenizer(train_data: list[str], max_len: int, save_path: str) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFKD(), StripAccents()])
    trainer = BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        # [SEP] used for Task-specific input transformations (3.3 in original paper)
        special_tokens=["[UNK]", "[PAD]", "[START]", "[END]", "[SEP]"],
        end_of_word_suffix="</w>",
    )
    tokenizer.train_from_iterator(train_data, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[START] $A [END]",
        pair="[START] $A [SEP] $B [END]",
        special_tokens=[
            ("[START]", tokenizer.token_to_id("[START]")),
            ("[END]", tokenizer.token_to_id("[END]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"))
    tokenizer.enable_truncation(max_len)
    tokenizer.decoder = BPEDecoder()
    tokenizer.save(save_path)
    return tokenizer


def get_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def decode(tokenizer: Tokenizer, sequence: list[int] | torch.Tensor) -> str:
    if isinstance(sequence, torch.Tensor):
        if sequence.ndim == 2:
            if sequence.shape[0] != 1:
                raise ValueError(
                    "Can't handle 2D tensor in decode with 1st dimension > 1"
                )
            sequence = sequence[0]
        sequence = sequence.tolist()
    return tokenizer.decode(sequence, skip_special_tokens=True)
