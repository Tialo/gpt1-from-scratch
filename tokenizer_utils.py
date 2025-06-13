import torch

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder


class Tokenizer:
    def __init__(self, hf_tokenizer: HFTokenizer):
        self.tokenizer = hf_tokenizer

    @classmethod
    def from_data(cls, train_data: list[str], vocab_size: int, max_len: int):
        tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = Sequence([NFKD(), StripAccents()])
        trainer = BpeTrainer(
            vocab_size=vocab_size,
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
        return cls(tokenizer)

    def encode(self, text: list[str] | str, include_eos: bool = True) -> torch.Tensor:
        if isinstance(text, str):
            encoded = torch.tensor(self.tokenizer.encode(text).ids)
            if not include_eos:
                return encoded[:-1]
        else:
            encoded = torch.tensor([t.ids for t in self.tokenizer.encode_batch(text)])
            if not include_eos:
                raise ValueError("only supported for not batched inputs")
        return encoded

    def decode(self, sequence: list[int] | torch.Tensor) -> str:
        if isinstance(sequence, torch.Tensor):
            if sequence.ndim == 2:
                if sequence.shape[0] != 1:
                    raise ValueError(
                        "Can't handle 2D tensor in decode with 1st dimension > 1"
                    )
                sequence = sequence[0]
            sequence = sequence.tolist()
        return self.tokenizer.decode(sequence, skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def change_max_len(self, max_len: int) -> None:
        self.tokenizer.enable_truncation(max_len)

    def save_pretrained(self, save_path: str) -> None:
        self.tokenizer.save(save_path)

    @classmethod
    def from_pretrained(cls, pretrained_path):
        return cls(HFTokenizer.from_file(pretrained_path))