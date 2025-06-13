import heapq
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from gpt import GPT


class Generator:
    def __init__(
        self, gpt: "GPT", sos_token_id: int, eos_token_id: int
    ):
        self.gpt = gpt
        self.max_len = gpt.config.max_len
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    @torch.no_grad
    def _get_top_tokens(
        self, generated: torch.Tensor, n_beams: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        generated - (batch_size, current_seq_len)
        """
        # (batch_size, current_seq_len, vocab_size)
        logits = self.gpt(generated)
        logits = logits[:, -1]  # (batch_size, vocab_size, )
        token_probs = logits.log_softmax(dim=-1)
        top_tokens = torch.topk(token_probs, k=n_beams, sorted=False, dim=-1)
        return top_tokens.indices, top_tokens.values  # each (batch_size, n_beams)

    @staticmethod
    def _get_normalized_score(
        beam: tuple[list[int], float], alpha: float = 0.6
    ) -> float:
        # 6.1 We used beam search with a beam size of 4 and length penalty α = 0.6
        tokens, score = beam
        return score * 6**alpha / (5 + len(tokens)) ** alpha

    def generate(
        self, tokens: torch.Tensor, max_tokens: int = 20, n_beams: int = 5
    ) -> torch.Tensor:
        if tokens.dim() == 2:
            if tokens.size(0) != 1:
                raise ValueError("batch_size > 1 is not supported...")
            tokens = tokens.squeeze(0)

        tokens_until_max_len = self.max_len - len(tokens)
        max_tokens = min(tokens_until_max_len, max_tokens)

        beams: list[tuple[list[int], float]] = [(tokens.tolist(), 0.0)]

        for _ in range(max_tokens):  # current_seq_len = len(tokens) + _
            candidate_beams = []
            tokens_to_process = []
            probabilities_to_process = []
            for generated, probability in beams:
                if generated[-1] == self.eos_token_id:
                    candidate_beams.append((generated, probability))
                else:
                    tokens_to_process.append(generated)
                    probabilities_to_process.append(probability)

            if not tokens_to_process:
                break

            # (n_beams - len(finished_beams), current_seq_len)
            batched_tokens_to_process = torch.tensor(tokens_to_process).to(tokens.device)  
            top_tokens, proba = self._get_top_tokens(batched_tokens_to_process, n_beams)
            for active_beam_index in range(len(tokens_to_process)):
                base_generated = tokens_to_process[active_beam_index]
                base_probability = probabilities_to_process[active_beam_index]
                for beam_index in range(n_beams):
                    new_token = top_tokens[active_beam_index, beam_index].item()
                    new_probabiliy = proba[active_beam_index, beam_index].item()
                    candidate_beams.append(
                        (
                            base_generated + [new_token],
                            base_probability + new_probabiliy,
                        )
                    )

            beams = heapq.nlargest(
                n_beams, candidate_beams, key=self._get_normalized_score
            )

        [best_generation, _] = max(beams, key=self._get_normalized_score)
        return torch.tensor(best_generation)
