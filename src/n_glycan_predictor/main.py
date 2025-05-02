import esm

import itertools

from pathlib import Path

import polars as pl

import time

import torch

from types import TracebackType

from typing import (
    Self,
    Sequence,
)



DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Timer:
    def __init__(self, msg: str) -> None:
        self.msg = msg
        self.start = time.perf_counter()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end = time.perf_counter()
        self.duration = self.end - self.start


def get_esm_embeddings(
    model: torch.nn.Module,
    alphabet: torch.nn.Module,
    sequences: Sequence[tuple[str, str]],
) -> torch.Tensor:
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter(sequences)
    with torch.no_grad():
        results = model(batch_tokens.to(DEVICE), repr_layers = [33], return_contacts = True)
    token_representations = results["representations"][33]
    sequence_representations = [
        token_representations[i, 1 : tokens_len - 1].mean(0)
        for i, tokens_len in enumerate((batch_tokens != alphabet.padding_idx).sum(1))
    ]
    return torch.stack(sequence_representations)


def main() -> int:
    with Timer("Loading model") as timer:
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model.to(DEVICE)
    print(f"Loaded model in {timer.duration:.2f}s")
    data_path = Path(__file__).parents[2] / "data"
    df = pl.read_csv(data_path / "data.csv")
    with Timer("Get embeddings") as timer:
        embeddings = get_esm_embeddings(
            model,
            alphabet,
            list(itertools.islice(df[["id", "sequence"]].iter_rows(), 1))
        )
    print(embeddings.shape, )
    print(len(next(itertools.islice(df[["id", "sequence"]].iter_rows(), 1))[1]))
    print(f"Got embeddings in {timer.duration:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())