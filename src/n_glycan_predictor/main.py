import esm

import os

from pathlib import Path

import polars as pl

import torch

from typing import Iterable


DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_esm_embeddings(
    model: torch.nn.Module,
    alphabet: torch.nn.Module,
    sequences: Iterable[tuple[str, str]]
) -> torch.Tensor:
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter(sequences)
    with torch.no_grad():
        results = model(batch_tokens.to(DEVICE), repr_layers = [33], return_contacts = True)
    token_representations = results["representations"][33]
    sequence_representations = [
        token_representations[i, 1 : tokens_len - 1].mean(0)
        for i, tokens_len
        in enumerate((batch_tokens != alphabet.padding_idx).sum(1))
    ]
    return torch.stack(sequence_representations)


def main() -> int:
    model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    model.to(DEVICE)
    alphabet.to(DEVICE)
    data_path = Path(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2), "data")).resolve()
    df = pl.read_csv(data_path / "data.csv")
    embeddings = get_esm_embeddings(model, alphabet, df[["id", "sequence"]].iter_rows())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())