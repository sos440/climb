"""
DEVELOPMENT USE ONLY

In the production environment, the processed outputs will be already included.

This script embeds all CoV sequences using the ESM2 model.

It reads the FASTA file containing the sequences, processes them in batches,
and saves the embeddings to a TSV file.
"""

import torch
import esm
import json
import argparse
import numpy as np
import os
from tqdm import tqdm
from typing import Any
from datetime import datetime
from esm import pretrained

# Map model size to ESM model name
MODEL_NAME_MAP = {
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
    "3B": "esm2_t36_3B_UR50D",
}
PATH_SEQ_JSON = "../outputs/cov_all.json"


def get_embeddings(
    batch: list[tuple[str, str]],
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    device: str,
    window_size: int = 1022,
    stride: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Get embeddings for a batch of sequences using ESM2 model.

    Args:
        batch: Sequence of tuples (key, sequence)
        model: ESM2 model
        alphabet: Alphabet
        batch_converter: Batch converter
        device: Device to use (e.g., "cuda" or "cpu")
        window_size: Window size
        stride: Stride
    """

    # Tokenize the batch
    labels, _, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)

    # Create a mask for the tokenized sequences
    # Exclude CLS, EOS, and PAD tokens from the valid mask
    valid = (
        (batch_tokens != alphabet.cls_idx) & (batch_tokens != alphabet.eos_idx) & (batch_tokens != alphabet.padding_idx)
    )

    # Initialize tensors to store embeddings and overcounts
    token_len = batch_tokens.size(1)
    batch_size = batch_tokens.size(0)
    all_embeddings = torch.zeros((batch_size, token_len, model.embed_dim), device=device)
    all_overcounts = torch.zeros((token_len,), device=device)

    for start in range(0, token_len, stride):
        end = min(start + window_size, token_len)
        window_tokens = batch_tokens[:, start:end]
        all_overcounts[start:end] += 1

        # Compute embeddings
        with torch.no_grad():
            results = model(window_tokens, repr_layers=[model.num_layers], return_contacts=False)
            embeddings = results["representations"][model.num_layers]
        # Update total embeddings
        all_embeddings[:, start:end] += embeddings

    all_overcounts = all_overcounts.unsqueeze(-1).float()
    norm_embeddings = all_embeddings / all_overcounts

    norm_embeddings = norm_embeddings * valid.unsqueeze(-1)
    counts = valid.sum(dim=1).unsqueeze(-1).float()
    norm_embeddings = norm_embeddings.sum(dim=1) / counts

    dict_embeddings = {key: embedding for key, embedding in zip(labels, norm_embeddings)}
    return dict_embeddings


# Embed all CoV sequences
if __name__ == "__main__":
    # Ensure the input file exists
    if not os.path.exists(PATH_SEQ_JSON):
        raise FileNotFoundError(f"Input file {PATH_SEQ_JSON} does not exist. Please run `fasta-to-json.py` first.")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a sequence with ESM2.")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=MODEL_NAME_MAP.keys(),
        default="650M",
        help="ESM model size to use: 150M, 650M, or 3B.",
    )
    args = parser.parse_args()

    assert args.model_size in MODEL_NAME_MAP, f"Invalid model size: {args.model_size}"

    model_name = MODEL_NAME_MAP[args.model_size]
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    batch_converter = alphabet.get_batch_converter()
    embed_dim = int(model.embed_dim)  # type: ignore

    # Change the mode to evaluation
    model.eval()

    # Check if GPU is available
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    with open(PATH_SEQ_JSON, "r") as f:
        cov_all = json.load(f)

    cov_all_list = list(cov_all.values())
    cov_all_labels = [entry["host"] for entry in cov_all_list]
    cov_all_embeddings = np.zeros((0, embed_dim), dtype=np.float32)

    # Generate the TSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_filename = f"../outputs/{timestamp}_cov_all_embed_{args.model_size}.npz"
    if not os.path.exists("../outputs"):
        os.makedirs("../outputs")

    for idx in tqdm(range(0, len(cov_all_list), 32)):
        batch = cov_all_list[idx : idx + 32]
        batch_seq = [(f"{i:08d}", entry["seq"]) for i, entry in enumerate(batch)]

        # Embed the batch
        batch_embeddings = get_embeddings(
            batch_seq,
            model,
            alphabet,
            batch_converter,
            device,  # type: ignore
        )

        # Convert to numpy array
        batch_embeddings_list = sorted(batch_embeddings.items(), key=lambda x: x[0])
        batch_embeddings = np.array(
            [embedding.cpu().numpy() for _, embedding in batch_embeddings_list], dtype=np.float32
        )
        cov_all_embeddings = np.vstack((cov_all_embeddings, batch_embeddings))

        # Save the embeddings to a .npz file
        np.savez_compressed(
            npz_filename,
            labels=cov_all_labels[: len(cov_all_embeddings)],
            embeddings=cov_all_embeddings,
        )
