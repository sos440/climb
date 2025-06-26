"""
This script processes a sequence using the ESM2 model, computes embeddings and probabilities for mutations at
each position, and exports the results to a TSV file under `outputs` directory.

Options:
- `--sequence`: The sequence to process (default: "SARS-CoV-2-WildType"). Supports multiple arguments. Use --help to see available sequences.
- `--model`: The ESM model to use (default: "650M"). Supports multiple arguments. Use --help to see available models.

Example usage:
    python embed.py --sequence SARS-CoV-2-WildType --model 650M
"""

import torch
import esm
import re
import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


# Constants
STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
PATH_SETTING = "settings.json"
PATH_SEQ_METADATA = "metadata/sequences.json"


@dataclass
class PositionResult:
    """Stores the results for each position in the sequence."""

    position: int
    original_aa: str
    embeddings: Dict[str, torch.Tensor]  # {AA: embedding}
    probabilities: Dict[str, float]  # {AA: probability}


def get_embeddings_at(
    sequence: str,
    pos: int,
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    window_size: int = 1022,
    stride: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Compute embeddings for all possible mutations at a given position in the sequence.

    Args:
        sequence: Sequence to process
        pos: Position to process
        model: ESM2 model
        alphabet: (Not used in this function, but required for compatibility)
        batch_converter: Batch converter for the model
        window_size: Window size
        stride: Stride
    """
    device = next(model.parameters()).device

    # Possible mutations
    mut_seqs = [sequence[:pos] + key + sequence[pos + 1 :] for key in STANDARD_AAS]
    _, _, mut_tokens = batch_converter([(key, seq) for key, seq in zip(STANDARD_AAS, mut_seqs)])
    mut_tokens = mut_tokens.to(device)

    token_len = mut_tokens.size(1)
    batch_size = mut_tokens.size(0)

    # Sum of all embeddings and overcounts (to handle overlapping windows)
    all_embeddings = torch.zeros((batch_size, token_len, model.embed_dim), device=device)
    all_overcounts = torch.zeros((token_len,), device=device)

    # Perform sliding window over the tokens
    for start in range(0, token_len, stride):
        end = min(start + window_size, token_len)
        window_tokens = mut_tokens[:, start:end]
        all_overcounts[start:end] += 1

        # Compute embeddings
        with torch.no_grad():
            results = model(window_tokens, repr_layers=[model.num_layers], return_contacts=False)
            embeddings = results["representations"][model.num_layers]
        # Update total embeddings
        all_embeddings[:, start:end] += embeddings

    # Reshape all_overcounts to (1, token_len, 1) for broadcasting
    all_overcounts = all_overcounts.view(1, -1, 1)
    # Take care of the overcounts induced by the window overlap
    normalized_embeddings = all_embeddings / all_overcounts
    # Remove CLS and EOS tokens
    normalized_embeddings = normalized_embeddings[:, 1:-1, :]
    # Compute the mean of the embeddings
    final_embeddings = torch.mean(normalized_embeddings, dim=1)
    # Build a dictionary of embeddings and return it
    dict_embeddings = {key: v for key, v in zip(STANDARD_AAS, final_embeddings)}
    return dict_embeddings


def get_probs_at(
    sequence: str,
    pos: int,
    model: Any,
    alphabet: Any,
    batch_converter: Any,
) -> dict[str, float]:
    """
    Compute probabilities of mutations at a given position in the sequence.
    """

    device = next(model.parameters()).device
    # New position after tokenization, because CLS and EOS tokens are added
    new_pos = pos + 1

    # Tokenize the sequence
    _, _, tokens = batch_converter([("masked", sequence)])
    tokens[0, new_pos] = alphabet.mask_idx
    tokens = tokens.to(device)

    # Compuate the left and right boundaries of the window containing the position
    seq_len = tokens.size(1)
    max_len = 1022
    if seq_len < max_len:
        left = 0
        right = seq_len
    else:
        left, right = new_pos - (max_len // 2), new_pos + (max_len // 2)
        if left < 0:
            right = max_len
            left = 0
        if right > seq_len:
            left = seq_len - max_len
            right = seq_len

    # Extract the window
    tokens = tokens[:, left:right]

    # Compute the logits
    with torch.no_grad():
        results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)

    # Compute the probabilities
    logits = results["logits"][0, new_pos - left]
    probs = torch.softmax(logits, dim=0)

    # Get probabilities for specific tokens
    mut_aas = [key for key in "ACDEFGHIKLMNPQRSTVWY" if key != sequence[pos]]
    cond_prob_total = 0
    token_probs = {}
    for token in mut_aas:
        token_idx = alphabet.get_idx(token)
        token_probs[token] = probs[token_idx].item()
        cond_prob_total += token_probs[token]
    # Normalize the probabilities
    for token in mut_aas:
        token_probs[token] /= cond_prob_total

    return token_probs


def get_result_at(
    sequence: str,
    pos: int,
    model: Any,
    alphabet: Any,
    batch_converter: Any,
) -> PositionResult:
    """
    Get the embeddings and probabilities for a specific position in the sequence.

    Args:
        sequence: Sequence to process
        pos: Position to process
        model: ESM2 model
    """
    original_aa = sequence[pos]
    embeddings = get_embeddings_at(sequence, pos, model, alphabet, batch_converter)
    probabilities = get_probs_at(sequence, pos, model, alphabet, batch_converter)
    res = PositionResult(
        position=pos,
        original_aa=original_aa,
        embeddings=embeddings,
        probabilities=probabilities,
    )
    return res


def load_model_by_name(model_name: str):
    """
    Dynamically load an ESM model from esm.pretrained by its name.

    Args:
        model_name: Name of the model constructor in esm.pretrained, e.g. "esm2_t33_650M_UR50D".

    Returns:
        A tuple (model, alphabet).

    Raises:
        AttributeError: If the model name is not found in esm.pretrained.
    """
    try:
        load_fn = getattr(esm.pretrained, model_name)
    except AttributeError:
        valid = [fn for fn in dir(esm.pretrained) if not fn.startswith("__")]
        raise ValueError(f"Model '{model_name}' not found. Available models are: {', '.join(valid)}")
    return load_fn()


def export_embeddings_to_tsv(
    seq_data: Dict[str, Any],
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    model_name: str = "unknown",
):
    """
    Export embeddings and probabilities for mutations at each position in the sequence to a TSV file.
    """
    assert "sequence" in seq_data, "Sequence data is missing in the JSON file."
    assert "output-path" in seq_data, "Output path is missing in the JSON file."
    assert "starting-index" in seq_data, "Starting index is missing in the JSON file."

    # Read the values from the sequence data
    seq = re.sub(r"[^A-Z]+", "", seq_data["sequence"].upper())
    output_path = seq_data.get("output-path", "unknown")
    pos_offset = seq_data.get("starting-index", 0)

    # Generate the TSV filename with timestamp
    cur_path = Path(__file__).parent / "outputs" / output_path
    if not cur_path.exists():
        cur_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = cur_path / f"{timestamp}_{model_name}.tsv"
    print(f"Saving files to: {csv_filename}")

    # Export to TSV
    with open(csv_filename, "w", newline="") as csvfile:
        # TSV writer
        writer = csv.writer(csvfile, delimiter="\t")

        # Write header
        writer.writerow(["pos", "wt", "mut", "prob", "change", "change_l1", "change_l2"])

        # Process each position in the sequence
        for pos, orig_aa in tqdm(enumerate(seq), total=len(seq), desc="Processing positions"):

            # Apply the model to the sequence and get the results at each position
            res = get_result_at(seq, pos, model, alphabet, batch_converter)

            orig_pos = pos + pos_offset
            mut_aas = res.probabilities.keys()
            original_embedding = res.embeddings[orig_aa]

            for mut_aa in mut_aas:
                prob = res.probabilities[mut_aa]
                embedding = res.embeddings[mut_aa]

                # Calculate the semantic change in various ways
                cos_sim = torch.nn.functional.cosine_similarity(
                    original_embedding.unsqueeze(0), embedding.unsqueeze(0)
                ).item()
                l1_norm = torch.norm(original_embedding - embedding, p=1).item()
                l2_norm = torch.norm(original_embedding - embedding, p=2).item()

                # Write to CSV
                writer.writerow([orig_pos, orig_aa, mut_aa, prob, cos_sim, l1_norm, l2_norm])


# Test code
if __name__ == "__main__":
    # Load settings from JSON file if available
    settings = {
        "model-options": {
            "150M": {"name": "esm2_t30_150M_UR50D"},
            "650M": {"name": "esm2_t33_650M_UR50D"},
            "3B": {"name": "esm2_t36_3B_UR50D"},
        }
    }
    if Path(PATH_SETTING).exists():
        with open(PATH_SETTING, "r") as f:
            settings = settings | json.load(f)
    else:
        print(f"Settings file {PATH_SETTING} not found. Using default settings.")

    # Load the sequence data from JSON file
    with open(PATH_SEQ_METADATA, "r") as f:
        seqs_metadata = json.load(f)
        assert len(seqs_metadata) > 0, "Sequences file is empty or not found."

    # Read the model options
    model_options = settings.get("model-options", {})

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a sequence with ESM2.")
    parser.add_argument(
        "--sequence",
        type=str,
        nargs="*",
        default=["SARS-CoV-2-WildType"],
        help=f"Sequence data to process. (Available: {', '.join(seqs_metadata.keys())})",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["esm2_t33_650M_UR50D"],
        help=f"ESM model to use. (Available: {', '.join(model_options.keys())})",
    )
    args = parser.parse_args()

    # Validate sequence names
    for seq in args.sequence:
        assert seq in seqs_metadata, f"Sequence {seq} not found in esm_sequences.json"
        assert "sequence" in seqs_metadata[seq], "Sequence data is missing in the JSON file."

    # Process the model names (650M -> esm2_t33_650M_UR50D, etc.)
    models = []
    for model_name in args.model:
        if model_name in model_options:
            model_name = model_options[model_name].get("name", model_name)
        models.append(model_name)

    # Load the ESM2 model
    for model_name in models:
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model, alphabet = load_model_by_name(model_name)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        batch_converter = alphabet.get_batch_converter()
        print(f"Loaded model: {model_name}")

        # Change the mode to evaluation
        model.eval()

        # Check if GPU is available
        device = next(model.parameters()).device
        print(f"Using device: {device}")

        for seq_name in args.sequence:
            # Load the settings for the specified virus
            print(f"Processing sequence: {seq_name}")

            export_embeddings_to_tsv(
                seq_data=seqs_metadata[seq_name],
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
                model_name=model_name,
            )
