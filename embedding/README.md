# Working with Embedding Vectors


## Embedding Calculator

This script processes a sequence using the ESM2 model, computes embeddings and probabilities for mutations at
each position, and exports the results to a TSV file under `outputs` directory.

Options:
- `--sequence`: The sequence to process (default: "SARS-CoV-2-WildType"). Supports multiple arguments. Use --help to see available sequences.
- `--model`: The ESM model to use (default: "650M"). Supports multiple arguments. Use --help to see available models.

Example usage:
    python embed.py --sequence SARS-CoV-2-WildType --model 650M


## Exporting UMAP

This script performs UMAP dimensionality reduction on protein embeddings
from ESM2 models and visualizes the results with distinct colors for each species.
It supports different model sizes and distance metrics.

Options:
    --model-size: Specify the ESM model size to use (150M, 650M, or 3B). Support for multiple sizes.
    --metric: Choose the distance metric for UMAP. See UMAP documentation for options. Support for multiple metrics.

Example usage:
    python export_umap.py --model-size 650M [--metric euclidean --neighbors 5]