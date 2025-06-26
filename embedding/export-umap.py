"""
This script performs UMAP dimensionality reduction on protein embeddings
from ESM2 models and visualizes the results with distinct colors for each species.
It supports different model sizes and distance metrics.

Options:
    --model-size: Specify the ESM model size to use (150M, 650M, or 3B). Support for multiple sizes.
    --metric: Choose the distance metric for UMAP. See UMAP documentation for options. Support for multiple metrics.

Example usage:
    python export-umap.py --model-size 650M [--metric euclidean --neighbors 5]
"""

import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import argparse

# Map model size to embedding file paths
MODEL_NAME_MAP = {
    "150M": "data/embedded-seqs/cov-wt-embed-150M.npz",
    "650M": "data/embedded-seqs/cov-wt-embed-150M.npz",
    "3B": "data/embedded-seqs/cov-wt-embed-3B.npz",
}

SCATTER_SHAPES = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a sequence with ESM2.")
    parser.add_argument(
        "--model-size",
        type=str,
        nargs="*",
        choices=MODEL_NAME_MAP.keys(),
        default=["650M"],
        help="ESM model size to use: 150M, 650M, or 3B.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="*",
        default=["euclidean"],
        help="Distance metric to use for UMAP. (Defaults to 'euclidean')",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=4,
        help="Number of neighbors for UMAP. (Defaults to 4)",
    )
    args = parser.parse_args()

    # Validate arguments
    assert len(args.model_size) > 0, "At least one model size must be specified."
    assert len(args.metric) > 0, "At least one metric must be specified."
    for model_size in args.model_size:
        assert model_size in MODEL_NAME_MAP, f"Invalid model size: {model_size}"
    assert args.neighbors > 0, "Number of neighbors must be a positive integer."

    for model_size in args.model_size:
        # Load the embedding files
        data = np.load(MODEL_NAME_MAP[model_size], allow_pickle=True)
        labels = data["labels"]
        embeddings = data["embeddings"]

        # print("Labels shape:", labels.shape)
        # print("Embeddings shape:", embeddings.shape)

        # Remove rows with NaN values in embeddings
        nan_filter = np.isnan(embeddings).any(axis=1)
        # print(f"# of rows with NaN values in embeddings: {nan_filter.sum()}")

        # Filter out NaN values
        labels = labels[~nan_filter]
        embeddings = embeddings[~nan_filter]

        # Collect unique labels, with "Human" as the first label
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != "Human"]
        unique_labels = ["Human"] + unique_labels.tolist()
        # print("Unique labels:", unique_labels)

        for metric in args.metric:
            print(f"Processing model size: {model_size}, metric: {metric}")
            # Compute UMAP
            reducer = umap.UMAP(
                n_neighbors=args.neighbors,
                min_dist=0.01,
                spread=1.0,
                metric=metric,
                random_state=42,
            )
            X_umap = reducer.fit_transform(embeddings)

            # Draw UMAP
            plt.figure(figsize=(10, 14))
            for i, label in enumerate(unique_labels):
                label_mask = labels == label
                plt.scatter(
                    X_umap[label_mask, 0],  # type: ignore
                    X_umap[label_mask, 1],  # type: ignore
                    s=15,
                    alpha=0.5,
                    label=label,
                    marker=SCATTER_SHAPES[i % len(SCATTER_SHAPES)],
                )

            # plt.legend(title="Data Source")
            plt.title("UMAP Projection (Distinct Color per Source Category)")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.grid(True)
            plt.legend(
                title="Species",
                bbox_to_anchor=(0.5, -0.15),  # Centered below the plot
                loc="upper center",
                borderaxespad=0.0,
                ncol=3,
            )
            plt.tight_layout()

            ax_orig = plt.gca()
            xlim = ax_orig.get_xlim()
            ylim = ax_orig.get_ylim()

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            plt.savefig(f"outputs/umap_{metric}_{model_size}_n{args.neighbors}.png", dpi=600)
            plt.show()
