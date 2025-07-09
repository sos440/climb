"""
This code analyzes the rank distributions, calculates their statistics,
and visualize them in the form of box-whisker plots.

* This code uses outputs of `main.py`. You can run the following code to generate data to be analyzed.

    ```bash
    python main.py --escape "SARS-CoV-2-WildType>*" --result "SARS-CoV-2-WildType-*" --scores
    ```

    To analyze other data, modify the inputs and the settings accordingly.

* Rank distribution is automatically calculated when `main.py` exports scores.
    They are stored as metadata in JSON files.

* Modify the setting file to change the data to analyze.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import csv
import os
import modules.analyzer as an
from tqdm import tqdm
from scipy.stats import wilcoxon
from numpy.typing import ArrayLike


# Constants
SETTING_PATH = "draw-figures-settings.json"
DATASET_PATHS = ["outputs", "outputs-reproduce/cov-wt-outputs"]


# Read the settings
assert os.path.exists(SETTING_PATH), "Setting file does not exist!"
with open(SETTING_PATH, "r") as f:
    settings = json.load(f)

# Validate the schema
assert "erase-ticks" in settings, "Setting file is missing `erase-ticks` field."
assert "validation-sets" in settings, "Setting file is missing `validation-sets` field."
assert "datasets" in settings, "Setting file is missing `datasets` field."
assert "time" in settings, "Setting file is missing `time` field."

# Read all the `.json` files with valid schema into the dictionary
DATABASE: dict[str, dict] = {}
for path in DATASET_PATHS:
    for root, dirs, files in os.walk(path):
        for file in files:
            # Read the JSON file
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)
            # Validate the file schema
            for key in ["escape", "result", "virus", "strain", "times", "distributions"]:
                if key not in data:
                    continue
            # Create a unique serial to prevent duplicates
            serial = f"({data['escape']})-({data['result']})-({data['virus']})-({data['strain']})"
            if serial in DATABASE:
                print(f"Warning: Duplicate entry found. Skipping...")
                continue
            # Add to the dictionary
            DATABASE[serial] = data


def report_rank_stats(data: ArrayLike, range: int, sep: str = "\n") -> str:
    """
    Calculate and report the rank statistics for a given rank distribution.

    :param data: The rank distribution data as an array-like structure.
    :param range: The range of the rank distribution.
    :param sep: Separator for the output string.
    :return: A formatted string with the rank statistics.
    """
    mean = np.mean(data)
    std = np.std(data)
    min_val = np.min(data)
    q1, median, q3 = np.quantile(data, [0.25, 0.5, 0.75])
    max_val = np.max(data)
    auc = 1 - mean / range
    res_list = [
        f"Mean: {mean:.2f}",
        f"Std: {std:.2f}",
        f"Min: {min_val:.2f}",
        f"Q1: {q1:.2f}",
        f"Median: {median:.2f}",
        f"Q3: {q3:.2f}",
        f"Max: {max_val:.2f}",
        f"AUC: {auc:.4f}",
    ]
    return sep.join(res_list)


def export_plots(escape: str):
    """
    Export plots and statistics for the given escape.
    """
    # Read the settings
    t0 = settings["time"]
    values = list(entry["value"] for entry in settings["datasets"])
    labels = list(entry["label"] for entry in settings["datasets"])
    v2l_map = {entry["value"]: entry["label"] for entry in settings["datasets"]}  # Value to label map

    # Filter out the relevant entries from the database
    db_filtered = {}
    for entry in DATABASE.values():
        if entry["escape"] != escape:
            continue

        value = entry["result"]
        strain = entry["strain"]
        times = entry["times"]
        if value not in values:
            continue
        if t0 not in times:
            continue
        db_filtered[value] = entry["distributions"]

    # Make sure all the datasets are found:
    for value in values:
        if value not in db_filtered.keys():
            raise ValueError(f"Dataset `{value}` for the escape `{escape}` not found.")

    # Prepare CSCS/CAC data (enumerated using the `value` keys)
    num_entries = len(values)
    data_cscs = {value: np.array(db_filtered[value]["cscs"]) for value in values}
    data_cac = {value: np.array(db_filtered[value][f"t={t0}"]) for value in values}

    # Read the strain information, especially, the sequence length
    with open(an.STRAIN_DATA[strain]["sequence-path"], "r") as f:
        reader = csv.reader(f, delimiter=",")
        seq_len = 0
        for row in reader:
            if row[2] in "ACDEFGHIKLMNPQRSTVWY":
                seq_len += 1

    # Prepare output directory
    virus_prefix = an.STRAIN_DATA[strain]["prefix"]
    escape_prefix = an.ESCAPE_DATA[escape]["prefix"]
    output_dir = os.path.join("outputs-figures", virus_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ### PART 1: Export statistics ###
    def _ns_pt1_statistics():
        """
        Export the rank statistics for CSCS and CAC.
        This function generates a text file containing the rank statistics for each dataset.
        The statistics include mean, standard deviation, minimum, Q1, median, Q3,
        maximum, and AUC (Area Under the Curve).
        The output is saved in a text file with the prefix `rank-stats-`.
        """
        # Introduce a namespace
        filename = f"rank-stats-(e={escape_prefix})-(t:{t0}).txt"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w") as f:
            f.write(f"Escape: {escape}\n")
            f.write(f"Strain: {strain}\n")
            f.write(f"Evolutionary Time Parameter (T): {t0}\n")

            f.write("\n" + ("=" * 50) + "\n")

            for value in values:
                f.write("\n")
                f.write(f"Model-Output Dataset: {value}\n")

                f.write("CSCS:\n")
                f.write("    " + report_rank_stats(data_cscs[value], seq_len * 19, " / ") + "\n")

                f.write("CAC:\n")
                f.write("    " + report_rank_stats(data_cac[value], seq_len * 19, " / ") + "\n")

    _ns_pt1_statistics()

    ### PART 2: Export box-whisker plot ###

    def _ns_pt2_box_whisker():
        """
        Export a box-whisker plot comparing the rank distributions of CSCS and CAC.
        This function generates a box-whisker plot for each dataset in the `values` list.
        The plot shows the rank distributions for CSCS and CAC side by side.
        The box-whisker plot includes the mean points for both CSCS and CAC.
        The output is saved in a PDF file with the prefix `boxplot-`.
        """
        padding = 20

        # Prepare the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = np.arange(num_entries)
        width = 0.15

        # Draw the box and whiskers
        for i, value in enumerate(values):
            ax.boxplot(
                data_cscs[value],
                positions=[positions[i] - width],
                widths=0.25,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="black"),
            )
            ax.boxplot(
                data_cac[value],
                positions=[positions[i] + width],
                widths=0.25,
                patch_artist=True,
                boxprops=dict(facecolor="lightgreen"),
                medianprops=dict(color="black"),
            )

        # Draw the mean points
        mean_cscs = [np.mean(data_cscs[value]) for value in values]
        mean_cac = [np.mean(data_cac[value]) for value in values]
        ax.plot(positions - width, mean_cscs, "k^-", markersize=10, linewidth=2.5, label="CSCS")
        ax.plot(positions + width, mean_cac, "ko-", markersize=10, linewidth=2.5, label=f"CAC (t={t0})")

        # Axes settings
        ax.set_xticks(positions)
        if settings["erase-ticks"]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xticklabels(labels)
            ax.legend()
            ax.set_ylabel("Mean Rank")
        ax.set_ylim(-padding * 19, (seq_len + padding) * 19)  # Maximum rank is `seq_len * 19`
        ax.set_title(f"Box-whisker plot of {escape} for CSCS and CAC (t={t0})")
        plt.tight_layout()

        # Save the figure
        filename = f"boxplot-(e={escape_prefix})-(t={t0}).pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    _ns_pt2_box_whisker()

    ### PART 3: Export AUC bar chart ###

    def _ns_pt3_auc_bar_chart():
        """
        Export a bar chart comparing the AUC (Area Under the Curve) for CSCS and CAC.
        This function calculates the AUC for each dataset in the `values` list.
        It can be shown that AUC in this case is equal to `1 - (mean rank / total size)`.
        The bar chart shows the AUC for CSCS and CAC side by side for
        each dataset.
        The output is saved in a PDF file with the prefix `auc-`.
        """
        # Prepare the AUC data
        auc_cscs = [1 - np.mean(data_cscs[value]) / (seq_len * 19) for value in values]
        auc_cac = [1 - np.mean(data_cac[value]) / (seq_len * 19) for value in values]

        # Draw the AUC bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = np.arange(num_entries)
        colors = ["lightblue", "lightgreen"]
        width = 0.15
        for i, value in enumerate(values):
            ax.bar(
                [positions[i] - width, positions[i] + width],
                [auc_cscs[i], auc_cac[i]],
                width=1.8 * width,
                color=colors,
            )

        # Axes settings
        ax.set_xticks(positions)
        if settings["erase-ticks"]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xticklabels(labels)
            ax.set_ylabel("AUC")
            # Create custom legend handles
            handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, ["CSCS", f"CAC (t={t0})"])]
            ax.legend(handles=handles, title="Scoring Method")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Bar chart of AUC for {escape} (t={t0})")
        plt.tight_layout()

        # Save the figure
        filename = f"auc-(e={escape_prefix})-(t={t0}).pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    _ns_pt3_auc_bar_chart()

    ### PART 4: Export the mean rank as a function of the evolutionary time parameter (T) ###

    def _ns_pt4_mean_rank_list_plot():
        """
        Export the mean rank as a function of the evolutionary time parameter (T).
        This function generates a line plot showing the mean rank for each dataset
        as a function of the evolutionary time parameter (T).
        The plot includes multiple lines, each representing a different PLM backbone.
        The x-axis represents the evolutionary time parameter (T), and the y-axis
        represents the mean rank.
        The output is saved in a PDF file with the prefix `listplot-`.
        """
        # Prepare data for list plot
        data_csc_ranks = {value: [np.mean(db_filtered[value][f"t={t}"]) for t in times] for value in values}

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, value in enumerate(values):
            ax.plot(
                times,
                data_csc_ranks[value],
                label=labels[i],
                marker="o",
                linewidth=2.5,
                linestyle="--",
                markersize=10,
            )

        # Axes settings
        ax.set_xticks(times)
        if settings["erase-ticks"]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Mean Rank")
            ax.set_xlabel("Evolutionary Parameter (T)")
            ax.legend()
        ax.set_ylim(0, 13000)
        ax.set_title(f"Mean rank of {escape} for CAC")
        plt.tight_layout()
        plt.xscale("log")

        filename = f"listplot-(e={escape_prefix}).pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    _ns_pt4_mean_rank_list_plot()

    ### PART 5: Export the bar charts of the mean rank as a function of model size ###

    def _ns_pt5_mean_rank_bar_chart():
        """
        Export a grouped bar chart of the mean rank as a function of model size.
        This function generates a bar chart comparing the mean rank for each model size
        at different evolutionary time parameters (T).
        The x-axis represents the evolutionary time parameter (T), and the y-axis
        represents the mean rank.
        The chart includes multiple bars for each evolutionary time parameter (T),
        each representing a different model size.
        The output is saved in a PDF file with the prefix `barchart-esm-`.
        """
        values_models = [
            "SARS-CoV-2-WildType-Random",
            "SARS-CoV-2-WildType-ESM-150M-Cos",
            "SARS-CoV-2-WildType-ESM-650M-Cos",
            "SARS-CoV-2-WildType-ESM-3B-Cos",
        ]
        for value in values_models:
            if value not in db_filtered:
                raise ValueError(f"Dataset `{value}` for the escape `{escape}` not found.")

        # Prepare data for bar chart
        num_entries = len(values_models)
        data_cac = {f"t={t}": [np.mean(db_filtered[value][f"t={t}"]) for value in values_models] for t in times}

        # Draw the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels_models = list(v2l_map[value] for value in values_models)
        colors = ["grey", "r", "g", "b"]
        width = 0.12
        for i, t in enumerate(times):
            means = np.array(data_cac[f"t={t}"])
            x = 1.25 * width * np.arange(num_entries)
            x = i + x - np.mean(x)
            ax.bar(x, means, width, color=colors, alpha=0.7)

        # Set x-ticks and labels
        ax.set_xticks(np.arange(len(times)))
        if settings["erase-ticks"]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xticklabels(times)
            ax.set_xlabel("Evolutionary Parameter (T)")
            ax.set_ylabel("Mean Rank")
            # Create custom legend handles
            handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels_models)]
            ax.legend(handles=handles, title="Model")
        ax.set_ylim(0, 12000)  # Hardcoded y-limit for better visibility
        ax.set_title(f"Grouped Bar Chart of {escape} for CAC")
        plt.tight_layout()

        # Save the figure
        filename = f"barchart-esm-(e={escape_prefix}).pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()

    _ns_pt5_mean_rank_bar_chart()

    ### PART 6: Export the Wilcoxon signed-rank test results ###
    def _ns_pt6_wilcoxon_test():
        """
        Export the Wilcoxon signed-rank test results for CSCS and CAC.
        This function performs the Wilcoxon signed-rank test for each dataset in the `values` list.
        The results are saved in a text file with the prefix `wilcoxon-`.
        """
        filename = f"wilcoxon-(e={escape_prefix})-(t={t0}).txt"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w") as f:
            f.write(f"Wilcoxon test results for escape: {escape}\n\n")

            for value in values:
                rank_cscs = np.array(db_filtered[value]["cscs"])
                rank_cac = np.array(db_filtered[value][f"t={t0}"])

                statistic, p_value = wilcoxon(rank_cac, rank_cscs, alternative="less")

                # 4. 결과 출력
                f.write(f"Result: {value}, T: {t0}\n")
                f.write(f"Wilcoxon statistics, one-sided (W): {statistic}\n")
                f.write(f"p-value: {p_value}\n\n")

            f.write("=" * 50 + "\n\n")

    _ns_pt6_wilcoxon_test()


if __name__ == "__main__":
    print("Starting the export...")
    for escape in tqdm(settings["validation-sets"]):
        tqdm.write(f"Processing escape: {escape}")
        export_plots(escape)
