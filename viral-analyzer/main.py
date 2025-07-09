"""
A collection of utility functions and unit tests for the analysis of the viral escape data.
"""

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Any

import plotly.graph_objects as go
import modules.analyzer as an
from modules.plotly_custom import create_ternary_contour
from modules.ternary_custom import draw_ternary_plot


STRAIN_DATA: dict[str, Any] = an.STRAIN_DATA
ESCAPE_DATA: dict[str, Any] = an.ESCAPE_DATA
RESULT_DATA: dict[str, Any] = an.RESULT_DATA


def random_df(strain: str, seed: int | None = None):
    """
    Creates a fake DataFrame for the specified strain with random semantic and grammar scores.
    This is used for providing negative controls for the analysis.

    Parameters
    ----------
    strain : str
        The strain type to create the fake DataFrame for. Should be one of the keys in STRAIN_DATA.
    """
    data = STRAIN_DATA.get(strain)
    assert data is not None, f"Invalid strain type: {strain}. Available types: {', '.join(STRAIN_DATA.keys())}."

    assert "sequence-path" in data, f"Data path not found for strain: {strain}."
    with open(data["sequence-path"], "r") as f:
        df = pd.read_csv(f, sep=",", header=0)

    # Build a fake DataFrame with random semantic and grammar scores
    rng = np.random.default_rng(seed)
    aa_map = {aa: i for i, aa in enumerate(sorted(an.STANDARD_AAS))}
    rows = []
    for _, row in df.iterrows():
        pos = row["pos"]
        original_aa = row["aa"]

        if original_aa not in aa_map:
            continue

        new_semantic = rng.standard_normal(size=len(aa_map))
        new_grammar = np.exp(rng.standard_normal(size=len(aa_map)))
        new_grammar[aa_map[original_aa]] = 0.0
        new_grammar = new_grammar / np.sum(new_grammar)

        for mutated_aa, i in aa_map.items():
            if mutated_aa == original_aa:
                continue
            rows.append(
                {
                    "pos": pos,
                    "wt": original_aa,
                    "mut": mutated_aa,
                    "prob": new_grammar[i],
                    "change": new_semantic[i],
                }
            )

    return pd.DataFrame(rows, columns=["pos", "wt", "mut", "prob", "change"])


class ScoreCalculationContext:
    """
    A context for the score calculation, containing the parameters and results of the analysis.
    """

    escape: str
    """The name of the escape mutation entry to use."""
    result: str
    """The name of the model inference result entry to use."""
    t: float
    """The evolutionary time parameter to use for the analysis."""
    virus: str
    """The name of the virus to use for the analysis."""
    strain: str
    """The name of the strain to use for the analysis."""
    bary_min: np.ndarray
    """The barycentric coordinates of the minimum CAC score."""
    fval_min: float
    """The minimum CAC score."""
    df: pd.DataFrame
    """The DataFrame with the model inference results."""

    def __init__(self, escape: str, result: str, t: float):
        self.escape = escape
        self.result = result
        self.t = t


def calc_iter(
    batch: Iterable[tuple[str, str, float]],
    **kwargs: dict[str, Any],
):
    """
    Iterate over a batch of triple `(escape, result, time)` and yield the context for each computation.

    Parameters
    ----------
    batch : Iterable[tuple[str, str, float]]
        An iterable of tuples containing the escape mutation type, result type, and evolutionary time parameter.
    """
    # Parse the options for the computation
    seed = kwargs.get("seed", None)  # Seed for random number generation

    # Sort the batch by the `result` parameter for efficient reuse of the DataFrame
    batch = sorted(batch, key=lambda x: x[1])

    # Store the last used data for reuse
    last_strain: str | None = None
    last_df: pd.DataFrame | None = None
    last_Q: np.ndarray | None = None

    # Iterate over the batch
    for escape, result, t in tqdm(batch, desc="Computing CSCS scores"):
        try:
            # Load the escape mutation data
            if escape not in ESCAPE_DATA:
                raise ValueError(f"Invalid escape type: {escape}. Available types: {', '.join(ESCAPE_DATA.keys())}.")
            escape_data = ESCAPE_DATA[escape]

            # Load the model inference result data
            if result not in RESULT_DATA:
                raise ValueError(f"Invalid result type: {result}. Available types: {', '.join(RESULT_DATA.keys())}.")
            result_data = RESULT_DATA[result]

            # Load the virus data
            if "virus" not in result_data or "strain" not in result_data:
                raise ValueError("The result data must contain the 'virus' and 'strain' keys.")
            virus = result_data["virus"]
            strain = result_data["strain"]

            # Validate the strain type
            if strain not in STRAIN_DATA:
                raise ValueError(f"Invalid strain type: {strain}. Available types: {', '.join(STRAIN_DATA.keys())}.")
            is_random = result_data["path"].lower().strip() == "random"
            can_reuse_strain = last_strain == strain

            # Compute the transition matrix
            if can_reuse_strain and (last_Q is not None):
                Q = last_Q
            else:
                Q = an.get_Q(strain)
                last_Q = Q

            # Load the data for the specified model inference result
            if can_reuse_strain and (last_df is not None):
                df = last_df
            else:
                if is_random:
                    df = random_df(strain, seed)
                else:
                    df = pd.read_csv(result_data["path"], sep="\t", header=0)

                # Initialize the DataFrame with the necessary columns
                df = an.rename_df(df, columns=result_data.get("columns", None))
                df = an.prepare_df(df, strain)

                # Add the escape mutations to the DataFrame
                mutation_code = df["original_aa"] + df["pos"].astype(str) + df["mutated_aa"]
                df["is_escape"] = mutation_code.isin(escape_data["mutations"])

                # Compute the original CSCS score (as in Hie. et. al.)
                df["rank_cscs"] = (
                    (
                        df["semantic"].rank(ascending=False, method="min").astype(int)
                        + df["grammar"].rank(ascending=False, method="min").astype(int)
                    )
                    .rank(ascending=True, method="min")
                    .astype(int)
                )

                # Store the last DataFrame for reuse
                last_df = df

            # Update the last strain
            last_strain = strain

            # Create a copy of the DataFrame for callback
            df_export = df.copy()

            # Compute the CLIMB regularization term
            df_export["log_clib_z"] = an.get_log_clib_z(df, Q, t)

            # Create a context for the score calculation
            context = ScoreCalculationContext(escape, result, t)
            context.virus = virus
            context.strain = strain
            context.df = df_export

            # Yield the context for further processing
            yield context
        except Exception as e:
            # Handle exceptions gracefully and continue with the next batch item
            print(f"Error processing escape '{escape}', result '{result}', time '{t}': {e}")
            continue


def callback_calc_cac(
    context: ScoreCalculationContext,
    **kwargs: dict[str, Any],
):
    """
    Callback function for optimizing the barycentric coordinates
    and computing the CAC scores.

    Parameters
    ----------
    context : ScoreCalculationContext
        The context containing the parameters and results of the analysis.
    tolerance : float, optional
        The tolerance for the minimization of the barycentric coordinates. Default is `0.1`.
    max_recursion : int, optional
        The maximum number of recursion steps for the minimization of the barycentric coordinates. Default is `256`.
    """
    # Parse the options for the computation
    options = {
        "tolerance": 0.1,
        "max_recursion": 256,
    } | kwargs
    tolerance: float = options.get("tolerance", 0.1)
    max_recursion: int = options.get("max_recursion", 256)

    # Read the context parameters
    df = context.df
    log_clib_z = df["log_clib_z"].to_numpy()

    # Internal function to compute the score based on barycentric coordinates
    escape_filter = df["is_escape"] == True

    def _get_score(bary_coord: np.ndarray) -> float:
        cacscs = an.get_cacscs(df, log_clib_z, bary_coord)
        mean_rank = cacscs[escape_filter].mean()
        return mean_rank

    # Find the minimum barycentric coordinates and the corresponding score
    bary_min, fval_min = an.minimize_bary_f(
        _get_score,
        tolerance=tolerance,
        max_recursion=max_recursion,
    )

    # Store the results in the DataFrame
    df["cacscs"] = an.get_cacscs(df, log_clib_z, bary_min)

    return context, bary_min, fval_min


def get_scores(
    escape: str,
    result: str,
    times: list[float] | None = None,
    **kwargs: dict[str, Any],
):
    """
    Computes the scores for the specified escape and result types over a range of evolutionary times,
    and saves the results to CSV files.
    """
    # Preamble:
    times = times or [0.01, 0.033, 0.1, 0.33, 1.0, 3.3, 10.0]

    escape_data = ESCAPE_DATA[escape]
    result_data = RESULT_DATA[result]
    virus = result_data["virus"]
    strain = result_data["strain"]
    virus_data = STRAIN_DATA[strain]

    output_root = Path(__file__).parent / "outputs" / virus_data["prefix"] / result_data["prefix"]
    output_prefix = escape_data["prefix"]
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Main logic:
    dists: dict[str, list[float]] = {}
    """A dictionary to store the ranks of escape mutations for each evolutionary time parameter."""
    cscs_exported = False
    for context in calc_iter([(escape, result, t) for t in times], **kwargs):
        # Read relevant values from the context
        context, bary_min, fval_min = callback_calc_cac(context, **kwargs)
        t = context.t
        df = context.df
        escape_filter = df["is_escape"] == True

        # Store the CSCS escape ranks to the dictionary only once (this does not depend on t)
        if not cscs_exported:
            cscs_exported = True
            cscs_escape = df["rank_cscs"][escape_filter]
            dists["cscs"] = cscs_escape.to_list()

        # Store the CAC escape ranks to the dictionary
        cacscs_escape = df["cacscs"][escape_filter]
        dists[f"t={t}"] = cacscs_escape.to_list()

        # Export the DataFrame to a CSV file
        output_path = output_root / f"{output_prefix}-(t-{t:.2f}).csv"
        df.to_csv(output_path, sep=",", index=False)

    def _report_mean_std(arr: list[float]):
        """
        Helper function to compute the mean and standard deviation of the ranks.
        """
        np_arr = np.array(arr)
        return f"Mean Rank: {np_arr.mean():.2f} ± {np_arr.std():.2f}"

    # Export a report containing the statistics of the escape ranks
    with open(output_root / f"{output_prefix}-stat.txt", "w") as f:
        lines = []
        lines.append(f"Virus: {virus}")
        lines.append(f"Strain: {strain}")
        lines.append(f"Escape Mutants: {escape}")
        lines.append(f"Result File: {result}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Standard CSCS")
        lines.append(_report_mean_std(dists["cscs"]))
        lines.append("")
        lines.append("-" * 80)
        for t in times:
            lines.append("")
            lines.append(f"T = {t}")
            lines.append(_report_mean_std(dists[f"t={t}"]))
        f.write("\n".join(lines))

    # Export the metadata to a JSON file
    with open(output_root / f"{output_prefix}-metadata.json", "w") as f:
        metadata = {
            "escape": escape,
            "result": result,
            "virus": virus,
            "strain": strain,
            "times": times,
            "distributions": dists,
        }
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def get_ablation_control(
    escape: str,
    result: str,
    repeat: int = 100,
    times: list[float] | None = None,
    seed_after: str = "increment",
    **kwargs: dict[str, Any],
):
    """
    Perform ablation control experiment for the specified escape and result types over a range of evolutionary times,
    and saves the results to CSV files.

    Here, ablation control experiment means computing the scores for the randomized semantic and grammar scores
    `repeat` number of times for the specified escape and result types. (`ablation` refers to the fact that
    model inference is not used.)
    """
    assert seed_after in ["increment", "randomize", "fixed"], f"Invalid seed_after value."
    seed = kwargs.get("seed", None)

    # Preamble: Sanitize the inputs and prepare the output directory
    times = times or [0.01, 0.033, 0.1, 0.33, 1.0, 3.3, 10.0]

    escape_data = ESCAPE_DATA[escape]
    result_data = RESULT_DATA[result]
    if result_data["path"].lower().strip() != "random":
        raise ValueError(f"The result data path for '{result}' must be specified as `random`.")

    virus = result_data["virus"]
    strain = result_data["strain"]
    virus_data = STRAIN_DATA[strain]

    output_root = Path(__file__).parent / "outputs-ablation" / virus_data["prefix"] / result_data["prefix"]
    output_prefix = escape_data["prefix"]
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Main logic:
    dists: dict[str, list[list[float]]] = {}
    for epoch in tqdm(range(repeat), desc="Computing Ablation Control Scores"):
        cscs_exported = False
        for context in calc_iter([(escape, result, t) for t in times], **(kwargs | dict(seed=seed))):
            context, bary_min, fval_min = callback_calc_cac(context, **kwargs)
            t = context.t
            df = context.df
            escape_filter = df["is_escape"] == True

            # Write the headers to the output stream
            if not cscs_exported:
                cscs_exported = True
                cscs_escape = df["rank_cscs"][escape_filter]
                dists.setdefault("cscs", []).append(cscs_escape.to_list())

            # Compute the mean rank for escape mutations
            cacscs_escape = df["cacscs"][escape_filter]
            dists.setdefault(f"t={t}", []).append(cacscs_escape.to_list())
        # Update the seed based on the specified seed_after option
        if seed is not None:
            if seed_after == "increment":
                seed += 1
            elif seed_after == "randomize":
                seed = np.random.default_rng(seed).integers(0, 2**32 - 1)

    def _report_mean_std(matrix: list[list[float]]):
        """
        Helper function to compute the mean and standard deviation of the list of ranks.
        Each entry of the list is a list of ranks for a single trial.
        """
        np_matrix = np.array(matrix)
        means = np_matrix.mean(axis=1)
        stds = np_matrix.std(axis=1)
        return [
            f"Mean Rank: {means.mean():.2f} ± {means.std():.2f}",
            f"Std Rank: {stds.mean():.2f} ± {stds.std():.2f}",
        ]

    # Export the statistics of the escape ranks
    with open(output_root / f"{output_prefix}-stat.txt", "w") as f:
        lines = []
        lines.append(f"Virus: {virus}")
        lines.append(f"Strain: {strain}")
        lines.append(f"Escape Mutants: {escape}")
        lines.append(f"Result File: {result}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Standard CSCS")
        lines.extend(_report_mean_std(dists["cscs"]))
        lines.append("")
        lines.append("-" * 80)
        for t in times:
            lines.append("")
            lines.append(f"T = {t}")
            lines.extend(_report_mean_std(dists[f"t={t}"]))
        f.write("\n".join(lines))

    # Export the metadata to a JSON file
    with open(output_root / f"{output_prefix}-metadata.json", "w") as f:
        metadata = {
            "escape": escape,
            "result": result,
            "virus": virus,
            "strain": strain,
            "times": times,
            "repeat": repeat,
            "distributions": dists,
        }
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def draw_ternary(
    escape: str,
    result: str,
    res: int = 30,
    times: list[float] | None = None,
    **kwargs: dict[str, Any],
) -> None:
    # Preamble: Sanitize the inputs and prepare the output directory
    options = {
        "axes": "psg",
        "pole_labels": {"p": "CLIMB", "s": "Semantic", "g": "Grammar"},
        "ncontours": 20,
        "colorscale": "Viridis",
        "showscale": True,
    } | kwargs

    options["axes"] = options.get("axes", "psg").lower().strip()
    if not sorted(options.get("axes", "psg")) == ["g", "p", "s"]:
        raise ValueError("Axes specification must be a permutation of 'p', 's', 'g'.")

    times = times or [0.01, 0.033, 0.1, 0.33, 1.0, 3.3, 10.0]

    escape_data = ESCAPE_DATA[escape]
    result_data = RESULT_DATA[result]
    virus = result_data["virus"]
    strain = result_data["strain"]
    virus_data = STRAIN_DATA[strain]

    output_root = Path(__file__).parent / "outputs-figures" / virus_data["prefix"] / result_data["prefix"]
    output_prefix = escape_data["prefix"]
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Construct a triangular grid of barycentric coordinates
    coords = np.array([np.array([i, j, res - i - j]) / res for i in range(res + 1) for j in range(res + 1 - i)])

    # Barycentric weights for the ternary plot
    x_s = coords[:, 0]  # Semantic change
    x_g = coords[:, 1]  # Log-Grammaticality
    x_p = coords[:, 2]  # CLIMB Regularizer

    # Compute the CAC for each barycentric coordinate in the triangular grid
    f_vals_list: list[list[float]] = []
    for context in calc_iter([(escape, result, t) for t in times], **kwargs):
        t = context.t
        df = context.df
        escape_filter = df["is_escape"] == True
        log_clib_z = df["log_clib_z"].to_numpy()

        def _get_score(bary_coord: np.ndarray) -> float:
            cacscs = an.get_cacscs(df, log_clib_z, bary_coord)
            mean_rank = cacscs[escape_filter].mean()
            return mean_rank

        f_vals_list.append([_get_score(bary) for bary in coords])

    # Prepare the ternary plot
    f_vals_merged = np.concatenate(f_vals_list)
    v_min = np.min(f_vals_merged)
    v_max = np.max(f_vals_merged)

    # Permute the axes based on the options provided
    axes_x_map = {"p": x_p, "s": x_s, "g": x_g}
    x_sorted = np.array([axes_x_map[axis] for axis in options["axes"]])
    label_sorted = [options["pole_labels"][axis] for axis in options["axes"]]

    # Export the ternary plots for each evolutionary time parameter
    for t, f_vals in tqdm(zip(times, f_vals_list), desc="Drawing Ternary Plots", total=len(times)):
        fig = create_ternary_contour(
            x_sorted,
            f_vals,
            pole_labels=label_sorted,
            ncontours=options["ncontours"],
            interp_mode="cartesian",
            colorscale=options["colorscale"],
            showscale=options["showscale"],
            v_min=v_min,
            v_max=v_max,
            # title={
            #     "text": s_title,
            #     "x": 0.5,
            #     "xanchor": "center",
            #     "yanchor": "top",
            # },
        )

        output_path = output_root / f"{output_prefix}-(t-{t:.2f}).pdf"
        fig.write_image(output_path, width=800, height=600)

    # Export the metadata to a JSON file
    with open(output_root / f"{output_prefix}-metadata.json", "w") as f:
        metadata = {
            "escape": escape,
            "result": result,
            "virus": virus,
            "strain": strain,
            "times": times,
            "res": res,
            "color-range": [v_min, v_max],
        }
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def draw_minimizer(
    escape: str,
    result: str,
    times: np.ndarray,
    **kwargs: dict[str, Any],
):
    # Preamble: Sanitize the inputs and prepare the output directory
    options = {
        "axes": "psg",
        "pole_labels": {"p": "CLIMB", "s": "Semantic", "g": "Grammar"},
        "colorscale": "rainbow",
        "showscale": True,
        "size": 50,
    } | kwargs

    options["axes"] = options.get("axes", "psg").lower().strip()
    if not sorted(options.get("axes", "psg")) == ["g", "p", "s"]:
        raise ValueError("Axes specification must be a permutation of 'p', 's', 'g'.")

    escape_data = ESCAPE_DATA[escape]
    result_data = RESULT_DATA[result]
    virus = result_data["virus"]
    strain = result_data["strain"]
    virus_data = STRAIN_DATA[strain]

    output_root = Path(__file__).parent / "outputs-figures" / virus_data["prefix"] / result_data["prefix"]
    output_prefix = escape_data["prefix"]
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    # Main logic:
    # Export the minimizers for the specified evolutionary times
    results = []
    output_path_csv = output_root / f"{output_prefix}-min.csv"
    if output_path_csv.exists():
        np_results = np.loadtxt(output_path_csv, delimiter=",", skiprows=1)
    else:
        for context in calc_iter([(escape, result, t) for t in times], **kwargs):
            # Read relevant values from the context
            context, bary_min, fval_min = callback_calc_cac(context, **kwargs)
            results.append([context.t, *bary_min, fval_min])

        np_results = np.array(results)
        np.savetxt(
            output_path_csv,
            np_results,
            delimiter=",",
            header="time,semantic,grammar,climb,minimum",
        )

    # Permute the axes based on the options provided
    x_s = np_results[:, 1]  # Semantic change
    x_g = np_results[:, 2]  # Log-Grammaticality
    x_p = np_results[:, 3]  # CLIMB Regularizer
    axes_x_map = {"p": x_p, "s": x_s, "g": x_g}
    x_sorted = np.array([axes_x_map[axis] for axis in options["axes"]])
    label_sorted = [options["pole_labels"][axis] for axis in options["axes"]]

    # # Export the minimizers
    # fig = go.Figure(
    #     data=go.Scatterternary(
    #         a=x_sorted[0],
    #         b=x_sorted[1],
    #         c=x_sorted[2],
    #         mode="markers",
    #         marker=dict(
    #             size=options["size"], color=times, colorscale=options["colorscale"], showscale=options["showscale"]
    #         ),
    #         # text=[f"A: {a_val:.2f}, B: {b_val:.2f}, C: {c_val:.2f}" for a_val, b_val, c_val in data[:, [2, 0, 1]]],
    #         # hoverinfo="text",
    #     )
    # )

    # # Customize the layout
    # fig.update_layout(
    #     title={
    #         "text": f"Mean Rank Minimizers for {strain}<br>Escape: {escape}, Result: {result}",
    #         "x": 0.5,
    #         "xanchor": "center",
    #         "yanchor": "top",
    #     },
    #     ternary=dict(
    #         aaxis=dict(title=label_sorted[0], showticklabels=False, showline=False),
    #         baxis=dict(title=label_sorted[1], showticklabels=False, showline=False),
    #         caxis=dict(title=label_sorted[2], showticklabels=False, showline=False),
    #     ),
    #     showlegend=False,
    # )

    fig, ax = draw_ternary_plot(
        x_sorted.T,
        times,
        axes_labels=label_sorted,
        grid_steps=5,
        cmap=options["colorscale"],
        marker_opts={
            "s": options["size"],
            "linewidth": 0,
            "alpha": 1,
        },
    )

    output_path_fig = output_root / f"{output_prefix}-min.pdf"
    # fig.write_image(output_path_fig, width=800, height=600)
    fig.savefig(output_path_fig, bbox_inches="tight", dpi=300)

    # Export the metadata to a JSON file
    with open(output_root / f"{output_prefix}-min-metadata.json", "w") as f:
        metadata = {
            "escape": escape,
            "result": result,
            "virus": virus,
            "strain": strain,
            "times": list(times),
        }
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def filter_wildcard(key: str, data: dict[str, Any]) -> list[str]:
    """
    Filters the keys in the data dictionary based on a wildcard pattern.

    Parameters
    ----------
    key : str
        The wildcard pattern to filter the keys.
    data : dict[str, Any]
        The dictionary to filter the keys from.

    Returns
    -------
    list[str]
        A list of keys that match the wildcard pattern.
    """
    if key.endswith("*"):
        return [k for k in data.keys() if k.startswith(key[:-1])]
    return [key] if key in data else []


def iter_escape_result(escape: str, result: str):
    """
    Iterate over the escape and result types, yielding pairs of (escape, result) tuples.

    Parameters
    ----------
    escape : str
        The escape mutation type to iterate over. Support very basic wildcard matching.
    result : str
        The result type to iterate over. Support very basic wildcard matching.
    """
    escape_batch = filter_wildcard(escape, ESCAPE_DATA)
    result_batch = filter_wildcard(result, RESULT_DATA)
    for r in result_batch:
        for e in escape_batch:
            yield e, r


def main():
    # Define the command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze the virus mutation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--escape",
        type=str,
        default="SARS-CoV-2-DMS",
        help=f"Specify the escape type to analyze. Adding `*` to the argument matches any types that starts with it. ({''.join(ESCAPE_DATA.keys())})",
    )
    parser.add_argument(
        "--result",
        type=str,
        default="SARS-CoV-2-Hie",
        help=f"Specify the result type to analyze. Adding `*` to the argument matches any types that starts with it. ({''.join(RESULT_DATA.keys())})",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Set to perform ablation control experiment. (Use --times to specify the time schedule and --repeat to specify the number of repeats.)",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Set to calculate the scores for the mutations. (Use --times to specify the time schedule.)",
    )
    parser.add_argument(
        "--ternary",
        action="store_true",
        help="Set to export the ternary plots of the mean CSCS for the specified evolutionary time parameters and resolutions. (Use --times to specify the time schedule.)",
    )
    parser.add_argument(
        "--minimizer",
        nargs="*",
        default=None,
        help="Set to export the plot of minimizing combination ratios. Accepts 3 optional values of the form `t0, t1, steps`.",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=float,
        help="Specify the evolutionary time parameters to export the ternary plots, separated by spaces.",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=30,
        help="Specify the resolution of the ternary plot.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Specify the number of repeats for the ablation control experiment. Default is `100`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specify the random seed for the random number generation. Default is `None`.",
    )
    args = parser.parse_args()

    # Export the scores for the specified escape and result types
    if args.scores:
        for escape, result in iter_escape_result(args.escape, args.result):
            print(f"Calculating scores for result '{result}' and escape '{escape}'...")
            get_scores(escape, result, times=args.times, seed=args.seed)

    # Export the statistical analysis of the ablation control experiment
    if args.ablation:
        for escape, result in iter_escape_result(args.escape, args.result):
            print(f"Performing ablation control for result '{result}' and escape '{escape}'...")
            get_ablation_control(escape, result, repeat=args.repeat, times=args.times, seed=args.seed)

    if args.ternary:
        for escape, result in iter_escape_result(args.escape, args.result):
            print(f"Drawing ternary plots for result '{result}' and escape '{escape}'...")
            draw_ternary(escape, result, res=args.res, times=args.times, seed=args.seed)

    if args.minimizer is not None:
        # Sanitize the input parameters
        t0 = float(args.minimizer[0]) if len(args.minimizer) > 0 else 0.001
        t1 = float(args.minimizer[1]) if len(args.minimizer) > 1 else 7
        steps = int(args.minimizer[2]) if len(args.minimizer) > 2 else 1000
        assert t0 > 0, f"Invalid t0 parameter: {t0}"
        assert t1 > t0, f"Invalid t1 parameter: {t1}"
        assert steps > 0, f"Invalid steps parameter: {steps}"
        times = np.linspace(t0, t1, steps)
        print(f"Plotting minimizers for {steps} points in the time interval [{t0}, {t1}]...")

        for escape, result in iter_escape_result(args.escape, args.result):
            print(f"Drawing minimizer plot for result '{result}' and escape '{escape}'...")
            draw_minimizer(escape, result, times=times, seed=args.seed)


if __name__ == "__main__":
    main()
