"""
A collection of functions to compute the CLIB scores and the CaCSCS score
for a given virus.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy.linalg import expm
from itertools import product
from typing import Callable, Any
from numpy.typing import ArrayLike
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sbs import SBS, SBSType


################################################################################
# Constants
################################################################################

BASE = ["A", "C", "G", "T"]
"""
Everything in this code is based on this order of bases.
For example, the codon `AAG` is encoded as `(0, 0, 2)`.
"""

STANDARD_AAS = set(list("ACDEFGHIKLMNPQRSTVWY"))

PATH_CODON_INDICES = "data/codon_indices.json"
PATH_STRAIN_DATA = "metadata/virus_strains"
PATH_ESCAPE_DATA = "metadata/escape_mutants"
PATH_RESULT_DATA = "metadata/model_results"


################################################################################
# Load hard-coded data
################################################################################

# Load the codon indices from the JSON file
LETTER_TO_CODONS: dict[str, np.ndarray] = {}
"""
The letter-to-codons table.

* `key` (*str*): the amino acid alphabet.
* `value` (*np.ndarray*): a list of codons, where each codon is encoded as a triplet of indices
            and the incoding is based on the `BASE` variable.
"""

if not os.path.exists(PATH_CODON_INDICES):
    raise FileNotFoundError(f"Codon indices file not found: {PATH_CODON_INDICES}")

with open(PATH_CODON_INDICES, "r") as f:
    letter_to_codons = json.load(f)
    for k, v in letter_to_codons.items():
        LETTER_TO_CODONS[k] = np.array(v)


# Load batch JSON data for virus strains
def load_batch_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    result = {}
    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r") as f:
                cur_result = json.load(f)
                if isinstance(cur_result, dict):
                    result = result | cur_result
    return result


STRAIN_DATA = load_batch_json(PATH_STRAIN_DATA)
"""A dictionary containing the data for each virus strain."""

ESCAPE_DATA = load_batch_json(PATH_ESCAPE_DATA)
"""A dictionary containing the escape mutations for each virus strain."""

RESULT_DATA = load_batch_json(PATH_RESULT_DATA)
"""A dictionary containing the results of the analysis for each virus strain."""


################################################################################
# DataFrame Utilities
################################################################################


def standardize_score(score: pd.Series) -> pd.Series:
    score = (score - score.mean()) / (score.std() + 1e-15)  # Avoid division by zero
    return score


class OutputDataColumnSpecs:
    def __init__(
        self,
        pos: str = "pos",
        original_aa: str = "wt",
        mutated_aa: str = "mut",
        grammar: str = "prob",
        semantic: str = "change",
    ):
        self.pos = pos
        self.original_aa = original_aa
        self.mutated_aa = mutated_aa
        self.grammar = grammar
        self.semantic = semantic

    def as_dict(self) -> dict[str, str]:
        return {
            self.pos: "pos",
            self.original_aa: "original_aa",
            self.mutated_aa: "mutated_aa",
            self.grammar: "grammar",
            self.semantic: "semantic",
        }

    @classmethod
    def to_spec(cls, columns: "OutputDataColumnSpecs | dict[str, str] | list[str] | None" = None):
        if columns is None:
            return OutputDataColumnSpecs()
        elif isinstance(columns, list):
            return OutputDataColumnSpecs(
                pos=columns[0],
                original_aa=columns[1],
                mutated_aa=columns[2],
                grammar=columns[3],
                semantic=columns[4],
            )
        elif isinstance(columns, dict):
            return OutputDataColumnSpecs(**columns)
        elif isinstance(columns, OutputDataColumnSpecs):
            return columns
        else:
            raise ValueError(
                f"Invalid columns specification: {columns}. Expected OutputDataColumnSpecs, dict, or list."
            )


def rename_df(
    df: pd.DataFrame,
    columns: OutputDataColumnSpecs | dict[str, str] | list[str] | None = None,
) -> pd.DataFrame:
    """
    Sanitize the DataFrame by renaming the columns according to the provided specifications.
    If no specifications are provided, it uses the default column names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sanitize.
    columns : OutputDataColumnSpecs | dict[str, str] | None
        The column specifications to use for renaming.

    Returns
    -------
    pd.DataFrame
        The sanitized DataFrame with renamed columns. The resulting DataFrame will contain columns:
        * `pos`: The position of the mutation.
        * `original_aa`: The original amino acid at the position.
        * `mutated_aa`: The mutated amino acid at the position.
        * `grammar`: The grammaticality score of the mutation.
        * `semantic`: The semantic change score of the mutation.
        * `is_viable`: Whether the mutation is viable.
        * `is_escape`: Whether the mutation is an escape mutation.
    """
    column_obj = OutputDataColumnSpecs.to_spec(columns)
    column_map = column_obj.as_dict()
    column_set = set(column_map.keys())
    assert column_set.issubset(df.columns), f"Missing columns: {column_set - set(df.columns)}"
    return df.rename(columns=column_map)


def prepare_df(df: pd.DataFrame, strain_name: str) -> pd.DataFrame:
    """
    Add a column with the codon for each position.
    The DataFrame is expected to have columns 'pos', 'codon', 'aa'.

    This adds the following columns to the DataFrame:
    * `codon`: The codon for the wild type amino acid at the position.
    * `mutation_id`: A unique identifier for the mutation in the format "codon->mutated_aa".
    * `semantic_z`: The standardized semantic change score.
    * `log_grammar`: The log of the grammaticality score.
    * `log_grammar_z`: The standardized log grammaticality score.

    This also validates that the amino acid sequence in the DataFrame matches
    the one in the codon table for the given positions.

    Args
    ----
    df : pd.DataFrame
        The DataFrame containing the inference results, with columns 'pos', 'original_aa', 'mutated_aa', 'grammar', 'semantic'.
    strain_name : str
        The name of the virus strain for which to prepare the DataFrame.
    """
    # Load the wild type codon sequence for the virus
    assert strain_name in STRAIN_DATA, f"Virus {strain_name} not found in the virus data."
    data = STRAIN_DATA[strain_name]
    assert "sequence-path" in data, f"Wild type codons not found for virus {strain_name}."
    wildtype_seq_path = data["sequence-path"]

    with open(wildtype_seq_path, "r") as f:
        df_seq_codon = pd.read_csv(f, sep=",", header=0)

    column_required = {"pos", "codon", "aa"}
    assert column_required.issubset(df_seq_codon.columns), "The DataFrame must have columns 'pos', 'codon', 'aa'."

    # Parse the position-codon mapping from the codon table
    pos_to_codon = {}
    pos_to_aa = {}
    for _, row in df_seq_codon.iterrows():
        pos_to_codon[row["pos"]] = row["codon"]
        pos_to_aa[row["pos"]] = row["aa"]

    df = df.copy()
    df = df[df["mutated_aa"].isin(STANDARD_AAS)].reset_index(drop=True)

    df["codon"] = df["pos"].apply(lambda x: pos_to_codon[x])  # Wild type codon

    # Validate whether the wild type amino acid sequence matches the one in the dataframe
    for _, row in df.iterrows():
        if pos_to_aa[row["pos"]] != row["original_aa"]:
            print(
                f"Amino acid mismatch at position {row['pos']} (internal: {pos_to_aa[row['pos']]}, provided: {row['original_aa']})"
            )

    # Give each mutation a unique ID
    df["mutation_id"] = ""  # Initialize the mutation ID
    for i, row in df.iterrows():
        df.at[i, "mutation_id"] = f"{row['codon']}->{row['mutated_aa']}"

    # Compute the intermediate scores
    df["semantic_z"] = standardize_score(df["semantic"])  # Standardized semantic change
    df["log_grammar"] = np.log(df["grammar"] + 1e-15)  # Log grammaticality
    df["log_grammar_z"] = standardize_score(df["log_grammar"])  # Standardized log grammaticality

    return df


################################################################################
# CLIB Calculations
################################################################################


def get_Q(strain: str) -> np.ndarray:
    """
    Compute the transition rate matrix Q,
    normalized so that the average mutation rate is 1 per unit time.

    Args
    ----
    virus_name : SBSType
        The name of the virus for which to compute the transition rate matrix.

    Returns
    -------
    Q : np.ndarray
        The transition rate matrix.
    """

    assert strain in STRAIN_DATA, f"Virus {strain} not found in the virus strain data."
    data = STRAIN_DATA[strain]
    assert "virus" in data, f"Virus name not found for strain {strain}."
    assert "stat-dist" in data, f"Stationary distribution not found for virus {strain}."

    sbs_virus = SBS(data["virus"])

    # Estimate the transition matrix modified by the stationary distribution
    stat_dist = np.array(data["stat-dist"], dtype=float)
    stat_dist = stat_dist / np.sum(stat_dist)
    Q = np.array([[0 if a == b else sbs_virus[f"*{a}*", b] for b in BASE] for a in BASE], dtype=float)

    # Normalize the modified transition matrix
    Q = Q / np.sum(Q)

    # Estimate the transition matrix
    for i in range(len(BASE)):
        Q[i] = Q[i] / stat_dist[i]
        Q[i, i] = -Q[i].sum()

    return Q


def get_clib_table(
    Q: np.ndarray,
    t: float,
) -> dict[str, float]:
    """
    Compute the CLIB table based on the average mutation rate.

    Parameters
    ----------
    Q : np.ndarray
        The transition rate matrix, normalized so that the average mutation rate is 1 per unit time.
        This is typically computed using the `get_Q` function.
    t : float
        the time scale based on the average number of mutations.

    Returns
    -------
    clib_table : dict[str, float]
        The CLIB table.
        * `key` : the mutation ID of the form `"XXX->A"` where `XXX` is the codon and `A` is the amino acid alphabet.
        * `value` : the CLIB score.
    """
    # Compute the transition matrix
    P = expm(t * Q)

    # Build the raw PIM table based on the position-independence assumption:
    # dict_pimcr_raw[letter][i0, i1, i2] = [P[i0, c[0]] * P[i1, c[1]] * P[i2, c[2]] for c in codon_indices]
    dict_clib_raw = {
        letter: np.sum(
            P[:, None, None, codon_indices[:, 0]]
            * P[None, :, None, codon_indices[:, 1]]
            * P[None, None, :, codon_indices[:, 2]],
            axis=3,
        )
        for letter, codon_indices in LETTER_TO_CODONS.items()
    }

    # Build the PIM table
    # pim_table[mut_id] = PIM score
    clib_table: dict[str, float] = {
        f"{BASE[i]}{BASE[j]}{BASE[k]}->{letter}": clib_tensor[i, j, k]
        for letter, clib_tensor in dict_clib_raw.items()
        for i, j, k in product(range(4), repeat=3)
    }

    # Exception for the empty IDs
    clib_table[""] = 0.0

    return clib_table


def get_log_clib_z(
    df: pd.DataFrame,
    Q: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute the standardized log CLIB scores for each mutation ID.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the inference results, with a column `mutation_id`.
    Q : np.ndarray
        The transition rate matrix, normalized so that the average mutation rate is 1 per unit time.
        This is typically computed using the `get_Q` function.
    t : float
        The time scale based on the average number of mutations.

    Returns
    -------
    log_clib_z : np.ndarray
        The standardized log CLIB scores for each mutation ID.
    """

    # Compute the
    clib_table = get_clib_table(Q, t)

    # Compute the raw CLIB scores for each mutation ID
    clib = df["mutation_id"].map(clib_table)
    log_clib = np.log(clib + 1e-15)
    log_clib_z = standardize_score(pd.Series(log_clib)).to_numpy()

    return log_clib_z


def get_cacscs(
    df: pd.DataFrame,
    log_clib_z: np.ndarray,
    bary_coord: ArrayLike,
) -> np.ndarray:
    """
    Compute the evaluation metric for the given barycentric coordinates, raw implementation.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the inference results, with columns:
        * `semantic_z`: standardized semantic change score
        * `log_grammar_z`: standardized log grammaticality score

    log_clib_z : np.ndarray
        the standardized log CLIB scores for each mutation ID

    bary_coord : np.ndarray
        the barycentric coordinates for the combination of semantic change, grammaticality, and CLIB score

    Returns
    -------
        The computed CaCSCS score
    """
    bary_coord = np.array(bary_coord, dtype=float)
    return (
        (bary_coord[0] * df["semantic_z"] + bary_coord[1] * df["log_grammar_z"] + bary_coord[2] * log_clib_z)
        .rank(ascending=False, method="min")
        .astype(int)
    )


def minimize_bary_f(
    f: Callable[[np.ndarray], float],
    tolerance: float = 1e-3,
    max_recursion: int = 256,
) -> tuple[np.ndarray, float]:
    """
    An algorithm to find the minimum of a function on a triangle.
    The code is deliberately verbose and low-level to speed up the computation.

    Parameters
    ----------
    f : np.ndarray -> float
        The function to minimize, which is a function of a barycentric coordinate in the triangle.

    tolerance : float, default=1e-3
        The tolerance for the minimum finding algorithm.

    Returns
    -------
    min_point : np.ndarray
        The minimizer of the function.

    min_value : float
        The minimum value of the function.
    """
    # Vertices of the initial triangle and the center
    v_a = np.array([1, 0, 0], dtype=float)
    v_b = np.array([0, 1, 0], dtype=float)
    v_c = np.array([0, 0, 1], dtype=float)
    v_o = (v_a + v_b + v_c) / 3

    # Function values at the vertices
    f_a = f(v_a)
    f_b = f(v_b)
    f_c = f(v_c)
    f_o = f(v_o)

    for depth in range(max_recursion + 1):
        # Calculate the midpoints and the centroids of the subtriangles
        v_m_ab = (v_a + v_b) / 2
        v_m_bc = (v_b + v_c) / 2
        v_m_ca = (v_c + v_a) / 2
        v_ctr_a = (4 * v_a + v_b + v_c) / 6
        v_ctr_b = (v_a + 4 * v_b + v_c) / 6
        v_ctr_c = (v_a + v_b + 4 * v_c) / 6

        # Function values at the vertices
        f_m_ab = f(v_m_ab)
        f_m_bc = f(v_m_bc)
        f_m_ca = f(v_m_ca)
        f_ctr_a = f(v_ctr_a)
        f_ctr_b = f(v_ctr_b)
        f_ctr_c = f(v_ctr_c)

        # Mean values of the function on each of the subtriangles using a quadrature rule
        subtri_mean = np.array(
            [
                (f_a + f_m_ab + f_m_ca + 9 * f_ctr_a) / 12,
                (f_b + f_m_bc + f_m_ab + 9 * f_ctr_b) / 12,
                (f_c + f_m_ca + f_m_bc + 9 * f_ctr_c) / 12,
                (f_m_ab + f_m_bc + f_m_ca + 9 * f_o) / 12,
            ]
        )

        # If the convergence criterion is met, return the center and the mean value
        if np.max(subtri_mean) - np.min(subtri_mean) < tolerance:
            return v_o, f_o
        if depth == max_recursion:
            return v_o, np.mean(subtri_mean).astype(float)

        # Find the subtriangle with the minimum mean value
        idx_min = np.argmin(subtri_mean)

        # Recursively find the minimum in the subtriangle
        if idx_min == 0:
            v_b = v_m_ab
            v_c = v_m_ca
            v_o = v_ctr_a
            f_b = f_m_ab
            f_c = f_m_ca
            f_o = f_ctr_a
        elif idx_min == 1:
            v_a = v_m_ab
            v_c = v_m_bc
            v_o = v_ctr_b
            f_a = f_m_ab
            f_c = f_m_bc
            f_o = f_ctr_b
        elif idx_min == 2:
            v_a = v_m_ca
            v_b = v_m_bc
            v_o = v_ctr_c
            f_a = f_m_ca
            f_b = f_m_bc
            f_o = f_ctr_c
        elif idx_min == 3:
            v_a = v_m_ab
            v_b = v_m_bc
            v_c = v_m_ca
            f_a = f_m_ab
            f_b = f_m_bc
            f_c = f_m_ca

    raise RuntimeError("Maximum recursion depth exceeded while minimizing the function.")


def find_minimizers(
    Q: np.ndarray,
    df: pd.DataFrame,
    t_schedule: np.ndarray,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """
    Find the minimizer of the score function for the given time schedule.

    Parameters
    ----------
    Q : np.ndarray
        The transition rate matrix, normalized so that the average mutation rate is 1 per unit time.

    df : pd.DataFrame
        The DataFrame containing the inference results, with columns:
        * `semantic_z`: standardized semantic change score
        * `log_grammar_z`: standardized log grammaticality score

    tolerance : float, default=1e-3
        Tolerance for the minimum finding algorithm.

    Returns
    -------
    result : np.ndarray
        An array containing the minimizers and the minimum values.
    """

    result = np.zeros((len(t_schedule), 4))

    for epoch, t in tqdm(enumerate(t_schedule), total=len(t_schedule)):
        # Compute the CLIB scores
        log_clib_z = get_log_clib_z(df, Q, t)

        # Internal function to compute the score based on barycentric coordinates
        def _get_score(bary_coord: np.ndarray) -> float:
            cacscs = get_cacscs(df, log_clib_z, bary_coord)
            mean_rank = cacscs[df["is_escape"] == True].mean()
            return mean_rank

        # Find the minimum barycentric coordinates and the corresponding score
        v_min, f_min = minimize_bary_f(_get_score, tolerance=tolerance)

        # Store the results
        result[epoch] = np.array([v_min[0], v_min[1], v_min[2], f_min])

    return result


if __name__ == "__main__":
    virus_name = "SARS-CoV-2"

    # Example usage
    Q = get_Q(virus_name)
    print("Transition rate matrix Q:\n", Q)

    # Example DataFrame
    SAMPLE_PATH = "inputs/analyze_semantics_cov_bilstm_512.txt"
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError(f"Sample data file not found: {SAMPLE_PATH}")

    df = pd.read_csv(SAMPLE_PATH, sep="\t", header=0)
    df = rename_df(df)
    df = prepare_df(df, virus_name)

    # Calculate the CLIB scores
    t = 0.1  # Example time scale
    log_clib_z = get_log_clib_z(df, Q, t)
    print("Normalized Log-CLIB computed")

    # Compute the CaCSCS score
    bary_coord = np.array([1, 1, 1]) / np.sqrt(3)  # Example barycentric coordinates
    cacscs = get_cacscs(df, log_clib_z, bary_coord)

    # Compute the mean rank for escape mutations
    mean_rank = cacscs[df["is_escape"] == True].mean()
    print("CaCSCS:", mean_rank)

    # Compute minimizers for a range of time scales
    t_schedule = np.linspace(0.01, 1.0, 10)  # Example time schedule
    minimizers = find_minimizers(Q, df, t_schedule, tolerance=1e-3)
    print("Minimizers:\n", minimizers)
