"""
A helper class for analyzing single base substitutions (SBS) in viral genomes.

Classes
-------
SBS
    A helper class for analyzing single base substitutions (SBS) in viral genomes.
"""

import pandas as pd
import numpy as np
import os

from typing import Literal, TypeAlias
from itertools import product


SBS_VIRUS_TYPES = [
    "SARS-CoV-2",
    "SARS-CoV",
    "MERS",
    "Other Betacoronavirus",
    "Influenza A",
    "HIV",
    "EBV",
    "Alphacoronavirus",
    "Torovirinae",
    "Roniviridae",
    "Mesoniviridae",
    "Arteriviridae",
]

BASES = "ACGT"

PATH_SBS_FREQUENCY = "./data/sbs_freq.csv"

SBSType: TypeAlias = Literal[
    "SARS-CoV-2",
    "SARS-CoV",
    "MERS",
    "Other Betacoronavirus",
    "Influenza A",
    "HIV",
    "EBV",
    "Alphacoronavirus",
    "Torovirinae",
    "Roniviridae",
    "Mesoniviridae",
    "Arteriviridae",
]
"""
A type alias for the supported virus types in the SBS analysis.
"""

assert os.path.exists(PATH_SBS_FREQUENCY), "SBS frequency data not found!"

db_sbs_freq = pd.read_csv(PATH_SBS_FREQUENCY)


class SBS:
    """
    A helper class for analyzing single base substitutions (SBS) in viral genomes.
    
    The SBS object supports indexing syntax to access the substitution frequencies:
    * `sbs[codon]` = Total number of SBS occurrences for `codon`.
    * `sbs[codon, new_base]` = Total number of SBS occurrences for `codon` in which the middle base changes to `new_base`.
    * The argument `codon` accepts codons (base triples) with wildcard support. Wildcards are either `*` or `_`. (Example: `__T` stands for any codon with the third base matching `T`.)
    * If `codon` is a single base, it will be treated as a codon with the given base in the middle. (Example: `A` is equivalent to `_A_`.)
    """

    def __init__(self, type: SBSType) -> None:
        """
        Initialize a new SBS object with the given virus type.

        Parameters
        ----------
        type : SBSType
            The type of virus to analyze. Check the literal type `SBSType` for available options.
        """

        assert type in SBS_VIRUS_TYPES, f"Invalid SBS type: {type}"

        self.type = type
        self.data = {f"{x}{y}{z}": {t: 0 for t in BASES} for x, y, z in product(BASES, repeat=3)}

        for row in db_sbs_freq.iterrows():
            sbs_str = str(row[1]["Sub Type"])

            sbs_original = sbs_str[0:2] + sbs_str[4:]
            sbs_substitute = sbs_str[3:4]

            self.data[sbs_original][sbs_substitute] = row[1][type]

    def __str__(self) -> str:
        return f"SBS({self.type})"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: str | tuple[str, str]) -> int:
        if isinstance(key, str):
            codon, subs = key, "ACGT"
        elif isinstance(key, tuple):
            assert len(key) == 2, f"Invalid key: {key}"
            codon, subs = key
        else:
            raise ValueError("Invalid key type")

        assert isinstance(codon, str), f"Invalid codon: {codon}"
        assert isinstance(subs, str), f"Invalid substitution: {subs}"
        assert len(codon) == 3 or (len(codon) == 1 and codon in BASES), f"Invalid codon: {codon}"

        if len(codon) == 1:
            codon = f"_{codon}_"

        if not (codon in self.data):
            res_acc = {t: 0 for t in BASES}

            def base_iter(x: str):
                if x in ["*", "_"]:
                    return BASES
                elif x in BASES:
                    return x
                else:
                    raise ValueError(f"Invalid base: {x}")

            for base_triple in product(base_iter(codon[0]), base_iter(codon[1]), base_iter(codon[2])):
                res = self.data["".join(base_triple)]
                for t in BASES:
                    res_acc[t] += res[t]

            self.data[codon] = res_acc

        return np.sum([n for t, n in self.data[codon].items() if t in subs])

    def freq(self, codon: str, subs: str) -> float:
        """
        `prob(codon, new_base)`

        Compute the relative frequency of the SBS occurrences in which the middle base of the codon `codon` changes to the new base `new_base`.
        """
        return self[codon, subs] / self[codon]


# Unit tests
if __name__ == "__main__":
    sbs = SBS("SARS-CoV-2")
    print("Number of SBS occurrences matching T__ > A:", sbs["T__", "A"])
    print("Relative frequency of SBS occurrences matching T__ > A:", sbs.freq("T__", "A"))
    print("Unit test done.")