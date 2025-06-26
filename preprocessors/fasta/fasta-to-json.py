"""
DEVELOPMENT USE ONLY

In the production environment, the `cov_all.fa` file is already converted to JSON format.

---

This script converts `cov_all.fa` to a JSON file.

Since `cov_all.fa` contains data of mixed formats, the script parses the FASTA file and extracts the relevant information.
"""

from Bio import SeqIO
import json
import os

fasta_file = "../data/cov_all.fa"
output_file = "../outputs/cov_all.json"


host_convert = {
    "": "Unknown",
    "Munia": "Munia",
    "Canine": "Dog",
    "Panthera_tigris_jacksoni": "Malayan Tiger",
    "Dabbling_Duck": "Duck",
    "Ferret": "Ferret",
    "Pangolin": "Pangolin",
    "Pig": "Pig",
    "Monkey": "Monkey",
    "Weasel": "Weasel",
    "Gerbil": "Gerbil",
    "Environment": "Unknown",
    "Mink": "Mink",
    "Common_Moorhen": "Common Moorhen",
    "Vole": "Vole",
    "Bustard": "Bustard",
    "Camel": "Camel",
    "Mustela_lutreola": "European Mink",
    "Swine": "Pig",
    "Peafowl": "Peacock",
    "Shrew": "Shrew",
    "_Human": "Human",
    "Cattle": "Cattle",
    "Dog": "Dog",
    "Bovine": "Cattle",
    "Antelope": "Antelope",
    "Homo sapiens": "Human",
    "Duck": "Duck",
    "Rhinolophus_malayanus": "Malayan Horseshoe Bat",
    "Mustela lutreola": "European Mink",
    "Panthera tigris jacksoni": "Malayan Tiger",
    "Deer": "Deer",
    "Beluga_Whale": "Beluga Whale",
    "Goat": "Goat",
    "White_Eye": "White-eye",
    "Bulbul": "Bulbul",
    "Turkey": "Turkey",
    "Rhinolophus_affinis": "Intermediate Horseshoe Bat",
    "Night_Heron": "Night Heron",
    "Sparrow": "Sparrow",
    "Rat": "Rat",
    "Mouse": "Mouse",
    "Human": "Human",
    "Manis_javanica": "Sunda Pangolin",
    "Falcon": "Falcon",
    "Pheasant": "Pheasant",
    "Chicken": "Chicken",
    "Unknown": "Unknown",
    "Chinchilla": "Chinchilla",
    "Pigeon": "Pigeon",
    "Bat": "Bat",
    "Panda": "Panda",
    "Horse": "Horse",
    "Dolphin": "Dolphin",
    "Civet": "Civet",
    "Alpaca": "Alpaca",
    "Thrush": "Thrush",
    "Felis_catus": "Domestic Cat",
    "Shelduck": "Shelduck",
    "Chimpanzee": "Chimpanzee",
    "Giraffe": "Giraffe",
    "Canis lupus familiaris": "Dog",
    "Quail": "Quail",
    "Rabbit": "Rabbit",
    "Hyena": "Hyena",
    "Jerboa": "Jerboa",
    "Goose": "Goose",
    "Cat": "Cat",
    "NA": "Unknown",
    "Magpie_Robin": "Magpie Robin",
    "Buffalo": "Buffalo",
}


output = {}
host_types = set()
uniques = set()

for record in SeqIO.parse(fasta_file, "fasta"):
    fields = [field.strip() for field in record.description.split("|")]
    id = None
    host = None

    if len(fields) >= 7 and fields[0] == "Spike":
        id = fields[3]
        host = fields[6]
    elif len(fields) >= 9 and fields[1] in [
        "S_protein",
        "spike",
        "Spike",
        "Spike_protein",
        "spike_protein",
        "spike_protein_S",
        "spike_protein_precursor",
        "Spike__S__glycoprotein",
        "spike_S_glycoprotein",
        "spike_glyprotein",
        "spike_glycprotein",
        "Spike_glycoprotein",
        "spike_glycoprotein",
        "spike_glycoprotein_S",
        "spike_glycoprotein_S2",
        "spike_glycoprotein__S_",
        "spike_glycoprotein_precursor",
        "spike_glycoprotein_mature_peptide",
        "spike_structural_protein",
        "spike_surface_glycoprotein",
        "truncated_spike_protein",
        "truncated_spike_glycoprotein",
        "E2_glycoprotein",
        "E2_glycoprotein_precursor",
        "envelope_spike_glycoprotein",
        "mature_spike_glycoprotein__aa_1_1144_",
        "spike_protein__AA_1_1433_",
        "spike_glycoprotein__AA_1_1158_",
        "putative_E2_glycoprotein_precursor",
        "putative_spike_glycoprotein",
        "putative_spike_glycoprotein_S",
        "Infectious_bronchitis_virus__M41__RNA_for_spike_protein",
        "virus_envelope_protein_spike",
        "S_peplomer_polypeptide_precursor",
        "surface_protein",
        "surface_glycoprotein",
        "surface_glycoprotein_S",
        "surface_spike_glycoprotein",
        "anti_receptor_protein",
        "E2",
        "S",
    ]:
        id = fields[3]
        host = fields[8]
    elif (
        len(fields) >= 3
        and fields[1].startswith("Chain")
        and fields[2] in ["SARS-CoV-2 spike glycoprotein", "Spike glycoprotein"]
    ):
        id = fields[0]
        host = ""
    elif (
        len(fields) >= 3
        and fields[1] == "surface glycoprotein"
        and fields[2] == "partial [Severe acute respiratory syndrome coronavirus 2]"
    ):
        id = fields[0]
        host = fields[3]
    elif len(fields) >= 3:
        id = record.id
        host = fields[2]
    if id is not None and len(id) > 40:
        print(f"Warning: ID is longer than 40 characters.: {id}")

    if host == "partial [Severe acute respiratory syndrome coronavirus 2]":
        print(record.description)
    if host == "Spike glycoprotein":
        print(record.description)

    seq = str(record.seq)
    if host in host_convert and seq not in uniques:
        host = host_convert[host]
        output[id] = {"seq": seq, "host": host}
        uniques.add(seq)

if not os.path.exists("../outputs"):
    os.makedirs("../outputs")
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Converted {fasta_file} to {output_file}")
print(f"Total unique sequences: {len(uniques)}")