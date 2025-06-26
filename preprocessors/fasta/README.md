# What is this?

This is a one-time use code for processing SARS-CoV-2 FASTA files into embedding vectors with metadata.


## FASTA to JSON

The file `fasta-to-json.py` converts the FASTA Files into a JSON file containing both sequences and their metadata.

## Embedding sequences

Once the above code has been successfully executed, `outputs/cov_all.json` will be created.
Then the file `embed-fasta.py` calculates the embedding vectors of the sequences in this file
and then stores them as `.npz` file for future use.