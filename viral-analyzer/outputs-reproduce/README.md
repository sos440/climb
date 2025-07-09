# About

This folder contains selected outputs from the main program that are needed to reproduce the figures and statistical analyses.

## 1. Data preparation

Run the following code to generate data to be analyzed.

```bash
python main.py --escape "SARS-CoV-2-WildType>*" --result "SARS-CoV-2-WildType-*" --scores
```

Move the generated outputs files to `outputs-reproduce` folder.

## 2. Analysis

In this section, we explain what each code does and how they can be used to reproduce the results in the article and supplementary.

### 2.1. `draw-figures.py`

This script generates various summaries of rank distribution for different PLM backbones and validation sets, including:

* Text file containing mean $\pm$ s.d., five-number summary, AUC
* Box-whisker plot comparing mean rank for CSCS and CAC across various PLM backbones
* Bar chart comparing AUC for CSCS and CAC
* List plot for mean rank as a function of evolutionary parameter $T$
* Bar chart comparing mean rank as a function of model size (for ESM-2 family)

Run this script using

```bash
python draw-figures.py
```

### 2.2. `draw-fig-delta-rank.ipynb`

This notebook generates the heatmaps for the $\Delta$-rank ($\text{rank}_\text{CAC} - \text{rank}_\text{CSCS}$) across various PLM backbones.

### 2.3. `draw-fig-sbs-dist.ipynb`

This notebook generates charts demonstrating conditional/unconditional probability of single-base substitution.