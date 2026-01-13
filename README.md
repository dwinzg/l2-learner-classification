# COLX 521 Lab 4

> Authors: Yusen Huang, Marco Wang, Darwin Zhang, and Tianhao Cao

## L2 Learner Classifier Project

**[Full Report](/reports/part_4_report.html)**

# About:
This project involves building an end-to-end supervised classification system for Native Language Identification (NLI) to predict the native language background of English learners based on their writing. Using a corpus of HTML documents scraped from the Lang-8 platform, the pipeline processes raw text to extract corpus linguistic features including document statistics, syntactic patterns via regular expressions, and lexicon-based style markers. The classifier specifically distinguishes between native speakers of Romance languages (French and Spanish) and East Asian languages (Japanese, Korean, and Mandarin) by analyzing how linguistic distance from English influences learner writing styles and error patterns.

```
.
├── data/
│   └── raw/
│   ├   ├── lang-8.zip
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
│
├── images/
│   └── part_5/
│       ├── dist_target.png
│       ├── categorical_distributions.png
│       ├── correlation.png
│       └── decision_tree.png
│
├── notebooks/
│   ├── Lab4.ipynb
│   └── part_3+5_notebook.ipynb
│
├── src/
│   ├── build_dataset.py
│   ├── part_1.py
│   ├── part_2_lexicon_pos.py
│   ├── part_2_stats.py
│   ├── part_3.py
│   ├── part_5.py
│   └── __init__.py
│
├── tests/
│   ├── test_part_1.py
│   └── test_part_2.py
│
├── main.py
├── environment.yml
└── README.md
```
# Pipeline Overview

The pipeline follows the structure of the lab assignment:

1. Part 1

Document iteration and basic feature extraction

2. Part 2

Lexicon-based and POS-based features

Statistical features

3. Part 3

Dataset construction

Feature vectorization (DictVectorizer)

Baseline decision tree

Feature ablation

Retraining with selected features

5. Part 5

Exploratory data analysis (EDA)

Hyperparameter tuning with GridSearchCV

Feature selection with RFECV

Final evaluation on dev and test sets

Automatic export of EDA figures

# How to Run the Pipeline
1. Create the environment
conda env create -f environment.yml
conda activate lab4

2. Run the full pipeline
python main.py

This will:

Build datasets

Train models

Perform feature ablation

Run hyperparameter optimization

Generate evaluation scores

Export EDA figures to images/part_5/

# EDA Outputs

When main.py is executed, the following figures are automatically generated:

dist_target.png — distribution of target labels

categorical_distributions.png — categorical feature distributions

correlation.png — feature correlation matrix

decision_tree.png — trained decision tree visualization

All figures are saved under:

images/part_5/

# Running Tests

Unit tests are provided for Parts 1 and 2.

pytest tests/

# Notebooks

Lab4.ipynb — original exploratory notebook

part_3+5_notebook.ipynb — cleaned notebook version aligned with pipeline code

# Reproducibility Notes

All paths are resolved relative to the project root

No manual intervention is required after setup

The pipeline is executable in non-interactive environments

EDA outputs are saved automatically (no inline plotting required)
