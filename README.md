# clpc-cbm-model
This repo contains the code written to implement our model, including the experiments used in the paper.

As our paper focuses on the second stage, the outputs of stage 1 are provided below to simplify the reproduction. 

We tested our model on 3 different datasets: CUB, Derm7pt, RIVAL10.

# Datasets Used

- **CUB:** A dataset composed of 11,788 images of birds. There are 200 species and 312 concepts defined in the data.

- **Derm7pt:** A dataset composed of 2013 images of skin diseases. There are 34 classes and 28 concepts defined in the data.

- **RIVAL10:** A dataset composed of 26,384 images. There are 10 classes and 18 concepts defined in the data.

# Setup Instructions
**Download stage 1 outputs ($\boldsymbol{\hat c}$):**
- *placeholder_link*
- Store the unzipped folder in `PROJECT_ROOT/output`.

**(OPTIONAL) Download images.**
- [CUB](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).
- [Derm7pt](https://derm.cs.sfu.ca/Download.html).
- RIVAL10: Follow the instructions [here](https://github.com/mmoayeri/RIVAL10/tree/gh-pages).

**Install all dependencies.**
- `pip install -r requirements.txt`

# Experiment Reproduction

## Model Performance
To get the performance of the model for Top-1 classification, run [notebook/phase_2.ipynb](notebook/phase_2.ipynb), using index 0, 1, or 2 at the top of the notebook to choose the dataset. 
```python
datasets = ['CUB', 'Derm7pt', 'RIVAL10']
use_dataset = datasets[1]
```

## Intervention Efficiency
Run [notebook/experiments/intervention_accuracy.ipynb](notebook/experiments/intervention_accuracy.ipynb). Again, use same indices to select dataset.

## Robustness to Noise
Run [notebook/experiments/test_robustness.ipynb](notebook/experiments/test_robustness.ipynb). Again, use same indices to select dataset.
