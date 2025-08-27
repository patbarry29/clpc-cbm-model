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
- [Google Drive](https://drive.google.com/drive/folders/158wBzLsIdrGg5WggX5W4Vw2qb3bLdQG_?usp=sharing)
- Store the 3 folders in `PROJECT_ROOT/output`.

**(OPTIONAL) Download images.**
- [CUB](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).
- [Derm7pt](https://derm.cs.sfu.ca/Download.html).
- RIVAL10: Follow the instructions [here](https://github.com/mmoayeri/RIVAL10/tree/gh-pages).

**Install all dependencies.**
- `pip install -r requirements.txt`

**Build project (from root):**
- `pip install -e .`

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


# (OPTIONAL) Stage 1 Reproduction
To reproduce this, first the datasets must be downloaded from the 3 links above.

## Dataset Preparation

### CUB and Derm7pt datasets

Once the datasets are downloaded, place the images for each dataset into `PROJECT_ROOT/images/{dataset_name}`.

The `data` folder is also required, the files required for both CUB and Derm7pt datasets are already provided in the repo.

### RIVAL10

RIVAL10 is a special case, as the images are mixed with the concept descriptions for each image.
- So we must run a command to separate these into different folders.
  - ```bash
    mkdir -p RIVAL10/images/train/images RIVAL10/images/test/images

    find RIVAL10/train -type f -iname '*.JPEG' ! -iname '*_merged_mask.JPEG' -exec mv {} RIVAL10/images/train/images/ \;
    find RIVAL10/test -type f -iname '*.JPEG' ! -iname '*_merged_mask.JPEG' -exec mv {} RIVAL10/images/test/images/ \;
    ```
- Now that we have an `images` folder in RIVAL10, we can just move this folder to `PROJECT_ROOT/RIVAL10/images`.

Now there are 3 folders left in `PROJECT_ROOT/data/RIVAL10`: `meta`, `train`, `test`
- `meta`: 3 files required
  - `label_mappings`, `train_test_split_by_url`, `wnid_to_class.json`
- `train` and `test`, with subfolder `ordinary`:
  - consists of `.npy` and `.pkl` files
  - One of each for every image. They describe the concept values present in each image.

## Running Experiments
Now that the datasets are all in the correct places, the structure should look like this:

<img width="259" alt="image" src="https://github.com/user-attachments/assets/840f0ab4-f829-4b75-956b-fce06bc19d1d" />

With this in place, we can run stage 1. Simply go to `notebook/{dataset}/training.ipynb`, and run the notebook.
- Note that this is an intensive process, particularly for CUB and RIVAL10.
- The learned model will be saved locally.
- Run `notebook/{dataset}/test.ipynb` to get the test accuracy and store the outputs of the CNN locally.
