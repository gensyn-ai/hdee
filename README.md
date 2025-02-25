# HDEE: Heterogeneous Domain Expert Ensemble
This repository contains the code and instructions to create HDEEs described in the paper titled ["HDEE: Heterogeneous Domain Expert Ensemble"](link).


## Requirements

Run the following commands to obtain the required packages:

```commandline
python -m pip install pdm
python -m pdm init 
python -m pdm install
```

## Running Experiments
To run an example HDEE setting given in the paper, follow the instructions given below.

The training run details are captured by hydra configs that are stored in the `/configs` folder.

The example scripts can be found in `/scripts` folder.

### Loading Datasets
To load datasets, make sure to replace <TOKEN_NAME> and <TOKEN> with your own HF token in the corresponding scripts.

To load and pretokenize OpenWebText dataset used to train the seed models, run script with:
```commandline
./scripts/load_openwebtext.sh
```

To load and pretokenize M2D2 trained domains, run script with:
```commandline
./scripts/load_trained_M2D2_domains.sh
```

To load and pretokenize other trained domains, run script with:
```commandline
./scripts/load_trained_other_domains.sh
```

*Split* each dataset into `train`, `validation` and `test` folders, if necessary. If these splits do not exist in the loaded HF dataset by default, then they can be split by the ratio 80-10-10. For example, OpenWebText dataset has a single split downloaded into `/dataset/openwebtext/raw/` folder, which should be split into `train`, `validation` and `test` folders under `/dataset/openwebtext/` wrt the split ratio.

*Update* the dataset paths in the corresponding `config` files.

### Training Seed Models

To train the seed models, run script with:
```commandline
./scripts/train_seed_models.sh
```

This script will train three seed models wrt the model sizes of Small (Close) setting (90M, 115M, 135M) described in the paper. To train different model sizes, update the `config` parameters of each expert given in `/configs/hdee_seed_models`.

### Training Heterogeneous Domain Experts

To train three iterations of the ensembles described in the paper, run script with:

```commandline
./scripts/train_domain_experts.sh
```

This script will train HDEEs wrt the domains given in the paper. At each iteration, three new domains will be trained on top of the existing experts. To train with different domains, download the corresponding datasets into `/datasets` folder and update the corresponding dataset paths of the training configs under `/config/hdee_3_iterations/train_domains` folder.

### Evaluating Ensembles

To evaluate the final iterations of the ensembles, run script with:

```commandline
./scripts/evaluate_ensembles.sh
```

This script will evaluate three ensembles: 
(i)$\texttt{M}_\texttt{Ho}$-$\texttt{I}_\texttt{Ho}$ (baseline): homogeneous model sizes and an equal number of steps for all models, (ii)$\texttt{M}_\texttt{Ho}$-$\texttt{I}_\texttt{He}$: homogeneous model sizes and unequal number of steps, (iii)$\texttt{M}_\texttt{He}$-$\texttt{I}_\texttt{Ho}$: heterogeneous model sizes and equal number of steps.

## Publication

```bibtex
@article{blagoev2025skippipe,
  title={HDEE: Heterogeneous Domain Expert Ensemble}, 
  author={Ersoy, O\u{g}uzhan and Kolehmainen, Jari and Andrade, Gabriel Passamani},
  year={2025},
  eprint={ },
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```
