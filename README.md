# Wunderpus
# Installation

```bash
conda env create -f environment.yml
conda activate sem_adv
```

If package conflicts occur:

```bash
pip install -r requirements.txt
```

# Dataset Preparation

The adversarial examples used in this work are generated from the non-adversarial fake datasets introduced by Abdullah et al.

To obtain the dataset structure and preprocessing pipeline, please follow the instructions provided in the Evolving Threat repository:

https://github.com/secml-lab-vt/EvolvingThreat-DeepfakeImageDetect

After preparing the dataset according to their instructions, place our attack scripts in the corresponding attack directory and execute the commands as described by Abdullah et al.

# Reproducing Results

The repository contains all code necessary to reproduce the attack generation reported in the paper.


