# ProCLIP
A Contrastive Learning-enhanced Protein Language Model for Protein-Protein interactions (PPIs) Prediction
<<<<<<< HEAD

## Quick Start 
Dependences

python>=3.10

As a prerequisite, you must have PyTorch (>=2.4) installed to use this repository.

The pre-trained weights of ProCLIP PPI classifiers are available. 

The pre-trained weights of the base ProCLIP will be released upon manuscript acceptance.

Benchmark datasets can be downloaded [here](https://zenodo.org/records/19503722).

```python
pip install fair-esm bidirectional-cross-attention h5py numpy pandas scikit-learn munch
```

## Usage 
Generate embeddings from pair sequences using the default configuration file: cfgs/inference_emb.yaml

```python
python 1_extract_emb.py
```

Predict interaction score for protein pairs. By default, the output score will be saved in 'results/interaction_score.csv'

```python
python 2_scan.py
```

To train a new classifier with customized data:
```python
python train_classifier.py 
```


