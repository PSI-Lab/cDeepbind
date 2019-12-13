# cDeepBind
cDeepBind is a fast and accurate model for learning sequence and structure specificities of RNA binding proteins.
  - [Download poster at MLCB 2019](https://github.com/PSI-Lab/cDeepbind/raw/master/docs/cdeepbind_poster_mlcb.pdf)
  - [Download pre-print on Biorxiv](https://www.biorxiv.org/content/10.1101/345140v1) 

## Model architecture
![arch](docs/cdeepbind_poster_schematic.png)


##  Reproducing the training and evaluation workflow

Setup an environment that contains all the necessary packages first.
```bash
conda env create -f environment.yml
conda activate cdeepbind
python setup.py develop
```

To train a new model follow:
```bash
python encode_data.py -d 2013
python train_model.py
```

To run evaluations on a pre-trained model follow:
```bash
python evaluate_model.py
```

## Results
![img](docs/model_comparison_paper.png)
![img](docs/model_scatter_paper.png)

