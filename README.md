# Reproducing the experiments
In order to reproduce our experiments the easiest way possible, we strongly urge you to use `poetry`. You can install all the necessary modules via:

    poetry install

## Gathering the datasets

### Synthetic data
Synthetic data can be generated via the following command:

    poetry run python3 src/preprocessing/generate_dynamic_graphs.py --type=TYPE

`TYPE` can be any of the following: `erdos_renyi`, `barabasi_albert_random`, `barabasi_albert_growth`
### Real-world data
#### Fetching the datasets
To store experimental data, create a new folder `data` as well as `data/raw_datasets`. 

After running the experiments, your folder structure should look like this:

```
filtration_surfaces/
├── data/
│   ├── raw_datasets/
│   ├── preprocessed_data/
│   └── labeled_datasets/
├── src/
│   └── ...
```

All raw datasets shall be stored in `raw_datasets`. To reproduce our experiments, you can use the following command from the source directory:

    wget -P data/raw_datasets https://www.chrsmrrs.com/graphkerneldatasets/infectious_ct1.zip

You can of course replace the link with a link to any other temporal dataset from TUDataset (available at https://chrsmrrs.github.io/datasets/docs/datasets/). Other temporal datasets are also supported, but we only provide a preprocessing script for the temporal datasets from TUDataset.

#### Preprocessing the datasets
We need to preprocess the datasets to match the dynamic graph representation that our model expects. To do so, run the following:

    poetry run python3 src/preprocessing/preprocess_datasets.py --dataset=DATASET --path=PATH_TO_THE_DATASET

So, to preprocess the `infectious_ct1` dataset, we would run:

    poetry run python3 src/preprocessing/preprocess_datasets.py --dataset=infectious_ct1 --path=./data/raw_datasets/infectious_ct1

## Training and testing our method
To train and test our method, run:

    poetry run python3 src/main.py --dataset=DATASET

where `DATASET` is the name of the folder in `data/labeled_datasets/` containing the dataset. This will construct filtration surfaces for the given dataset and store them in `data/preprocessed_data/DATASET` for later reuse. Afterwards, the filtration surfaces are used as an input for the random forest classifier. 

If the filtration curves already exists in `data/preprocessed_data/DATASET` (because they were already constructed by a previous run), then the filtration curves are directly loaded.

# 
This code adapts and extends existing code from O'Bray, Rieck and Borgwardt (2021) on filtration curves. Their code can be found at https://github.com/BorgwardtLab/filtration_curves and their corresponding paper shall be referenced as:

```bibtex
@inproceedings{OBray21a,
    title        = {Filtration Curves for Graph Representation},
    author       = {O'Bray, Leslie and Rieck, Bastian and Borgwardt, Karsten},
    doi          = {10.1145/3447548.3467442},
    year         = 2021,
    booktitle    = {Proceedings of the 27th ACM SIGKDD International
                Conference on Knowledge Discovery \& Data Mining~(KDD)},
    publisher    = {Association for Computing Machinery},
    address      = {New York, NY, USA},
    pubstate     = {inpress},
}
```