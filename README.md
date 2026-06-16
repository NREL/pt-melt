# pt-melt

PT-MELT (PyTorch Machine Learning Toolbox) is a collection of architectures, processing, and utilities that are transferable over a range of ML applications.

A toolbox for researchers to use for machine learning applications in the PyTorch language. The goal of this software is to enable fast start-up of machine learning tasks and to provide a reliable and flexible framework for development and deployment. The toolbox contains generalized methods for every aspect of the machine learning workflow while simultaneously providing routines that can be tailored to specific application spaces.

## Environment

First, create a new conda environment and activate:

`conda create -n pt-melt python`

`conda activate pt-melt`

Finally, install the `ptmelt` as a package through pip either through a local install from a git clone

### Local git clone

If you cloned the repo and would like to install from the local git repo, navigate to the head directory where `setup.py` is located and type:

`pip install .`

If you want to update the pip install to make sure dependencies are current:

`pip install --upgrade .`

### Directly from github

To install the `ptmelt` package directly from github simply type:

pip install git+https://github.com/NREL/pt-melt.git

### Example Notebooks

If you want to run the example notebooks, they require a couple additional packages which can all be pip installed:

1. `scikit-learn`
2. `ipykernel`
3. `matplotlib`

## Hyperparameter tuning

PT-MELT includes native helpers for Ray Tune workflows in `ptmelt.utils.hp_tuning`.
The builder supports `ann`, `resnet`, `bnn`, `rnn`, `temporal_transformer`, and
`vae` model configurations.

```python
from ray import tune
from ptmelt.utils.hp_tuning import run_ray_tune

result = run_ray_tune(
	train_dl=train_dl,
	val_dl=val_dl,
	base_config={
		"arch_type": "ann",
		"num_features": num_features,
		"num_outputs": num_outputs,
		"epochs": 25,
		"learning_rate": 1e-3,
		"loss_fn": "mse",
	},
	search_space={
		"width": tune.choice([32, 64, 128]),
		"depth": tune.randint(1, 5),
	},
	metric="val_loss",
	mode="min",
	num_samples=20,
)

print(result.best_config)
print(result.best_hyperparameters)
print(result.metric_details)
```

`run_ray_tune` returns best configuration details, the searched hyperparameter
subset, selected metric details, and per-trial history dataframes.

## Contributing

pip install black isort flake8
