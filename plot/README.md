# loss-to-loss-notebooks

This repo contains the data and notebooks needed to generate the figures for the paper [Loss-to-loss prediction: scaling laws for all datasets](https://arxiv.org/abs/2411.12925).

## Dependencies

The notebooks rely on the following dependencies:
- matplotlib
- seaborn
- numpy
- pandas
- scipy
- jax
- optax

## Data

The data is stored in 3 csv files.

- `sweep.csv` contains the data for the sweep of small models, including model hyperparameters and evaluation performance.
- `extrpolation.csv` contains similar data for the 3.3B models (1e21 FLOPs).
- `opt_params_data.csv` contains the outputs from running the scaling law fits in `curve_fitting.py` (which can take several minutes to re-generate). These fits are only based on the sweep data.

## Notebooks

We provide 5 notebooks to generate the figures in the paper.

- `loss_relationships.ipynb` generates all the figures from the main paper on loss-to-loss prediction.
- `accuracy_relationships.ipynb` generates the appendix figures on the relationship between loss and accuracy.
- `translation_experiment.ipynb` generates the figures for the translation experiments.
- `curve_fitting.ipynb` generates the scaling law plots from the appendix.
- `theory/make_plots_clean.ipynb` generates the figures for the appendix on linear models.

## Citation

Please cite the following paper:

```bibtex
@article{brandfonbrener2024loss,
      title={Loss-to-Loss Prediction: Scaling Laws for All Datasets}, 
      author={Brandfonbrener, David and Anand, Nikhil and Vyas, Nikhil and Malach, Eran and Kakade, Sham},
      journal={arXiv preprint arXiv:2411.12925},
      year={2024}
}
```
