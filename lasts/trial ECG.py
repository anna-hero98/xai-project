from blackboxes.loader import cached_blackbox_loader
from datasets.datasets import build_cbf
from autoencoders.variational_autoencoder import load_model
from utils import get_project_root, choose_z
from surrogates.shapelet_tree import ShapeletTree
from neighgen.counter_generator import CounterGenerator
from wrappers import DecoderWrapper
from surrogates.utils import generate_n_shapelets_per_size
from explainers.lasts import Lasts
import numpy as np
import sys
import os

# Explicitly add the correct path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


random_state = 0
np.random.seed(random_state)
dataset_name = "cbf"

_, _, _, _, _, _, X_exp_train, y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test = build_cbf(n_samples=600, random_state=random_state)

blackbox = cached_blackbox_loader("cbf_knn.joblib")
encoder, decoder, autoencoder = load_model(get_project_root() / "autoencoders" / "cached" / "vae" / "cbf" / "cbf_vae")

i = 0
x = X_exp_test[i].ravel().reshape(1, -1, 1)
z_fixed = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0], blackbox=blackbox, check_label=True, mse=False)

neighgen = CounterGenerator(blackbox, DecoderWrapper(decoder), n_search=10000)

n_shapelets_per_size = generate_n_shapelets_per_size(X_exp_train.shape[1])
surrogate = ShapeletTree(random_state=random_state, shapelet_model_kwargs={ "min_samples_split": [0.002, 0.01, 0.05, 0.1, 0.2]})

lasts_ = Lasts(blackbox, encoder, DecoderWrapper(decoder), neighgen, surrogate, verbose=True, binarize_surrogate_labels=True, labels=["cylinder", "bell", "funnel"])

lasts_.fit(x, z_fixed)

exp = lasts_.explain()

lasts_.plot("latent_space")
lasts_.plot("morphing_matrix")
lasts_.plot("counterexemplar_interpolation")
lasts_.plot("manifest_space")
lasts_.plot("saliency_map")
lasts_.plot("subsequences_heatmap")
lasts_.plot("rules")
lasts_.neighgen.plotter.plot_counterexemplar_shape_change()