#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging

import matplotlib
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .visu_utils import buffer_to_image

logger = logging.getLogger(__name__)


def mscatter(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def compute_embeddings_skeletons(loader, model, num_outputs):
    X = torch.zeros([0, num_outputs]).cpu()
    with torch.no_grad():
        for (inputs, filenames) in loader:
            # First views of the whole batch
            inputs = inputs.cuda()
            model = model.cuda()
            X_i = model.forward(inputs[:, 0, :])
            # Second views of the whole batch
            X_j = model.forward(inputs[:, 1, :])
            # First views and second views are put side by side
            X_reordered = torch.cat([X_i, X_j], dim=-1)
            X_reordered = X_reordered.view(-1, X_i.shape[-1])
            X = torch.cat((X, X_reordered.cpu()), dim=0)
            del inputs
    return X


def compute_tsne(loader, model, num_outputs):
    X = compute_embeddings_skeletons(loader, model, num_outputs)
    tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)
    X_tsne = tsne.fit_transform(X.detach().numpy())
    return X_tsne


def plot_tsne(X_tsne, buffer, labels=None, savepath=None, type=""):
    """Generates TSNE plot either in a PNG image buffer or as a plot

    Args:
        X_tsne: TSNE N_features rows x 2 columns
        buffer (boolean): True -> returns PNG image buffer
                          False -> plots the figure
    """
    fig, ax = plt.subplots(1)
    logger.info(f"Matplotlib backend = {matplotlib.get_backend()}")
    logger.info(f"X_tsne shape = {X_tsne.shape}")
    nb_points = X_tsne.shape[0]
    m = np.repeat(["o"], nb_points)
    if labels is None:
        c = np.tile(np.array(["b", "r"]), nb_points // 2)
    else:
        c = labels

    mscatter(X_tsne[:, 0], X_tsne[:, 1], c=c, m=m, s=8, ax=ax)

    if buffer:
        return buffer_to_image(buffer=io.BytesIO())
    elif savepath:
        plt.savefig(f"{savepath}/tsne_{type}.png")
    else:
        plt.ion()
        plt.show()
        plt.pause(0.001)
