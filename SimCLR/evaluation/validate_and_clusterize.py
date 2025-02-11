#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Validates and clusterizes

"""
######################################################################
# Imports and global variables definitions
######################################################################
import json
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from SimCLR.data.datamodule import DataModule
from SimCLR.data.datamodule import DataModule_Visualization
from SimCLR.evaluation.clustering import Cluster
from SimCLR.models.contrastive_learner_visualization \
    import ContrastiveLearner_Visualization
from SimCLR.utils.config import process_config
from SimCLR.utils.plots.visualize_tsne import plot_tsne
# from sklearn.cluster import OPTICS

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = logging.getLogger(__name__)

"""
We call:
- embedding, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""


@hydra.main(config_name='config', config_path="configs")
def postprocessing_results(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    config = process_config(config)

    # Sets seed for pseudo-random number generators
    # in: pytorch, numpy, python.random
    # seed_everything(config.seed)

    # Trick
    # Makes a dummy plot before invoking anatomist in headless mode
    if not config.analysis_path:
        plot = plt.figure()
        plt.ion()
        plt.show()
        plt.pause(0.001)

    data_module = DataModule_Visualization(config)
    data_module.setup(stage='validate')

    # Show the views of the first skeleton after each epoch
    model = ContrastiveLearner_Visualization(config,
                                             mode="encoder",
                                             sample_data=data_module)
    model = model.load_from_checkpoint(config.checkpoint_path,
                                       config=config,
                                       mode="encoder",
                                       sample_data=data_module)
    summary(model, tuple(config.input_size), device="cpu")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.max_epochs,
        logger=tb_logger,
        flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        resume_from_checkpoint=config.checkpoint_path)
    result_dict = trainer.validate(model, data_module)[0]
    embeddings, filenames = model.compute_representations(
        data_module.train_val_dataloader())

    # Gets coordinates of first views of the embeddings
    nb_first_views = (embeddings.shape[0]) // 2
    index = np.arange(nb_first_views) * 2
    embeddings = embeddings[index, :]
    filenames = filenames[::2]

    # plot_knn_buckets(embeddings=embeddings,
    #                 dataset=data_module.dataset_train,
    #                 n_neighbors=6,
    #                 num_examples=3
    #                 )

    # log.info("knn meshes done")

    # plot_knn_examples(embeddings=embeddings,
    #                   dataset=data_module.dataset_val,
    #                   n_neighbors=6,
    #                   num_examples=3,
    #                   savepath=config.analysis_path
    #                   )

    # log.info("knn examples done")

    # Makes Kmeans and represents it on a t-SNE plot
    X_tsne = model.compute_tsne(
        data_module.train_val_dataloader(),
        "representation")
    n_clusters = 2

    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    plot_tsne(X_tsne=X_tsne[index, :],
              buffer=False,
              labels=clustering.labels_,
              savepath=config.analysis_path,
              type='kmeans')
    for eps in [1., 1.5, 1.8, 2., 2.2, 2.5, 3., 3.5]:
        clustering = DBSCAN(eps=eps).fit(embeddings)
        # clustering = OPTICS().fit(embeddings)
        plot_tsne(X_tsne=X_tsne[index, :],
                  buffer=False,
                  labels=clustering.labels_,
                  savepath=config.analysis_path,
                  type=f"dbscan_{eps}")

    af = AffinityPropagation().fit(embeddings)
    cluster_labels_ini = af.labels_
    initial_centers = af.cluster_centers_indices_
    n_clusters_ = len(initial_centers)
    while n_clusters_ > 5:
        af = AffinityPropagation().fit(embeddings[af.cluster_centers_indices_])
        center_cluster_labels = af.labels_
        x_cluster_label = af.predict(embeddings)
        n_clusters_ = len(af.cluster_centers_indices_)
        print(n_clusters_)
    plot_tsne(X_tsne=X_tsne[index,
                            :],
              buffer=False,
              labels=x_cluster_label,
              savepath=config.analysis_path,
              type="af")

    cluster = Cluster(X=embeddings, root_dir=config.analysis_path)
    silhouette_dict = cluster.plot_silhouette()
    result_dict.update(silhouette_dict)
    result_dict.update({
        "latent_space_size": config.num_representation_features,
        "temperature": config.temperature})

    # Saves results in files
    with open(f"{config.analysis_path}/result.json", 'w') as fp:
        json.dump(result_dict, fp)
    torch.save(embeddings, f"{config.analysis_path}/train_val_embeddings.pt")
    with open(f"{config.analysis_path}/train_val_filenames.json", 'w') as f:
        json.dump(filenames, f, indent=2)


if __name__ == "__main__":
    postprocessing_results()
