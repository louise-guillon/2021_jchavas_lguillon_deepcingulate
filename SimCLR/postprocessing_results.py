#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
""" Training SimCLR on skeleton images

"""
######################################################################
# Imports and global variables definitions
######################################################################
import logging

import hydra
import os
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from SimCLR.contrastive_learner_visualization import ContrastiveLearner_Visualization
from SimCLR.datamodule import DataModule
from SimCLR.datamodule import DataModule_Visualization
from SimCLR.utils import process_config
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.cluster import OPTICS

from SimCLR.postprocessing.visualize_tsne import plot_tsne
from SimCLR.postprocessing.visualize_nearest_neighhbours import plot_knn_examples
from SimCLR.postprocessing.visualize_nearest_neighhbours import plot_knn_buckets

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
      
@hydra.main(config_name='config', config_path="config")
def postprocessing_results(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    config = process_config(config)

    # Sets seed for pseudo-random number generators
    # in: pytorch, numpy, python.random
    # seed_everything(config.seed)

    # Trick
    # Makes a dummy plot before invoking anatomist in headless mode
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
    trainer.test(model) 
    embeddings, _ = model.compute_representations(data_module.train_dataloader())
    
    # Gets coordinates of first views of the embeddings
    nb_first_views = (embeddings.shape[0])//2
    index = np.arange(nb_first_views)*2
    embeddings = embeddings[index, :]

    # plot_knn_buckets(embeddings=embeddings,
    #                 dataset=data_module.dataset_train,
    #                 n_neighbors=6,
    #                 num_examples=3
    #                 )
    
    # log.info("knn meshes done")

    plot_knn_examples(embeddings=embeddings,
                      dataset=data_module.dataset_train,
                      n_neighbors=6,
                      num_examples=3
                      )
    
    # log.info("knn examples done")

    # Makes Kmeans and represents it on a t-SNE plot
    X_tsne = model.compute_tsne(data_module.train_dataloader(), "representation")
    n_clusters = 2

    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    # clustering = DBSCAN(eps=2).fit(embeddings)
    # clustering = OPTICS().fit(embeddings)
    plot_tsne(X_tsne=X_tsne[index,:], buffer=False, labels=clustering.labels_)

    input("Press [enter] to continue.")

if __name__ == "__main__":
    postprocessing_results()
