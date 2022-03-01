#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Training SimCLR on skeleton images

"""
######################################################################
# Imports and global variables definitions
######################################################################
import logging

import hydra
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from SimCLR.data.datamodule import DataModule
from SimCLR.models.contrastive_learner import ContrastiveLearner
from SimCLR.utils.config import process_config

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = logging.getLogger(__name__)

"""
We use the following definitions:
- embedding or representation, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""


@hydra.main(config_name='config', config_path="configs")
def train(config):
    config = process_config(config)

    data_module = DataModule(config)

    model = ContrastiveLearner(config,
                               mode="encoder",
                               sample_data=data_module)

    summary(model, tuple(config.input_size), device="cpu")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.max_epochs,
        logger=tb_logger,
        flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        log_every_n_steps=config.log_every_n_steps,
        resume_from_checkpoint=config.checkpoint_path)

    trainer.fit(model, data_module)

    print("Number of hooks: ", len(model.save_output.outputs))


if __name__ == "__main__":
    train()
