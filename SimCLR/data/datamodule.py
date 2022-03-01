#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Data module
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from SimCLR.data.datasets import create_sets


class DataModule(pl.LightningDataModule):
    """Data module class
    """

    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config

    def setup(self, stage=None, mode=None):
        self.dataset_train, self.dataset_val, self.dataset_test, _ = \
            create_sets(self.config)

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  sampler=RandomSampler(self.dataset_train),
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test


class DataModule_Visualization(pl.LightningDataModule):
    """Data module class for visualization
    """

    def __init__(self, config):
        super(DataModule_Visualization, self).__init__()
        self.config = config

    def setup(self, stage, mode=None):
        self.dataset_train, self.dataset_val, self.dataset_test,\
            self.dataset_train_val = \
            create_sets(self.config, mode='visualization')

    def train_val_dataloader(self):
        loader_train = DataLoader(self.dataset_train_val,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test
