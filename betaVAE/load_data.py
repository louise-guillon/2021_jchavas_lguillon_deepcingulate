# -*- coding: utf-8 -*-
# /usr/bin/env python3


"""
Tools in order to create pytorch dataloaders
"""
import os
import sys

import pandas as pd
import numpy as np
from preprocess import *


def create_subset(config):
    """
    Creates dataset HCP_1 from HCP data

    Args:
        config: instance of class Config

    Returns:
        subset: Dataset corresponding to HCP_1
    """
    train_list = pd.read_csv(config.subject_dir, header=None, usecols=[0],
                             names=['subjects'])
    train_list['subjects'] = train_list['subjects'].astype('str')

    tmp = pd.read_pickle(os.path.join(config.data_dir, "Rskeleton.pkl")).T
    tmp.index.astype('str')

    tmp = tmp.merge(train_list, left_on = tmp.index, right_on='subjects', how='right')

    filenames = list(train_list['subjects'])

    subset = SkeletonDataset(dataframe=tmp, filenames=filenames)

    return subset
