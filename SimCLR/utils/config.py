#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
log = logging.getLogger(__name__)


def process_config(config) -> DictConfig:
    """Does whatever operations on the config file
    """

    log.info(OmegaConf.to_yaml(config))
    log.info("Working directory : {}".format(os.getcwd()))
    config.input_size = eval(config.input_size)
    log.info("config type: {}".format(type(config)))
    return config
