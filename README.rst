
Unsupervised Representation Learning of Cingulate Cortical Folding Patterns
------------

The project aims to study cortical folding patterns in the cingulae region thanks to unsupervised deep learning methods.


Installation
------------

We first install the singularity brainvisa image following https://brainvisa.info/web/download.html

We then enter the brainvisa environment:

.. code-block:: shell

    bv opengl=nv bash
    
We can use a virtual environment:

.. code-block:: shell

    virtualenv --python=python3 --system-site-packages venv
    . venv/bin/activate
    
We then download and install the present deep learning package:

.. code-block:: shell

    git clone https://github.com/XXXX_xxxx_xxxx_xxxx
    cd XXX
    pip3 install -e .
    
As an alternative to the last step, we can also use the provided requirements file:

.. code-block:: shell

    pip3 install -r requirements.txt
    
Training the models
-------------------
Data are available in the `data <data/>`_ directory.

To train and evaluate each model, we follow the corresponding README:
    * `README of beta-VAE model <betaVAE/readme.md>`_
    * `README of SimCLR model <SimCLR/README.rst>`_
    
Results
-------

.. image:: images/pipeline_v5.png
.. image:: images/clustering_v5.png
.. image:: images/ma.png

Dependencies
-----------
- python >= 3.6
- pytorch >= 1.4.0
- numpy >= 1.16.6
- pandas >= 0.23.3



