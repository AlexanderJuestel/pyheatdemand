.. _installation_ref:

Installation
============

PyHD is supported for Python version 3.10 and younger. Previous versions are officially not supported.
It is recommended to create a new virtual environment using Anaconda before using PyHD.
The main dependencies of PyHD are GeoPandas and Rasterio for the vector data and raster data processing, Matplotlib for plotting,
GeoPy for extracting coordinates from addresses, OSMnx for getting OpenStreet Maps building footprints from coordinates,
Rasterstats for analyzing the resulting heat demand maps and more secondary dependencies like Pandas, NumPy, Shapely, etc.

Installation via PyPi
~~~~~~~~~~~~~~~~~~~~~

PyHD can be installed via PyPi using::

    pip install pyhd


Installation via Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~~

PyHD is also available from conda-forge::

    conda install -c conda-forge pyhd


Forking or cloning the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyHD repository can be forked or cloned from https://github.com/AlexanderJuestel/pyhd::

    git clone https://github.com/AlexanderJuestel/pyhd.git

A list of `requirements.txt <https://github.com/AlexanderJuestel/pyhd/blob/main/requirements.txt>`_ and an `environment.yml <https://github.com/AlexanderJuestel/pyhd/blob/main/environment.yml>`_ can be found in the repository.
