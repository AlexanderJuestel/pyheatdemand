.. _installation_ref:

Installation
============

PyHeatDemand is supported for Python version 3.10 and younger. Previous versions are officially not supported.
It is recommended to create a new virtual environment using the `Anaconda Distribution <https://www.anaconda.com/download>`_ before using PyHeatDemand.
The main dependencies of PyHeatDemand are `GeoPandas <https://geopandas.org/en/stable/>`_ and `Rasterio <https://rasterio.readthedocs.io/en/stable/>`_ for the vector data and raster data processing, `Matplotlib <https://matplotlib.org/>`_ for plotting,
`GeoPy <https://geopy.readthedocs.io/en/stable/>`_ for extracting coordinates from addresses, `OSMnx <https://osmnx.readthedocs.io/en/stable/>`_ for getting `OpenStreet Maps <https://www.openstreetmap.org/#map=6/51.330/10.453>`_ building footprints from coordinates,
`Rasterstats <https://pythonhosted.org/rasterstats/>`_ for analyzing the resulting heat demand maps and more secondary dependencies like `Pandas <https://pandas.pydata.org/>`_, `NumPy <https://numpy.org/>`_, `Shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_, etc.

Installation via PyPi
---------------------

PyHeatDemand can be installed via `PyPi <https://pypi.org/>`_ using::

    pip install pyheatdemand


Installation via Anaconda
--------------------------

PyHeatDemand is also available from `conda-forge <https://conda-forge.org/>`_::

    conda install -c conda-forge pyheatdemand


Forking or cloning the repository
---------------------------------

The PyHeatDemand repository can be forked or cloned from https://github.com/AlexanderJuestel/pyheatdemand::

    git clone https://github.com/AlexanderJuestel/pyheatdemand.git

A list of `requirements.txt <https://github.com/AlexanderJuestel/pyheatdemand/blob/main/requirements.txt>`_ and an `environment.yml <https://github.com/AlexanderJuestel/pyheatdemand/blob/main/environment.yml>`_ can be found in the repository.
