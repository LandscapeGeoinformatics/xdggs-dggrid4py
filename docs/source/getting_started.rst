Installation
============

In order to work with IGEO7 (using `DGGRID <https://github.com/sahrk/DGGRID>`_), the ``dggrid`` executable needs to be available. You can compile it yourself, or install into the conda/micromamba environment from conda-forge:

.. code-block:: bash

    micromamba install -c conda-forge dggrid
    export DGGRID_PATH=<path to dggrid executable>


Users can install the plugin from pip (under the same virtual environment of ``dggrid``):

.. code-block:: bash

   pip install xdggs-dggrid4py

To install the latest updates from GitHub (under the same virtual environment of ``dggrid``):


.. code-block:: bash

   pip install git+https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py


Installation for development
============================


Clone the source from github:

.. code-block:: bash
    
    git clone https://github.com/LandscapeGeoinformatics/xdggs-dggrid4py.git

Setup development environment:

.. code-block:: bash
    
    cd xdggs-dggrid4py
    micromamba create -n xdggs_dggrid4py_dev -f environment.yml
    micromamba active xdggs_dggrid4py_dev
    poetry install


    

