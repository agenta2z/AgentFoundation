Dataset Examples
================

This section provides hands-on Jupyter notebook examples demonstrating how to work
with datasets in the Generative Recommenders framework. These examples are designed
to help you quickly get started with data loading, exploration, and custom dataset
integration.

.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks

   quickstart
   data_exploration
   custom_integration
   training_workflow

Overview of Examples
--------------------

+-------------------------+--------------------------------------------------+
| Example                 | Description                                      |
+=========================+==================================================+
| Quick Start             | Load data in under 5 minutes                     |
+-------------------------+--------------------------------------------------+
| Data Exploration        | Analyze dataset statistics and distributions     |
+-------------------------+--------------------------------------------------+
| Custom Integration      | Add your own dataset to the framework            |
+-------------------------+--------------------------------------------------+
| Training Workflow       | End-to-end training with data loading            |
+-------------------------+--------------------------------------------------+

Running the Examples
--------------------

All examples can be run as Jupyter notebooks or Python scripts::

    # Option 1: Jupyter Notebook
    jupyter notebook examples/quickstart.ipynb

    # Option 2: Python Script
    python examples/quickstart.py

Prerequisites
-------------

Ensure you have the required dependencies installed::

    pip install -r requirements.txt
    pip install jupyter pandas matplotlib seaborn

Download the Data First
-----------------------

Before running examples, preprocess the public datasets::

    python preprocess_public_data.py

This downloads and prepares MovieLens-1M, MovieLens-20M, and Amazon Books.
