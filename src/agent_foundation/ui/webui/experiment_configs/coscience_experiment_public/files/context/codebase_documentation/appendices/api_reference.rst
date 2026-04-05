==============
API Reference
==============

This appendix provides a quick reference to the main APIs in both systems.

Generative Recommenders API
===========================

Main Entry Points
-----------------

``main.py``
~~~~~~~~~~~

.. code-block:: bash

   python3 main.py --gin_config_file=<config_file> --master_port=<port>

``preprocess_public_data.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 preprocess_public_data.py

``run_fractal_expansion.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 run_fractal_expansion.py \
       --input-csv-file <input> \
       --write-dataset <True/False> \
       --output-prefix <output>

Core Modules
------------

``generative_recommenders.modules.stu``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``STU`` - Abstract base class
- ``STULayer`` - Single layer implementation
- ``STUStack`` - Stack of layers
- ``STULayerConfig`` - Configuration dataclass

``generative_recommenders.modules.hstu_transducer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``HSTUTransducer`` - Full model pipeline

``generative_recommenders.ops.hstu_attention``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hstu_mha()`` - Multi-head attention
- ``delta_hstu_mha()`` - Cached incremental attention

``generative_recommenders.ops.hstu_compute``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hstu_preprocess_and_attention()`` - Fused preprocessing
- ``hstu_compute_output()`` - Fused output computation

``generative_recommenders.dlrm_v3.configs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``get_hstu_configs()`` - Configuration factory
- ``get_embedding_configs()`` - Embedding configuration

Wukong API
==========

PyTorch
-------

``model.pytorch.Wukong``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from model.pytorch import Wukong

   model = Wukong(
       num_layers=int,
       num_sparse_emb=int,
       dim_emb=int,
       dim_input_sparse=int,
       dim_input_dense=int,
       num_emb_lcb=int,
       num_emb_fmb=int,
       rank_fmb=int,
       num_hidden_wukong=int,
       dim_hidden_wukong=int,
       num_hidden_head=int,
       dim_hidden_head=int,
       dim_output=int,
       dropout=float,
   )

   output = model(sparse_inputs, dense_inputs)

TensorFlow
----------

``model.tensorflow.Wukong``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from model.tensorflow import Wukong

   model = Wukong(
       num_layers=int,
       num_sparse_emb=int,
       dim_emb=int,
       num_emb_lcb=int,
       num_emb_fmb=int,
       rank_fmb=int,
       num_hidden_wukong=int,
       dim_hidden_wukong=int,
       num_hidden_head=int,
       dim_hidden_head=int,
       dim_output=int,
       dropout=float,
   )

   output = model([sparse_inputs, dense_inputs])

Component Classes
-----------------

Both frameworks provide:

- ``Embedding`` - Feature embedding layer
- ``MLP`` - Multi-layer perceptron
- ``LinearCompressBlock`` - LCB component
- ``FactorizationMachineBlock`` - FMB component
- ``ResidualProjection`` - Residual projection
- ``WukongLayer`` - Complete interaction layer

Dependencies
============

Generative Recommenders
-----------------------

.. code-block:: text

   torch>=2.6.0
   fbgemm_gpu>=1.1.0
   torchrec>=1.1.0
   gin_config>=0.5.0
   pandas>=2.2.0
   tensorboard>=2.19.0
   pybind11

Wukong
------

.. code-block:: text

   torch>=2.2
   tensorflow>=2.16
   pytest>=8.1  (optional, for tests)
