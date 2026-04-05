==================
Inference Workflows
==================

This document describes the inference workflows for deploying trained models.

Generative Recommenders Inference
=================================

M-FALCON Algorithm
------------------

The M-FALCON (Microbatched-Fast Attention Leveraging Cacheable OperatioNs) algorithm
enables efficient inference by caching K/V computations.

**Key Concepts**:

1. **User History Prefill**: Compute K, V for user history once
2. **KV Caching**: Store K, V in cache for reuse across candidates
3. **Microbatching**: Process candidates in batches of size ``m``
4. **Modified Attention**: Candidates attend to cached KV, not each other

**Complexity Reduction**:

- Without M-FALCON: O(m × n² × d)
- With M-FALCON: O((n + bm)² × d) ≈ O(n² × d) when bm << n

MLPerf Inference
----------------

DLRMv3 inference integrates with MLPerf LoadGen for standardized benchmarking:

.. code-block:: bash

   python3 generative_recommenders/dlrm_v3/inference/main.py \
       --config configs/inference.gin

Model Serving
-------------

For production serving:

.. code-block:: python

   from generative_recommenders.modules import HSTUTransducer

   # Load trained model
   model = HSTUTransducer(...)
   model.load_state_dict(torch.load("checkpoint.pt"))
   model.set_is_inference(True)

   # Process user history once
   with torch.no_grad():
       # Prefill KV cache with user history
       user_embeddings = preprocess_user_history(user_data)
       model.prefill_kv_cache(user_embeddings)

       # Process candidate batches
       for candidate_batch in candidate_batches:
           scores = model.cached_forward(candidate_batch)
           yield scores

Wukong Inference
================

PyTorch Inference
-----------------

.. code-block:: python

   import torch
   from model.pytorch import Wukong

   # Load trained model
   model = Wukong(...)
   model.load_state_dict(torch.load("wukong_checkpoint.pt"))
   model.eval()

   # Inference
   with torch.no_grad():
       sparse_inputs = torch.tensor(...)
       dense_inputs = torch.tensor(...)

       logits = model(sparse_inputs, dense_inputs)
       predictions = torch.sigmoid(logits)

TensorFlow Inference
--------------------

.. code-block:: python

   import tensorflow as tf
   from model.tensorflow import Wukong

   # Load trained model
   model = Wukong(...)
   model.load_weights("wukong_checkpoint")

   # Inference
   sparse_inputs = tf.constant(...)
   dense_inputs = tf.constant(...)

   logits = model([sparse_inputs, dense_inputs])
   predictions = tf.nn.sigmoid(logits)

Batch Inference Optimization
----------------------------

For large-scale batch inference:

.. code-block:: python

   def batch_inference(model, data_loader, device="cuda"):
       model.eval()
       model.to(device)

       all_predictions = []
       with torch.no_grad():
           for sparse_batch, dense_batch in data_loader:
               sparse_batch = sparse_batch.to(device)
               dense_batch = dense_batch.to(device)

               logits = model(sparse_batch, dense_batch)
               predictions = torch.sigmoid(logits)
               all_predictions.append(predictions.cpu())

       return torch.cat(all_predictions, dim=0)

Performance Tips
================

1. **Use Appropriate Kernel**: For inference, use ``TRITON_CC`` kernels on H100+ GPUs
2. **Enable KV Caching**: For sequential models, always enable KV caching
3. **Batch Appropriately**: Larger batches improve throughput; smaller batches reduce latency
4. **Use Mixed Precision**: FP16/BF16 inference can significantly improve throughput

Cross-References
================

- :doc:`/architecture/generative_recommenders` - GR architecture
- :doc:`/architecture/wukong` - Wukong architecture
- :doc:`training` - Training workflows
