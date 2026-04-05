Kernel Optimizations
====================

.. note::
   Line numbers reference ``fbs_8b028c_rke_opt`` codebase unless noted.
   For ``fbs_1ce9d8_jia`` (proposals target), line numbers may differ.

This document provides a comprehensive overview of GPU kernel optimizations and
related configurations in the ROO/HSTU model architecture.

Already Implemented Optimizations
---------------------------------

The following optimizations have been implemented in the codebase:

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Optimization
     - Commit/Source
     - Impact
     - Status
   * - NumCandidatesInfo Pattern
     - ``f7b3c5cf``
     - 2-8% QPS improvement
     - ✅ Done
   * - PT2 Input Preprocessor
     - ``fde87298``
     - Significant speedup
     - ✅ Done
   * - PT2 HSTU Loss
     - ``1bed91e0``
     - Kernel fusion (221→30 kernels)
     - ✅ Done
   * - PT2 on SIM
     - ``d03af38d``
     - Significant speedup
     - ✅ Done
   * - SDD Lite Pipeline
     - ``f220c6d5``
     - 4-5% QPS improvement
     - ⚠️ DISABLED
   * - Inplace Data Copy
     - ``f220c6d5``
     - -3% peak memory
     - ⚠️ DISABLED
   * - bf16 on task_arch
     - ``1f2108d0``
     - Memory/speed gains
     - ✅ Done
   * - int32 id_score_list
     - ``1ce9d80b``
     - 50% indices memory savings
     - ✅ Done
   * - activation_memory_budget = 0.05
     - V2 config
     - Enables activation recomputation
     - ✅ Done
   * - gradient_as_bucket_view=True
     - TorchRec default
     - ~4GB memory savings
     - ✅ Done
   * - pin_memory=True
     - DPP config
     - 16% faster CPU→GPU transfers
     - ✅ Done
   * - cache_load_factor=0.1
     - Prod config
     - UVM caching for embeddings
     - ✅ Done
   * - CacheAlgorithm.LFU
     - Prod config
     - Optimal for rec workloads
     - ✅ Done
   * - automatic_dynamic_shapes
     - PyTorch default
     - Dynamic shape handling
     - ✅ Done

Current Kernel Infrastructure
-----------------------------

The model utilizes several GPU kernel technologies:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Component
     - Location (fbs_8b028c_rke_opt)
     - Location (fbs_1ce9d8_jia)
   * - Triton via Spindle
     - ``pytorch_modules_roo.py:1436-1448``
     - Not verified
   * - SDPA (Flash Attention)
     - ``pytorch_modules_roo.py:879``
     - ``pytorch_modules_roo.py:746``
   * - FBGEMM load_library
     - ``pytorch_modules.py:97-100``
     - Not verified
   * - torch._inductor config
     - ``model_roo_v0.py:69-77``
     - Similar
   * - SwishLayerNorm
     - Numerous occurrences
     - Similar

Triton Kernel Selection
~~~~~~~~~~~~~~~~~~~~~~~

Spindle uses different Triton kernel variants for training vs inference:

.. code-block:: python

   kernel=HammerKernel.TRITON if is_train else HammerKernel.TRITON.TRITON_CC

- **Training**: ``HammerKernel.TRITON`` (standard Triton kernel)
- **Inference**: ``HammerKernel.TRITON.TRITON_CC`` (Triton CudaCompiled variant)

torch._inductor Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model configures PyTorch's inductor for kernel optimization
(``model_roo_v0.py:69-77``):

.. code-block:: python

   torch._functorch.config.activation_memory_budget = 0.05
   torch._functorch.config.activation_memory_budget_runtime_estimator = "flops"
   torch._functorch.config.activation_memory_budget_solver = "dp"

   torch._inductor.config.allow_buffer_reuse = False
   torch._inductor.config.combo_kernels = True
   torch._dynamo.config.suppress_errors = True

FBGEMM Integration
~~~~~~~~~~~~~~~~~~

FBGEMM provides GPU-optimized embedding operations (``pytorch_modules.py:97-100``):

.. code-block:: python

   try:
       torch.ops.load_library(
           "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
       )
   except OSError:
       pass

Scaled Dot-Product Attention (SDPA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model uses PyTorch's SDPA which dispatches to Flash Attention when available
(``pytorch_modules_roo.py:879``):

.. code-block:: python

   attn_output = torch.nn.functional.scaled_dot_product_attention(
       q, k, v, attn_mask=None, dropout_p=0.0,
   )

Known Issues & Uncertainties
----------------------------

⚠️ Gradient Flow Bug (CRITICAL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a known gradient flow issue with the ``@torch.no_grad()`` decorator:

.. list-table::
   :header-rows: 1
   :widths: 45 15 25 15

   * - File
     - Line
     - Decorators
     - Bug Present
   * - ``hammer/modules/utils.py``
     - 245-246
     - ``@torch.fx.wrap`` + ``@torch.no_grad()``
     - **YES**
   * - ``hammer/modules/sequential/utils.py``
     - 476
     - ``@torch.fx.wrap`` only
     - **NO**

**Reference**: ``key_insights.md`` Insight #4

⚠️ allow_buffer_reuse=False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The buffer reuse is disabled at ``model_roo_v0.py:75``:

.. code-block:: python

   torch._inductor.config.allow_buffer_reuse = False

The reason for this setting is unknown - may be due to correctness issues in
earlier PyTorch versions.

⚠️ activation_memory_budget Discrepancy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a discrepancy in the ``activation_memory_budget`` setting:

- ``model_roo_v0.py:69`` → **0.05** (VERIFIED)
- ``flattened_..._datafm.py:1331`` → **0.5** (per proposal_Q4.md:49)

Runtime precedence unclear - needs verification.

Future Optimization Opportunities
---------------------------------

1. **FlashAttention Backend Verification**

   Verify which backend SDPA dispatches to and consider explicitly forcing
   Flash Attention when appropriate.

2. **Selective Activation Checkpointing**

   Apply selective checkpointing for DeepCrossNet layers to reduce memory
   pressure.

3. **activation_memory_budget Tuning**

   Values of 0.1-0.3 may provide better tradeoffs than current 0.05 setting.

4. **Memory Profiling Hooks**

   Add profiling hooks to understand memory allocation patterns.

5. **SwishLayerNorm Kernel Fusion**

   Numerous SwishLayerNorm instances across the codebase present opportunities
   for kernel fusion to reduce memory bandwidth.

6. **NumTargetsInfo Extension**

   Proposed P0-11 builds on NumCandidatesInfo pattern for additional QPS gains.

Cross-References
----------------

- :doc:`architecture` - Overall system architecture
- :doc:`roo_architecture` - ROO-specific architecture details
- :doc:`hstu_datafm` - HSTU component documentation
- ``iteration_1/proposal_Q4.md`` - Original optimization proposals
- ``iteration_1/implementation/`` - Implementation tracking
