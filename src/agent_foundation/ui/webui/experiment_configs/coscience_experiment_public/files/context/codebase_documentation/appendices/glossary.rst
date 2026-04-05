========
Glossary
========

This glossary defines key terms used throughout the documentation.

Generative Recommenders Terms
=============================

DLRM
   Deep Learning Recommendation Model. Traditional architecture for recommendation systems
   using embedding tables and interaction layers.

GR
   Generative Recommender. The approach of reformulating recommendation as a generative
   modeling problem.

HSTU
   Hierarchical Sequential Transduction Unit. The core building block of Generative
   Recommenders, which processes sequential user interactions.

STU
   Sequential Transduction Unit. A single layer within the HSTU architecture,
   performing attention and output computation.

M-FALCON
   Microbatched-Fast Attention Leveraging Cacheable OperatioNs. Algorithm for
   efficient inference by caching K/V values and processing candidates in microbatches.

UIH
   User Interaction History. The sequence of past user interactions used as context
   for recommendations.

Target-Aware Attention
   Attention mechanism where candidate items can attend to user history but not
   to each other, enabling efficient batch inference.

Pointwise Aggregated Attention
   Non-softmax attention that preserves the intensity of user preferences,
   unlike normalized attention which dilutes strong signals.

Sampled Softmax
   Loss function that samples a subset of negative items during training,
   enabling efficient training with large item vocabularies.

KV Cache
   Key-Value cache storing pre-computed attention keys and values for efficient
   incremental inference.

Jagged Tensor
   Tensor representation for variable-length sequences without padding,
   using offset arrays to denote sequence boundaries.

Wukong Terms
============

FM
   Factorization Machine. Model architecture for capturing pairwise feature
   interactions efficiently.

FMB
   Factorization Machine Block. Component capturing higher-order feature interactions
   through optimized FM computation.

LCB
   Linear Compress Block. Component that linearly recombines embeddings without
   increasing interaction order.

Low-Rank Approximation
   Technique to reduce FM complexity from O(n²d) to O(nkd) by projecting to
   a lower-dimensional space.

Interaction Order
   The degree of feature combinations captured. Layer i in Wukong captures
   interactions of order 1 to 2^i.

Scaling Law
   The relationship between model complexity and quality improvement.
   Wukong demonstrates sustained improvement across two orders of magnitude.

General Deep Learning Terms
===========================

Embedding
   Dense vector representation of categorical features, learned during training.

Attention
   Mechanism for computing weighted combinations of values based on
   query-key similarity.

Multi-Head Attention
   Attention with multiple parallel attention heads, each learning different
   aspects of the relationships.

Layer Normalization
   Normalization technique that normalizes across features for each sample,
   commonly used in transformers.

Residual Connection
   Skip connection that adds the input to the output of a layer,
   enabling training of deep networks.

Dropout
   Regularization technique that randomly zeros elements during training
   to prevent overfitting.

Training Terms
==============

GIN
   Configuration format used by Generative Recommenders for hyperparameter
   management.

Distributed Training
   Training across multiple GPUs or machines using data parallelism.

Checkpoint
   Saved model state enabling training resumption or inference deployment.

Evaluation Metrics
==================

HR@K
   Hit Rate at K. Fraction of test samples where the true item appears
   in the top-K recommendations.

NDCG@K
   Normalized Discounted Cumulative Gain at K. Ranking metric that accounts
   for the position of relevant items.

AUC
   Area Under the ROC Curve. Metric measuring the model's ability to
   distinguish between positive and negative samples.
