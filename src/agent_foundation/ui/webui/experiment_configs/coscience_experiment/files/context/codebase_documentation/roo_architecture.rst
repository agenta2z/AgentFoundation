.. _cfr-roo-architecture:

====================================
ROO Architecture Deep Dive
====================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

ROO (Rank-Ordered Objectives) is a pairwise ranking optimization technique
that directly optimizes ranking quality instead of point-wise prediction.

ROO Motivation
==============

**Problem with Point-wise Prediction:**

.. code-block:: text

   Example User Request with 3 Candidates:
   - Post A: P(like) = 0.30
   - Post B: P(like) = 0.28
   - Post C: P(like) = 0.32

   Ranked by scores: [C, A, B]
   But what if user actually prefers A > C > B?

**ROO Solution:** Train to predict relative preferences, not absolute scores.

ROO vs Standard Model
=====================

.. figure:: ../../_static/cfr_roo_vs_pointwise.svg
   :alt: ROO vs Point-wise Model Comparison
   :align: center
   :width: 100%

   Comparison of standard point-wise vs ROO pairwise model

RO vs NRO Feature Split
=======================

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Type
     - Examples
     - Characteristics
     - Processing
   * - **RO (Request-Only)**
     - User demographics, user history, session context
     - Fixed per request, shared across all candidates
     - Processed once per request
   * - **NRO (Non-Request-Only)**
     - Post content, author info, engagement stats
     - Unique per candidate item
     - Processed per candidate

Dense Architecture ROO
======================

.. code-block:: python

   class DenseArchROO(nn.Module):
       """Dual-path dense processing for ROO"""

       def forward(
           self,
           ro_features: Tensor,   # [B, ro_dim]
           nro_features: Tensor,  # [B*N, nro_dim]
           id_score_embeddings: Dict[str, Tensor]
       ) -> Tensor:
           ro_output = self.ro_mhta(ro_features)
           nro_output = self.nro_mhta(nro_features)
           combined = self.combine_ro_nro(ro_output, nro_output)
           return combined

Training Loss
=============

**ROO Pairwise RankNet Loss:**

.. code-block:: python

   def roo_ranknet_loss(score_positive, score_negative, weights):
       P_ij = torch.sigmoid(score_positive - score_negative)
       loss = -torch.log(P_ij + 1e-8) * weights
       return loss.mean()

**Combined Loss (Production):**

.. code-block:: python

   def combined_loss(predictions_std, labels_std,
                     scores_positive, scores_negative,
                     weights, alpha=0.7, beta=0.3):
       std_loss = standard_loss(predictions_std, labels_std, weights)
       roo_loss = roo_ranknet_loss(scores_positive, scores_negative, weights)
       return alpha * std_loss + beta * roo_loss

ROO Benefits
============

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Point-wise
     - ROO Pairwise
   * - **Optimization**
     - P(click) calibration
     - Relative ranking quality
   * - **Loss Function**
     - BCE per sample
     - RankNet (pairwise BCE)
   * - **NDCG/MRR**
     - Suboptimal
     - Directly optimized

Next Steps
==========

- :doc:`hstu_datafm` - HSTU sequential modeling
- :doc:`training_workflow` - Complete training guide
