# Recommendation System Architecture: 2024-2025 Landscape

The recommendation systems field is experiencing its most transformative period
since deep learning's arrival, with **scaling laws emerging for the first time**
and a **paradigm shift from discriminative to generative models**. Meta's
trillion-parameter HSTU achieved **12.4% production gains**, while ByteDance now
processes **10,000-length sequences** in real-time. This convergence of
LLM-inspired architectures with industrial-scale deployment signals a
fundamental restructuring of how recommendations work.

## Executive summary: The top 10 breakthrough developments

The most impactful findings from 2024-2025 cluster around three themes:
architectural scaling, generative transformation, and production efficiency.

**Scaling breakthroughs** dominate the landscape. Meta's
**[Wukong](https://arxiv.org/abs/2403.02545)** paper establishes the first
empirically validated scaling law for recommendations—model quality improves as
a power law of training compute across two orders of magnitude, mirroring
GPT-style scaling. The **[HSTU architecture](https://arxiv.org/abs/2402.17152)**
(Hierarchical Sequential Transduction Units) scales to **1.5 trillion
parameters** while running 5.3-15.2x faster than FlashAttention2 through custom
kernels. Tsinghua/Tencent's research on
**[embedding collapse](https://arxiv.org/abs/2310.04400)** identifies why naive
model scaling fails: embedding matrices occupy low-dimensional subspaces. Their
multi-embedding solution delivered **3.9% GMV lift** (~hundreds of millions USD
annually) at WeChat Moments.

**Generative recommenders** are now production-proven.
[HSTU](https://arxiv.org/abs/2402.17152) reformulates recommendation as
sequential transduction—treating user actions like language tokens for next-item
prediction. Kuaishou's **[OneRec](https://arxiv.org/abs/2502.11583)** unifies
retrieval and ranking into a single generative model with preference alignment
via DPO, deployed to 300M+ daily users. Google's
**[TIGER](https://arxiv.org/abs/2305.05065)** introduced semantic IDs using
RQ-VAE, enabling items to share hierarchical tokens for improved cold-start
generalization.

**Long-sequence modeling** has broken through production barriers. ByteDance's
**STCA** (Spatial-Temporal Chunk Attention) processes 10,000-length behavior
sequences end-to-end on Douyin, while Kuaishou's
**[TWIN V2](https://arxiv.org/abs/2302.02352)** handles 100,000+ historical
behaviors through divide-and-conquer retrieval. Both systems serve 400-600M
daily active users with acceptable latency.

**Multi-task optimization** advanced significantly. Kuaishou's
**[GradCraft](https://arxiv.org/abs/2407.19682)** provides theoretical
guarantees for global gradient direction alignment across all tasks
simultaneously—not just pairwise conflict resolution. Meta's
**[MultiBalance](https://arxiv.org/abs/2411.11871)** achieves efficient gradient
balancing on representation layers (rather than parameters) with **zero QPS
degradation**, enabling production deployment on ads and feeds systems.

**Hardware-ML co-design** enables order-of-magnitude gains. Meta's
**[Andromeda](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)**
retrieval engine, co-designed with NVIDIA Grace Hopper Superchip, achieved
**10,000x model capacity increase**, **3x QPS improvement**, and **100x feature
extraction speedup** for ads retrieval serving billions of daily requests.

---

## Architectural innovations: From feature engineering to sequence learning

### Two-tower models evolve beyond their limitations

The fundamental constraint of two-tower architectures—that user and item
representations must be computed independently—drove significant 2024-2025
innovation. **[ContextGNN](https://arxiv.org/abs/2411.19513)** proposes a hybrid
approach: pair-wise GNN representations for familiar user-item pairs combined
with two-tower representations for exploratory items. A learned fusion network
determines which representation to use based on user preference patterns.
Results show **20% improvement over best pair-wise baselines** and **344% over
best two-tower baselines**.

**[FIT](https://arxiv.org/abs/2509.12948)** (Fully Interacted Two-Tower)
introduces learnable interactions while preserving inference efficiency. The
Meta Query Module creates a learnable item meta-matrix enabling early user-item
interaction, while the Lightweight Similarity Scorer processes the full
similarity matrix between towers using small FC layers. This addresses the
expressiveness limitations of dot-product similarity while maintaining two-tower
deployment characteristics.

[Pinterest's production system](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e)
demonstrates **multi-embedding retrieval** using Capsule Networks—generating
multiple user embeddings to capture diverse intents (a user seeking recipe
inspiration differs from one planning home renovation). Combined with
**conditional retrieval** that filters candidates based on predicted user
interest categories, this approach delivered **>1% improvement in saves and
clicks** on their 10-billion-daily-recommendation system.

### Feature interaction layers see targeted improvements

DCN (Deep Cross Network) continues evolving. **GDCN** (Gated Deep Cross Network)
adds information gates that dynamically filter feature interactions at each
order, delaying the performance plateau to depth 8-10 versus DCN-V2's early
saturation. The field-level dimension optimization component compresses models
from ~20M to ~5M parameters with comparable accuracy.
**[DCNv3](https://arxiv.org/abs/2311.04635)** introduces Tri-BCE loss providing
direct supervision to different sub-networks, achieving state-of-the-art using
only explicit feature interactions—eliminating the need for implicit DNN
components and improving interpretability.

Pinterest's production system uses **MaskNet** for bitwise feature crossing and
**DHEN** (Deep Hierarchical Embedding Network) with transformers for field-wise
crossing, delivering **0.15-0.35% engaged sessions improvement** at scale.

### Multi-task learning architectures mature for production

The field has moved beyond academic exploration to production-hardened
solutions. Kuaishou's **[GradCraft](https://arxiv.org/abs/2407.19682)**
addresses a critical limitation: existing multi-task methods resolve gradient
conflicts pairwise, potentially creating new conflicts when combined. GradCraft
provides **theoretical guarantees for global direction balance** across all
tasks simultaneously through projection-based elimination. Deployed on tasks
including EffectiveView, LongView, Like, Follow, and Forward at Kuaishou.

Meta's **[MultiBalance](https://arxiv.org/abs/2411.11871)** solves the
production-deployment gap in multi-objective optimization. Prior methods (MGDA,
MoCo) cause 70-80% QPS degradation—unacceptable for production ads systems.
MultiBalance achieves **0.738% Normalized Entropy improvement with neutral
training cost** by balancing shared representation gradients rather than shared
parameter gradients, using moving-average gradient computation for stability.

Tencent's **PLE** (Progressive Layered Extraction) remains foundational, with
its progressive separation of task-specific and shared experts widely
referenced.
[Netflix demonstrated multi-task consolidation benefits](https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a)—combining
user-to-item, item-to-item, and query-to-item recommendations into a single
model improved performance through knowledge transfer while simplifying system
architecture.

---

## Sequential modeling: From thousands to hundreds of thousands

### Ultra-long sequence architectures emerge

The sequence length race intensified dramatically. Traditional systems truncate
user histories to hundreds of items; 2024-2025 pushed this to tens of thousands.

ByteDance's **STCA** (Spatial-Temporal Chunk Attention) enables **10,000-length
sequence modeling** end-to-end on Douyin at production latency. The key
innovation is efficient chunking strategies for attention computation combined
with memory-optimized training/inference pipelines. **LONGER** extends this
further, with hierarchical attention mechanisms and progressive sequence
compression handling sequences scaling from thousands to tens of thousands.

Kuaishou's **[TWIN V2](https://arxiv.org/abs/2302.02352)** captures **100,000+
historical behaviors** through a two-stage interest network with target-aware
attention. The divide-and-conquer retrieval strategy efficiently indexes
long-term behaviors, enabling lifelong user modeling at the 400M+ DAU scale.
**[KuaiFormer](https://arxiv.org/abs/2411.10057)** moves retrieval from score
estimation to Next Action Prediction, achieving marked increases in average
daily usage time.

Alibaba's **ETA** (Efficient Target-Aware Attention) uses locality-sensitive
hashing for sub-linear complexity attention, providing foundation for subsequent
long-sequence work. The end-to-end trainable hash functions enable practical
deployment of lifelong behavior modeling.

### Attention mechanisms adapt for recommendation data

Standard Transformer attention is problematic for recommendation's
high-cardinality, non-stationary data. HSTU's **pointwise normalization**
(replacing softmax) handles non-stationary vocabularies inherent in streaming
recommendation data. **LinRec** achieves linear complexity O(Nd) through
L2-normalized attention with changed dot-product order, delivering **7.1-28%
improvement** while enabling practical long-sequence modeling.

[Google's YouTube Music transformer](https://research.google/blog/transformers-in-music-recommendation/)
demonstrates production application: self-attention over user action sequences
enables context-dependent preference understanding (gym versus commute listening
patterns), with actions encoded as intention, salience, metadata, and track
embeddings. Results showed significant skip-rate reduction and increased session
time.

---

## Generative recommendation: The paradigm shift

### Generative recommenders prove viable at scale

The most significant architectural shift is from discriminative (score
prediction) to generative (item sequence generation) models. This isn't
speculative—multiple trillion-parameter systems now run in production.

Meta's **[HSTU](https://arxiv.org/abs/2402.17152)** reformulates recommendation
as sequential transduction. User actions become tokens; next-item prediction
becomes next-token generation. The M-FALCON algorithm enables micro-batched
inference with 10x-1000x acceleration. Production deployment on Meta surfaces
achieved **12.4% topline metric improvement** with 1.5 trillion parameters
serving billions of users.
[Open-source code available on GitHub](https://github.com/meta-recsys/generative-recommenders).

Kuaishou's **[OneRec](https://arxiv.org/abs/2502.11583)** demonstrates
end-to-end generative retrieval AND ranking in a single model. The architecture
uses 4-pathway encoding (user static features, short-term behavior, positive
feedback history, lifelong history) with T5-style encoder-decoder. Session-wise
list generation produces interdependent video recommendations rather than
independent scoring. **Iterative Preference Alignment** using DPO variants
optimizes for multiple business objectives.

Xiaohongshu's **RankGPT** modifies HSTU with action-oriented organization that
reduces sequence length by 2x and achieves **94.8% speed-up** with 0.06% AUC
gain. Alibaba's **GPSD** pretrains ID embeddings in a generative model, then
transfers to discriminative CTR models—breaking the "one-epoch phenomenon" that
limits DLRM training and enabling scaling laws to emerge.

### Semantic IDs enable new capabilities

Google's **[TIGER](https://arxiv.org/abs/2305.05065)** introduces semantic IDs
using RQ-VAE (Residual-Quantized VAE): items receive hierarchical token
sequences based on content similarity rather than random hash IDs. This enables:

- **Cold-start generalization**: new items share tokens with similar existing
  items
- **Compositionality**: the system can reason about item relationships
- **Reduced vocabulary**: hierarchical tokens compress the ID space

Meta's **[Semantic ID](https://arxiv.org/abs/2504.02137)** work extends this
concept for ads, using hierarchical clusters from item content similarity to
create stable item representations. This addresses embedding instability,
impression skew, and cold-start challenges.

OneRec uses **RQ-Kmeans** for semantic IDs, incorporating collaborative signals
(not just content) and maximizing codebook utilization. The constrained
generation via Trie-based beam search prevents hallucinated IDs—a critical
production requirement.

### LLM integration patterns crystallize

Three distinct LLM integration approaches emerged:

**Hierarchical architecture**: ByteDance's
**[HLLM](https://arxiv.org/abs/2409.12740)** separates Item LLM (up to 7B
parameters extracting features from text descriptions) from User LLM (predicting
future interactions from item embeddings). This achieves state-of-the-art on
benchmarks with excellent cold-start performance while requiring only 5 epochs
versus 50-200 for traditional models.
[Code and model weights available on GitHub](https://github.com/bytedance/HLLM).

**Knowledge augmentation**: Huawei's **KAR** uses LLMs to generate factual and
reasoning knowledge about items, then distills this into efficient
recommendation models. By preprocessing and prestoring LLM outputs, inference
latency is unaffected. Results: **+7% improvement on news platform**, **+1.7% on
music platform** in online A/B tests.

**Feature enhancement**:
**[LLM-ESR](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html)**
(NeurIPS 2024) uses LLM semantic embeddings to enhance sequential recommenders
for long-tail users/items through dual-view modeling and retrieval-augmented
self-distillation—with zero extra inference overhead.

### Diffusion models find specific niches

While less dominant than LLM-based approaches, diffusion models show promise for
specific applications.
**[DCDR](https://dl.acm.org/doi/10.1145/3589335.3648313)** (Discrete Conditional
Diffusion Reranking) extends diffusion to discrete space for item sequence
generation, deployed at a 300M+ DAU video app. **DiffMM** applies multi-modal
graph diffusion for modality-aware recommendations.

The key limitation remains iterative inference cost. Current production
deployments use diffusion for reranking rather than retrieval, where the smaller
candidate set makes iterative generation tractable.

---

## Production systems: Efficiency at billion-user scale

### Hardware-ML co-design enables order-of-magnitude gains

Meta's
**[Andromeda](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)**
retrieval engine demonstrates the potential of hardware-ML co-design. Built
around NVIDIA Grace Hopper Superchip:

- **10,000x model capacity increase** through hierarchical indexing enabling
  sub-linear inference
- **3x QPS improvement** via model elasticity for dynamic resource allocation
- **100x feature extraction latency improvement** through GPU preprocessing
- **+6% recall improvement**, **+8% ads quality improvement**, **+22% ROAS
  increase**

The system processes billions of daily requests across Instagram and Facebook,
handling tens of millions of ad candidates per request while meeting tight
latency constraints. Meta's Training and Inference Accelerator (MTIA) provides
additional custom silicon optimization.

### Embedding systems address scalability barriers

ByteDance's **[Monolith](https://arxiv.org/abs/2209.07663)** pioneered
collisionless embedding tables using Cuckoo HashMap, enabling online training
with minute-level sparse parameter updates. The system powers TikTok's
recommendations with memory-efficient design and fault-tolerant periodic
snapshotting. Code is open-source.

Tsinghua/Tencent's
**[multi-embedding design](https://arxiv.org/abs/2310.04400)** with
embedding-set-specific interaction modules addresses the embedding collapse
problem that prevents naive scaling. This is now deployed at Tencent's
advertising platform.

Pinterest's **ID embedding pre-training** uses contrastive learning to create
high-quality embeddings before recommendation model training, delivering
**0.6-1.2% repins/clicks increase**.

### Infrastructure evolution supports model proliferation

[Meta's engineering team documented the journey to **1,000+ recommendation models**](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/)
across Instagram. Key infrastructure includes:

- Model registry with core and extended metadata attributes
- Automated model launch tooling (reduced deployment from days to hours)
- Model stability metrics and SLOs
- Entity-level personalization

The
**[Generative Ads Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)**
demonstrates LLM-scale training for ads: **23x increase in effective training
FLOPs**, **1.43x improvement in hardware efficiency**, near-linear scaling to
thousands of GPUs. Post-training techniques propagate knowledge across surfaces.

---

## Trend analysis: Convergence and divergence across organizations

### Converging trends (multiple organizations adopting)

**Scaling laws are achievable for recommendations**. Meta
([Wukong](https://arxiv.org/abs/2403.02545),
[HSTU](https://arxiv.org/abs/2402.17152)), Netflix, Pinterest (PinFM), and
academic studies now confirm that recommendation models can follow LLM-like
scaling laws when properly designed. The critical insight: generative training
(not discriminative) is essential for scaling laws to emerge.

**Long-sequence modeling is essential**. Every major platform (ByteDance,
Kuaishou, Alibaba, Meta) has invested heavily in extending sequence length from
hundreds to thousands or tens of thousands. Techniques vary (chunked attention,
hash-based retrieval, hierarchical compression) but the direction is universal.

**LLM knowledge improves recommendations**. Whether through hierarchical
architectures (ByteDance [HLLM](https://arxiv.org/abs/2409.12740)), knowledge
distillation (Huawei KAR), or foundation model approaches (Netflix, Pinterest),
LLM integration is now standard. The focus is avoiding LLM inference
latency—preprocessing and caching strategies dominate.

**Multi-task gradient management is production-critical**. Both Meta
([MultiBalance](https://arxiv.org/abs/2411.11871)) and Kuaishou
([GradCraft](https://arxiv.org/abs/2407.19682)) published significant work on
gradient balancing in 2024, emphasizing production efficiency constraints that
earlier academic work ignored.

### Diverging approaches (where organizations disagree)

**Generative vs. hybrid architectures**. Meta and Kuaishou commit to end-to-end
generative recommenders. Alibaba (GPSD), Pinterest (PinFM), and others pursue
hybrid approaches—generative pretraining followed by discriminative deployment.
The hybrid approach offers lower infrastructure disruption and risk.

**Semantic ID construction**. Google uses RQ-VAE (content-based). Kuaishou uses
RQ-Kmeans (incorporating collaborative signals). Meta explores hierarchical
clustering from content similarity. Each approach has different cold-start
versus personalization tradeoffs.

**Attention architecture choices**. HSTU uses pointwise normalization for
non-stationary vocabularies. LinRec uses L2-normalized linear attention for
computational efficiency. Standard Transformer attention remains common with
various compression strategies. No clear winner has emerged.

### Unique organizational approaches

| Organization  | Distinctive Focus                                                                                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Meta**      | Trillion-parameter generative models; hardware co-design (MTIA, Grace Hopper); scaling laws                                                                                                                                    |
| **ByteDance** | Online training infrastructure ([Monolith](https://arxiv.org/abs/2209.07663)); ultra-long sequences (STCA 10k); real-time adaptation                                                                                           |
| **Kuaishou**  | End-to-end generative retrieval+ranking ([OneRec](https://arxiv.org/abs/2502.11583)); lifelong modeling (TWIN V2 100k+); live-streaming                                                                                        |
| **Alibaba**   | Hybrid generative-discriminative (GPSD); trigger-induced recommendations; e-commerce search                                                                                                                                    |
| **Tencent**   | Multi-task learning (PLE); social graph integration; [embedding collapse solutions](https://arxiv.org/abs/2310.04400)                                                                                                          |
| **Netflix**   | Long-term satisfaction optimization; [foundation models for streaming](https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a); reward engineering |
| **Pinterest** | [Multi-embedding retrieval](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e); visual understanding; Capsule Networks for intent diversity                |
| **Google**    | Semantic IDs ([TIGER](https://arxiv.org/abs/2305.05065)); [transformer-based music recommendation](https://research.google/blog/transformers-in-music-recommendation/); generative retrieval research                          |

---

## Underexplored areas and research opportunities

### Promising directions with limited investigation

**Real-time preference alignment** for generative recommenders remains nascent.
OneRec's IPA demonstrates potential, but methods for continuously updating
preference models with streaming feedback need development. Current approaches
require periodic offline DPO/RLHF training.

**Cross-domain generative transfer** is underexplored. While LLMs demonstrate
cross-task transfer, applying this to multi-domain recommendations (e.g.,
video→e-commerce→music) lacks systematic study. Semantic IDs potentially enable
this through shared token vocabularies.

**Multimodal generative recommendations** beyond text remain early-stage. DiffMM
and related work show potential, but production systems primarily use multimodal
features for encoding rather than generation. Generating personalized visual
content for recommendations is nascent.

**Efficiency-focused semantic ID research** is needed. Current RQ-VAE approaches
require significant compute for ID assignment. Methods for efficient incremental
ID updates as catalogs change are missing.

**Evaluation paradigms for generative recommendations** lag behind model
capabilities. Traditional metrics (AUC, NDCG) may not capture benefits of
session-wise list generation, diversity, or long-term satisfaction. New
evaluation frameworks are needed.

### Research gap analysis for novel contributions

| Gap                                      | Current State                                | Opportunity                                |
| ---------------------------------------- | -------------------------------------------- | ------------------------------------------ |
| **Efficient semantic ID updates**        | Static ID assignment requires full recompute | Incremental methods for streaming catalogs |
| **Multi-objective generative alignment** | Binary or few-objective DPO                  | Scaling to dozens of business objectives   |
| **Cross-domain semantic tokens**         | Domain-specific ID spaces                    | Universal item tokenization                |
| **Real-time preference learning**        | Offline alignment training                   | Online RLHF for recommendations            |
| **Generative evaluation**                | Point prediction metrics                     | Session-level, diversity-aware metrics     |
| **Sparse generative models**             | Dense trillion-parameter models              | MoE and sparse attention for GR            |

---

## Recommended reading list (ranked by impact and relevance)

### Tier 1: Essential reading for practitioners

1. **[HSTU/Generative Recommenders](https://arxiv.org/abs/2402.17152)** (Meta,
   ICML 2024) — Production-proven trillion-parameter architecture with scaling
   laws.
   [Open-source code available](https://github.com/meta-recsys/generative-recommenders).

2. **[Wukong](https://arxiv.org/abs/2403.02545)** (Meta, ICML 2024) — First
   demonstration of recommendation scaling laws using stacked factorization
   machines.

3. **[OneRec](https://arxiv.org/abs/2502.11583)** (Kuaishou, 2025) — End-to-end
   generative retrieval and ranking with preference alignment.
   Production-deployed.

4. **[GradCraft](https://arxiv.org/abs/2407.19682)** (Kuaishou, KDD 2024) —
   Multi-task gradient balancing with theoretical guarantees. Open-source.

5. **[Embedding Collapse](https://arxiv.org/abs/2310.04400)** (Tsinghua/Tencent,
   ICML 2024) — Critical insight for model scaling with production solution.

### Tier 2: Important for system design

6. **[HLLM](https://arxiv.org/abs/2409.12740)** (ByteDance, 2024) — Hierarchical
   LLM architecture for sequential recommendations.
   [Code and weights available](https://github.com/bytedance/HLLM).

7. **[MultiBalance](https://arxiv.org/abs/2411.11871)** (Meta, 2024) —
   Production-efficient multi-objective gradient balancing.

8. **[TIGER](https://arxiv.org/abs/2305.05065)** (Google, NeurIPS 2023/2024) —
   Foundational semantic ID work enabling generative retrieval.

9. **[TWIN V2](https://arxiv.org/abs/2302.02352)** (Kuaishou, CIKM 2024) —
   Ultra-long sequence modeling at scale.

10. **[DCNv3](https://arxiv.org/abs/2311.04635)** — Latest feature interaction
    layer advances with interpretability benefits.

### Tier 3: Specialized topics

11. **[Pinterest Embedding-Based Retrieval](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e)**
    (2025) — Production multi-embedding and conditional retrieval.

12. **[Netflix Foundation Model](https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a)**
    (2024-2025) — Streaming-specific foundation model design.

13. **[KuaiFormer](https://arxiv.org/abs/2411.10057)** (Kuaishou, 2024) —
    Transformer-based retrieval for short-video recommendations.

14. **[Meta Andromeda](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)**
    (2024) — Hardware-ML co-design for retrieval.

15. **[LLM-ESR](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html)**
    (NeurIPS 2024) — LLM enhancement for long-tail sequential recommendation.

### Additional Resources

- **[Meta GEM (Generative Ads Model)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)**
  — LLM-scale training for ads
- **[Instagram 1000 Models Journey](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/)**
  — Infrastructure for model proliferation
- **[Google Music Recommendations](https://research.google/blog/transformers-in-music-recommendation/)**
  — Transformer application in production
- **[ByteDance Monolith](https://arxiv.org/abs/2209.07663)** — Real-time
  recommendation system with collisionless embeddings
- **[ContextGNN](https://arxiv.org/abs/2411.19513)** — Beyond two-tower
  architectures
- **[FIT Two-Tower](https://arxiv.org/abs/2509.12948)** — Learnable fully
  interacted two-tower model
- **[Semantic ID for Stability](https://arxiv.org/abs/2504.02137)** — Embedding
  stability through semantic IDs
- **[DCDR Diffusion Reranking](https://dl.acm.org/doi/10.1145/3589335.3648313)**
  — Discrete diffusion for recommendation

---

## Conclusion: A field in transformation

The 2024-2025 period marks a fundamental shift in recommendation system
architecture. **Scaling laws**, long considered unachievable in recommendations
due to sparse, heterogeneous data, are now empirically demonstrated across
multiple organizations. The key insight: generative training objectives
(next-item prediction) enable scaling where discriminative objectives (CTR
prediction) plateau.

**Generative recommenders** have moved from research curiosity to production
reality. Meta, Kuaishou, Xiaohongshu, and others run billion-user systems built
on sequence-to-sequence architectures that treat user behavior as language. The
unification of retrieval and ranking into single generative models eliminates
stage-specific optimization and enables session-aware list generation.

**Hybrid approaches** offer a pragmatic middle path: generative pretraining
creates high-quality embeddings that transfer to discriminative deployment,
preserving existing infrastructure while capturing scaling benefits. This
pattern appears consistently across Alibaba, Pinterest, Netflix, and others.

**Multi-task learning** has matured from a research problem to production
necessity. The field now understands how to balance dozens of objectives without
catastrophic QPS degradation, enabling sophisticated multi-objective
optimization at scale.

The remaining challenges are significant: inference latency for autoregressive
generation, semantic ID collision handling, preference alignment for many
objectives, and evaluation frameworks that capture generative model benefits.
But the direction is clear—the next generation of recommendation systems will be
generative, scaled, and sequence-native.

For practitioners building large-scale multi-task recommendation systems, the
priority is clear: invest in sequence modeling infrastructure capable of
handling thousands of historical interactions, experiment with hybrid
generative-discriminative approaches to capture scaling benefits with manageable
risk, and implement production-ready gradient balancing for multi-objective
optimization. The trillion-parameter era of recommendations has arrived.

---

## Quick Reference: All Paper Links

### Core Papers

| Paper                          | Link                                                                                                                 |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| HSTU (Generative Recommenders) | https://arxiv.org/abs/2402.17152                                                                                     |
| Wukong (Scaling Laws)          | https://arxiv.org/abs/2403.02545                                                                                     |
| OneRec                         | https://arxiv.org/abs/2502.11583                                                                                     |
| TIGER                          | https://arxiv.org/abs/2305.05065                                                                                     |
| GradCraft                      | https://arxiv.org/abs/2407.19682                                                                                     |
| MultiBalance                   | https://arxiv.org/abs/2411.11871                                                                                     |
| Embedding Collapse             | https://arxiv.org/abs/2310.04400                                                                                     |
| HLLM                           | https://arxiv.org/abs/2409.12740                                                                                     |
| DCNv3                          | https://arxiv.org/abs/2311.04635                                                                                     |
| TWIN V2                        | https://arxiv.org/abs/2302.02352                                                                                     |
| KuaiFormer                     | https://arxiv.org/abs/2411.10057                                                                                     |
| ContextGNN                     | https://arxiv.org/abs/2411.19513                                                                                     |
| FIT Two-Tower                  | https://arxiv.org/abs/2509.12948                                                                                     |
| Semantic ID                    | https://arxiv.org/abs/2504.02137                                                                                     |
| Monolith                       | https://arxiv.org/abs/2209.07663                                                                                     |
| LLM-ESR                        | https://proceedings.neurips.cc/paper_files/paper/2024/hash/2f0728449cb3150189d765fc87afc913-Abstract-Conference.html |
| DCDR                           | https://dl.acm.org/doi/10.1145/3589335.3648313                                                                       |

### Industry Blog Posts

| Source                        | Link                                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Meta Andromeda                | https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/          |
| Meta GEM                      | https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/ |
| Instagram 1000 Models         | https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/                         |
| Pinterest Embedding Retrieval | https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e                                 |
| Netflix Multi-Task            | https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a                    |
| Google Music Transformers     | https://research.google/blog/transformers-in-music-recommendation/                                                                                    |

### Code Repositories

| Repo                         | Link                                                   |
| ---------------------------- | ------------------------------------------------------ |
| HSTU/Generative Recommenders | https://github.com/meta-recsys/generative-recommenders |
| HLLM                         | https://github.com/bytedance/HLLM                      |
