Here is your report converted to Markdown (.md) format. You can copy-paste this
directly into any Markdown editor or repository.

---

# ID-Based Recommendation/Ranking Modeling Architecture Development at Meta: 2024-2025 Comprehensive Analysis

## TLDR

Meta's recommendation and ranking systems underwent fundamental transformation
during 2024-2025, marked by five critical innovations:

- **Foundation Model Scale-Up**: RankFM evolved from 500B to 25T parameters,
  with HSTU architectures delivering 200-300% capacity savings and significant
  production wins across ads ranking
- **Semantic Tokenization Breakthrough**: RQ-VAE and RQ-KMeans systems achieved
  682× compression ratios while enabling unified vocabularies, driving major
  wins across IG (+0.11% teen engagement), FB (+0.04% sessions), and Ads
  platforms
- **Advanced Sequence Modeling**: Transformer-based architectures (HSTU, LLaTTE,
  CoFormer) scaled user behavior prediction from 90 days to 18+ months of
  history, achieving 2.55× higher scaling efficiency and 2-4% conversion
  improvements
- **Custom Hardware Revolution**: MTIA silicon program delivered 2.3×
  performance/CapEx improvements over Nvidia GPUs, serving ~5% of Meta's ads
  revenue with $125M+ savings through four hardware generations
- **Cross-Platform Unification**: Knowledge distillation and unified models
  (Atlas, POLARIS) achieved 20-45% transfer ratios, expanding coverage 3× across
  FB/IG/Ads with 0.1-1.7% GAS improvements

These advances represent a paradigm shift toward LLM-scale, semantically-aware,
hardware-optimized recommendation systems delivering billions in revenue impact.

---

## Foundation Models and LLM-Scale Recommendations

Meta achieved breakthrough scale in foundation models with **RankFM's evolution
across four major versions**.
[RankFM V1 deployed 500B parameters with 90-day user history](https://docs.google.com/document/d/18ADCYnv6ytuzz54CHL-M9Jxus-j8nBgd7xyy62ZNPkY),
scaling to **V4's planned 25T parameters** handling 18+ months of behavioral
data.

**HSTU (Hierarchically Stacked Transformer Units)** emerged as the dominant
architecture, inspired by LLaMA/GPT designs with autoregressive attention and
relative position bias.
[Production deployments demonstrated remarkable efficiency gains](https://fb.workplace.com/groups/1033540429995021/permalink/9904321329583509/):
**200-300% capacity savings** through GPU training node reductions (16→8 nodes)
and inference QPS improvements (2.7× speedup).

[GEM (Generative Ads Recommendation Model) delivered 5%+ conversion improvements](https://fb.workplace.com/groups/256834280626682/permalink/739481079028664/)
across multiple ad surfaces. **Hardware acceleration on AMD MI300x and Nvidia
B200/GB200** achieved near-parity or better performance than H100.

The **FM/Expert paradigm** decoupled foundation models from product experts,
enabling faster iteration cycles.
[Knowledge transfer techniques like StudentAdapter improved transfer ratios](https://fb.workplace.com/groups/1033540429995021/permalink/24239587108963692/)
from FM to downstream models, with **UserFM+RankFM-nano** building cross-surface
user representation libraries.

---

## Generative Retrieval and Semantic ID Systems

**Semantic ID generation revolutionized content representation** through two
dominant approaches: RQ-VAE and RQ-KMeans.
[RQ-VAE achieved compression ratios up to 682× for 1024-dimensional embeddings compressed to 24 bits](https://docs.google.com/document/d/1Yw2IixI0J2p3Icdl9wH6w3HV3kB9ZHbMcgPfyJYwgQY)
while maintaining reconstruction quality.

**Hierarchical codebook structures** enabled coarse-to-fine semantic
representation with vocabulary sizes up to 8×4096 (1200T tokens).
[Production deployments achieved significant wins](https://fb.workplace.com/groups/1033540429995021/permalink/9751223081560002/):
**IG Reels +0.11% teen TS and +0.22% reshare**, **IG OneFeed +0.04% sessions**,
and **FB Reels +0.77% messenger DAU**.

[OVIS platform now supports RQ-KMeans for semantic ID generation](https://fb.workplace.com/groups/vector.search.faiss.fyi/permalink/2339684333149964/),
providing distributed clustering with GPU acceleration and warm-start
capabilities. **SentencePiece tokenization outperformed heuristic n-grams** for
semantic coverage.

[Generative retrieval models like RecGPT](https://fb.workplace.com/groups/1033540429995021/permalink/9961138997235075/)
demonstrated +0.20% SS GAS over 14 days through hierarchical cluster prediction.
**Unified vocabularies** enabled content understanding across modalities, with
semantic IDs solving cold-start challenges while **maintaining 90%+ efficiency**
compared to traditional embedding-based approaches.

---

## Sequence Learning and User Behavior Modeling

**Event-based learning paradigms** replaced traditional feature engineering,
with
[Meta's ads system achieving 2-4% more conversions](https://fb.workplace.com/groups/1063021180410554/permalink/8655247307854532/)
through direct learning from engagement and conversion events rather than
hand-crafted features.

**HSTU architectures scaled dramatically**, from modeling 512 recent
interactions to
[handling sequence lengths up to 16K tokens representing 18+ months of user history](https://fb.workplace.com/groups/3583444701719170/permalink/8660911837305739/).
[Advanced techniques like semi-local attention, FP8 mixed precision training, and 2-stage cached inference](https://docs.google.com/document/d/18ADCYnv6ytuzz54CHL-M9Jxus-j8nBgd7xyy62ZNPkY)
enabled this scaling.

[**LLaTTE (LLM Style Latent Transformers)** demonstrated 2.55× higher scaling efficiency](https://fb.workplace.com/groups/1033540429995021/permalink/24952822034306859/)
than traditional architectures, achieving 0.55% NE gain at 380X scale.
**CoFormer introduced convolution-based multi-resolutional learning**, capturing
both local and global patterns in user sequences.

[Production deployments showed remarkable wins](https://fb.workplace.com/groups/353618119088178/permalink/1409062816877031/):
**IG Feed HSTU achieved +0.094% TS and +0.056% Teen Cap15 sessions**.
**Stochastic length mechanisms** and **positional sampling** enabled efficient
training on billion-scale datasets while maintaining model performance across
different sequence lengths.

---

## AI Infrastructure and Model Scaling

**MTIA (Meta Training and Inference Accelerator)** evolved through four
generations, delivering unprecedented hardware-software co-design.
[Artemis (production chip) achieved 3× per-chip compute performance and 1.5× performance per watt](https://fb.workplace.com/groups/fyi/permalink/7617808634942897/)
over previous versions.

[**Production deployments demonstrated 2.3× performance/CapEx** relative to Nvidia GPUs](https://fb.workplace.com/groups/527654686243695/permalink/749566440719184/),
with MTIA serving **~5% of Meta's ads revenue** and delivering **$125M+
savings**. The 8×8 grid of processing elements with 256MB on-chip SRAM provides
optimal balance for recommendation workloads.

**Future roadmap includes Olympus (MTIA v2)** with
[CUDA-compatible SIMT architecture enabling seamless GPU-to-MTIA migration](https://fb.workplace.com/groups/1607831063005391/permalink/1942847902837037/).
**3D packaging technology** will stack compute and memory vertically for higher
density.

[**Model consolidation efforts** achieved remarkable efficiency gains](https://fb.workplace.com/groups/1033540429995021/permalink/9639157779433200/):
**Interformer on MTIA delivered 0.1% NE gain with on-par inference QPS** through
hardware-model co-design. **GPU scaling strategies** included distributed
training optimizations, with some models reducing from 16 to 8 training nodes
while maintaining performance.

**Advanced compiler optimizations** using open-source Triton language and
**mixed-precision training** enabled efficient deployment across Meta's global
infrastructure.

---

## Cross-Domain Learning and Knowledge Transfer

**Knowledge distillation achieved breakthrough transfer ratios**, with
[current implementations showing 20-25% transfer from foundation models to production models](https://fb.workplace.com/groups/1033540429995021/permalink/9494749160540730/),
targeting improvements to **45%+ through LoopFM innovations**.

[**Atlas unified user representations** consolidated 24 fragmented SUM models into a single universal model](https://fb.workplace.com/groups/1033540429995021/permalink/9775471412468502/),
achieving **0.06% AF-CMF, 0.08% AF-OC, and 0.08% IG-CTR NE gains** with 0.43%+
GAS headroom. **27× reduction in embedding size** through HiVE Vector
Quantization maintained 100% NE parity.

[**Hierarchical POLARIS** expanded GEM coverage from 25% to 42% eCPM revenue](https://fb.workplace.com/groups/929822892542773/permalink/1089086583283069/),
enabling **lossless and costless knowledge transfer** through user-ad embeddings
rather than single prediction values.

**Multi-surface learning** enabled unified models across Facebook, Instagram,
and Ads platforms.
[Domain adaptation techniques included Domain Specific Batch Normalization (DSBN), pairwise learning, and submodel Mixture of Experts](https://www.internalfb.com/wiki/Signal_Growth_Ranking_and_Performance/SUMO%2B/SGAI4P_CDML/CDMLSAIL%3A_Proven_CDML_Methods_for_2025H1_R%26D_-_Self-serving_Guideline_for_SAIL_Product_Teams/).

**Production launches demonstrated substantial impact**:
[Marketplace knowledge distillation achieved 0.09% session gains](https://fb.workplace.com/groups/1115642875273221/permalink/2921980934639397/),
while cross-domain models showed **1.5-2× quality improvements** over
traditional domain-specific iterations.

---

## Conclusion

Meta's 2024-2025 transformation in recommendation/ranking architecture
represents a paradigm shift toward **intelligent, unified, and
hardware-optimized systems**. The convergence of LLM-scale foundation models,
semantic tokenization, advanced sequence learning, custom silicon, and
cross-platform unification has delivered unprecedented business value.

**Key Strategic Achievements:**

| **Domain**             | **Innovation**                         | **Business Impact**                                   |
| ---------------------- | -------------------------------------- | ----------------------------------------------------- |
| **Foundation Models**  | RankFM 25T parameters, HSTU scaling    | 200-300% capacity savings, 2-4% conversion gains      |
| **Semantic Systems**   | 682× compression, unified vocabularies | +0.11% teen engagement, cross-platform generalization |
| **Sequence Learning**  | 18+ month behavioral modeling          | 2.55× scaling efficiency, LLM-style architectures     |
| **Custom Hardware**    | MTIA 4-generation evolution            | 2.3× performance/CapEx, $125M+ savings                |
| **Knowledge Transfer** | 45% transfer ratios, unified models    | 0.1-1.7% GAS, 3× coverage expansion                   |

This comprehensive modernization positions Meta's recommendation systems as
industry-leading platforms capable of **delivering personalized experiences at
planetary scale** while maintaining **exceptional efficiency and business ROI**.
The foundation established in 2024-2025 enables continued innovation toward even
more sophisticated AI-driven personalization capabilities.

---

Let me know if you need this as a downloadable file or want further formatting
tweaks!

# Meta Recommendation System Breakthroughs (2024-2025)

Meta has achieved breakthrough advances in recommendation system architecture
across five critical dimensions:

- **Foundation Models**: RankFM evolved from 500B to 25T parameters with HSTU
  architecture achieving 12.4% A/B test improvements and power-law scaling
- **Multi-Task Consolidation**: MTML frameworks enabling 60+ model consolidation
  with 2.4% GAS improvements and $292M daily revenue impact
- **Generative Systems**: OneRec-inspired paradigms with semantic IDs achieving
  10x computational efficiency and end-to-end optimization
- **Hardware Co-design**: MTIA program delivering 1.94x-2.55x better cost
  efficiency than H100 with $125M+ savings across custom silicon generations
- **Sequential Learning**: Advanced temporal modeling processing 100K+ video
  sequences with real-time personalization and causal inference

These innovations collectively represent a paradigm shift from traditional DLRM
approaches to unified, scalable, AI-native recommendation systems serving
billions of users with measurable business impact.

---

## Executive Summary

Meta's recommendation and ranking systems have undergone fundamental
architectural transformation during 2024-2025, evolving from traditional Deep
Learning Recommendation Models (DLRMs) to sophisticated foundation model-based
approaches. This comprehensive analysis reveals **five interconnected innovation
tracks** that collectively redefine large-scale recommendation system
capabilities.

The evolution centers on **RankFM's dramatic scaling trajectory**—from 500B
parameters handling 90-day user history to projected 25T parameters processing
18+ months of cross-surface interactions. This scaling follows **power-law
behaviors similar to LLMs**, with the HSTU (Hierarchical Sequential Transduction
Unit) architecture achieving 12.4% production A/B test improvements.

**Multi-task learning frameworks** have enabled unprecedented model
consolidation, with opportunities to merge 60+ models across view-through,
engagement, and conversion products. The **Knowledge Distillation Data
Platform** processes FM-to-VM knowledge transfer with 2.4% GAS improvements
while empowering $292M in daily revenue.

**Generative recommendation paradigms**, inspired by OneRec's end-to-end
approach, leverage **semantic IDs and tokenization** to replace traditional
cascade architectures. Meta's implementations include InstaBrain for Instagram
and RPG for parallel semantic ID generation, achieving significant cold-start
improvements.

**Custom silicon development through MTIA** delivers 1.94x-2.55x better cost
efficiency than NVIDIA H100, with Artemis chips deployed across 16 data centers.
The roadmap extends through Olympus (2027) with CUDA-compatible architecture,
representing **$125M+ in realized savings**.

**Sequential modeling advances** capture temporal user behavior through
sophisticated architectures processing ultra-long sequences (100K+ videos),
real-time personalization, and causal inference for quality optimization,
delivering measurable engagement improvements across Meta's product ecosystem.

---

## Large-Scale Foundation Models and Architecture Innovations

Meta's foundation model strategy centers on **RankFM's aggressive scaling
roadmap** that demonstrates unprecedented growth in recommendation model
capabilities. The evolution trajectory shows clear generational advances:

- **RankFM V1-V4 Progression**: 500B parameters with 90-day user interaction
  history (H2 2023) to projected 25T parameters supporting cross-app generative
  capabilities (H2 2026). Each generation achieves **8x to 10x parameter
  scaling** while extending temporal coverage from months to years.
- **HSTU Architecture Breakthroughs**: Inspired by Llama/GPT architectures but
  optimized for recommendation workloads. Hierarchical sequential transduction
  units with relative positional biases enable **power-law scaling behavior**.
  Production deployments demonstrate **12.4% A/B test improvements** with
  trillion-parameter scale models.
- **HSTU-Ultra Efficiency Achievements**: 2025 H1 delivered performance gains:
  doubling UIH sequence length from 8k to 16k, **1.15x training speedup and 2.7x
  inference speedup**. Optimizations include FlashAttention3 CUDA kernels,
  semi-local attention, FP8 mixed precision training, and 2-stage sequence
  caching.
- **Scaling Law Implementation**: Template-based approaches with FLOPS
  prediction accuracy within 2.5% error. **Zen unified architecture** achieves
  2.3x scaling efficiency by integrating multiple architectural components
  (COFFEE, RecGPT, Wukong, Interformer, PINET).

**Links:**

- [HSTU-Ultra 2025 H1 Lookback](https://fb.workplace.com/groups/228479108798071/permalink/1280765063569465/)
- [RankFM Roadmap](https://docs.google.com/document/d/18ADCYnv6ytuzz54CHL-M9Jxus-j8nBgd7xyy62ZNPkY)

---

## Multi-Task Learning and Model Consolidation

Meta's **Multi-Task Multi-Label (MTML) framework** has become the backbone for
model consolidation across recommendation surfaces, enabling shared
representation learning while maintaining task-specific optimization
capabilities.

- **MTML Architecture Implementation**: Shared feature representations with
  task-specific heads, handling dependencies where one task's output becomes
  input for another. Supports both independent and dependent task architectures,
  enabling conditional probability modeling (e.g., p(Reply|Click) =
  pReply/pClick). Used in Messenger ranking, ads ranking, and cross-surface
  applications.
- **Model Consolidation Opportunities**: **SAC-WS9 framework** identifies merger
  candidates across 60+ models spanning view-through (VT), engaged-view
  conversion (EVC), and click-through (CT) products. **MARCO initiative**
  demonstrates 1% iRev potential with 25MW capacity savings.
- **Knowledge Distillation Scaling**: **KDDP** achieved **2.4% GAS with 30%
  traffic increase**, empowering ~$292M in daily revenue supporting 123 FM
  models. Advanced capabilities include **Polaris sequential evaluation**,
  **LoopFM** for FM-to-VM transfer, and offline EBF generation.
- **Unified Architecture Initiatives**: **RankLLM TriUnity** proposal with
  ItemLLM, UserLLM, Adapters, and Product models. **Cross-surface learning** via
  **UserFM+RankFM-nano** for unified user representation.
- **Multi-Domain Learning**: Handles different traffic types and signal loss
  scenarios through **multi-scenario optimization**.

**Links:**

- [Knowledge Distillation Platform H1](https://fb.workplace.com/groups/1656080455006805/permalink/1740625866552263/)
- [MARCO Click Refinement](https://fb.workplace.com/groups/1033540429995021/permalink/9885888658093443/)

---

## Generative Recommendation Systems and Semantic IDs

Meta's adoption of **generative recommendation paradigms** represents a
fundamental shift from traditional cascade architectures to end-to-end
generative frameworks, with **semantic IDs** serving as the critical bridge
between foundation models and recommendation systems.

- **OneRec-Inspired Architecture Evolution**: Unified encoder-decoder
  frameworks. **OneRec from Kuaishou** achieved 25% traffic deployment with 10x
  computational FLOPs improvement, 23.7% training MFU, and 28.8% inference MFU.
  Processes multi-scale user pathways (static, short-term, positive-feedback,
  lifelong up to 100K videos).
- **Meta's RPG Implementation**: **Parallel semantic ID generation**, generating
  all tokens simultaneously. Optimized product quantization for up to 64 tokens
  with **graph-constrained decoding**.
- **InstaBrain (Instagram)**: **LLM-native recommendation model** understanding
  both human language and IG media language through semantic IDs. Utilizes
  **>99% parameters for computation**.
- **Semantic ID Generation Technologies**: **RQ-VAE** for hierarchical
  clustering and **RankID** as unified semantic ID solution. Applications span
  Instagram Reels cold start, Facebook IFR LSR iterations, and content freshness
  improvements.
- **LIGER Hybrid Framework**: Unifies dense and generative retrieval approaches,
  combining **beam search retrieval** with ranking.

**Links:**

- [OneRec Knowledge Sharing](https://fb.workplace.com/groups/463336019437955/permalink/635136195591269/)
- [InstaBrain Design](https://fb.workplace.com/groups/233247545227463/permalink/1250577616827779/)
- [Semantic IDs Content Freshness](https://www.internalfb.com/wiki/MRS_Value_Alignment/Distribution/Using_Semantic_IDs_to_improve_content_freshness/)

---

## Retrieval Systems and Hardware Co-design

Meta's **hardware co-design strategy** encompasses custom silicon development,
GPU acceleration optimization, and infrastructure co-design, delivering
significant cost efficiency and performance improvements across billion-scale
serving infrastructure.

- **MTIA Evolution**: Four generations with clear performance progression.
  **Artemis (2024)**: 8x8 grid of RISC-V cores, 708 TFLOPS INT8, deployed across
  16 data centers. **Athena (2025)**: training with HBM memory and liquid
  cooling. **Olympus (2027+)**: **CUDA-compatible SIMT architecture**.
- **Performance and Cost Benefits**: **1.94x-2.55x better cost efficiency than
  NVIDIA H100** with **$125M+ in realized savings**. MTIA deployments achieve
  **2.1x Perf/ICE vs NVIDIA** for 7 production models, AMD implementations
  deliver **1.2x Perf/ICE** across 5 models with **17.7M ICE savings**.
- **SilverTorch GPU Retrieval Platform**: Industry's first GPU retrieval
  solution for recommendations, combining models with index, filter, and VM
  artifacts.
- **Meta Andromeda Architecture**: Next-generation retrieval engines through
  **co-design of ML, system, and hardware**.
- **Model-Hardware Co-design Principles**: **SRAM memory optimization**,
  arithmetic intensity maximization, and **memory hierarchy considerations**.
- **Serving Optimization Strategy**: **Globally optimal resource allocation**
  with ROI-based mechanisms, hardware elasticity, and AI-driven automation.

**Links:**

- [MTIA Evolution Overview](https://www.internalfb.com/wiki/Meta%27s_MTIA_Evolution%3A_How_Four_Generations_of_Custom_Chips_are_Reshaping_AI_Computing/)
- [Andromeda Launch](https://fb.workplace.com/groups/1063021180410554/permalink/8730879400291322/)
- [Interformer MTIA Co-design](https://fb.workplace.com/groups/1033540429995021/permalink/9639157779433200/)

---

## Sequential Modeling and User Behavior Learning

Meta's **sequential modeling capabilities** represent sophisticated temporal
pattern learning that processes ultra-long user interaction sequences for
enhanced personalization and behavioral understanding across recommendation
systems.

- **HSTU Sequential Architecture**: Combines hierarchical transduction units
  with relative positional and temporal biases. Scales from 90-day to 18+ month
  user histories, demonstrating **power-law scaling behavior**.
- **Long-Term User History Processing**: Multi-scale feature engineering through
  specialized pathways: user static, short-term, positive feedback, and lifelong
  pathways processing up to 100,000 videos.
- **Temporal Pattern Learning Innovations**: **Deep Causal Sequential Models
  (DCSM)** for quality bid personalization, implementing causal transformers
  with **causal graph informed attention** and Mixture-of-Experts.
- **Time-Sensitive Modeling Systems**: **Long-term Interest Clock (LIC)** for
  fine-grained time perception.
- **Industry-Scale Sequential Applications**: E-commerce journey modeling with
  **Event-Based Supervision (EBS)**, **Real-Time UHM (RT-UHM)**, and
  **NextPredict** for real-time client sequential modeling.
- **Advanced Sequential Frameworks**: **USM-LLM** leveraging LLM-enhanced user
  modeling, **LONGER framework** for ultra-long sequences, and Meta's **sequence
  learning paradigm shift**.

**Links:**

- [End-to-End Long History](https://fb.workplace.com/groups/3583444701719170/permalink/8660911837305739/)
- [DCSM Quality Personalization](https://fb.workplace.com/groups/facebookmonetizationfyi/permalink/30077820525173176/)
- [Sequence Learning Paradigm](https://fb.workplace.com/groups/1063021180410554/permalink/8655247307854532/)

---

## Technical Architecture Integration and Future Roadmap

Meta's comprehensive approach integrates the five innovation tracks into a
**unified technical ecosystem** with significant interdependencies and
synergistic effects.

- **Cross-Domain Architecture Synergies**: Semantic IDs enable generative
  recommendations, hardware co-design supports foundation model scaling, and
  RankFM leverages MTIA acceleration, HSTU sequential processing, and MTML
  consolidation.
- **Infrastructure Convergence Patterns**: SilverTorch GPU retrieval supports
  both traditional and generative paradigms, MTIA custom silicon optimizes for
  both training and inference, and knowledge distillation platforms enable
  efficient deployment.
- **Production Integration Architecture**: End-to-end optimization from data
  ingestion through real-time serving. Neutron Star retrieval maximizes GPU
  predictor latency, Andromeda establishes efficient scaling laws, and serving
  optimization frameworks enable globally optimal resource allocation.
- **Scaling Law Implementations**: Model parameters (RankFM 500B→25T), sequence
  lengths (HSTU 8k→16k→100K), and hardware efficiency (MTIA 1.94x-2.55x cost
  improvement).
- **Future Roadmap Convergence**: Olympus CUDA-compatible architecture (2027+),
  generative foundation models replacing traditional cascades, and cross-app
  semantic understanding through advanced tokenization.

---

## Business Impact and Strategic Value

Meta's recommendation system innovations deliver **quantifiable business
impact** across engagement, revenue, efficiency, and competitive positioning.

- **Revenue and Engagement Metrics**: RankFM scaling achieves 12.4% A/B test
  improvements, MTML consolidation enables $292M daily revenue impact, and
  knowledge distillation platforms support 123 FM models. Sequential learning
  delivers 2-4% more conversions in ads.
- **Cost Efficiency Achievements**: $125M+ realized savings through MTIA
  deployment, 1.94x-2.55x better cost efficiency than NVIDIA H100, AMD
  implementations contribute 17.7M ICE savings, and serving optimizations
  project $45M capacity savings.
- **Operational Excellence Metrics**: HSTU-Ultra efficiency gains, SilverTorch
  GPU retrieval computation improvements, and semantic ID implementations with
  cold-start performance enhancements.
- **Competitive Positioning Advantages**: Industry leader in foundation model
  scaling, custom silicon development, and generative recommendation paradigms.
- **Strategic Value Creation**: Reduced vendor dependence, improved user
  experience, operational efficiency, and future-ready infrastructure.

---

## Implementation Insights and Lessons Learned

Meta's journey toward next-generation recommendation systems reveals **critical
implementation insights** and best practices for deploying advanced AI
technologies at billion-user scale.

- **Foundation Model Scaling Strategies**: Incremental deployment, careful
  capacity management, template-based scaling laws, and power-law behavior
  validation.
- **Hardware-Software Co-design Principles**: Memory hierarchy optimization,
  arithmetic intensity maximization, and activation buffer management.
- **Multi-Task Learning Deployment**: Shared representation learning, gradient
  balancing, and task dependency management.
- **Generative System Implementation**: Semantic ID quality, tokenization
  strategies, and end-to-end optimization.
- **Sequential Modeling Production Lessons**: Stochastic length sampling,
  real-time processing, and temporal bias incorporation.
- **Organizational and Process Insights**: Cross-functional collaboration,
  iterative deployment, comprehensive A/B testing, and standardized evaluation
  frameworks.

---

**All links are in-place and reference internal Meta resources for further
reading.** If you need this in a downloadable file or want further formatting
tweaks, let me know!

# Slimper + SIM Proposal: Detailed Summary & Key Concepts

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results & Gains](#key-results--gains)
3. [Technical Details](#technical-details)
    - Slimper V1 & V2
    - SIM Integration
4. [Principals Driving Gains](#principals-driving-gains)
5. [SIM: What It Means](#sim-what-it-means)
    - Architecture
    - Workflow
    - Variants
6. [Evaluation & Metrics](#evaluation--metrics)
7. [Risks & Validation](#risks--validation)
8. [References & Further Reading](#references--further-reading)

---

## 1. Overview

**Slimper + SIM** is a model architecture proposal for Meta’s personalization and recommendation systems.  
It merges the Slimper V2 architecture (a major revamp for efficiency and quality) with SIM (Search-based Interest Modeling), aiming for additive NE (Normalized Entropy) gains and improved resource efficiency.

---

## 2. Key Results & Gains

| Model Variant      | NE Gain | Training QPS Impact |
|--------------------|---------|---------------------|
| Slimper V2         | +0.405% | -5.99%              |
| SIM                | +0.1225%| -6%                 |
| Slimper + SIM      | +0.575% | -16.29%             |

- **No cannibalization:** Gains are additive, not overlapping.
- **Resource Impact:** Notable increase in GPU memory usage (+20.78%) and combined QPS cost (-29.8%), but gains justify the cost per validation bars.

---

## 3. Technical Details

### Slimper V1

- 3 layers, sequence length 512.

### Slimper V2

- **Sequence length:** 512 → 2048
- **Layers:** 3 → 6
- Added raw watch time tensor, 3 DataFM features, responsiveness sparse features.
- Reduced HSTU embedding dim to 128.
- UHM fusion split for seq len scaleup.
- Removed layer-wise task blob update.
- Enabled model compile.
- Split large HBM embedding table for GPU memory optimization.
- Position embedding optional.

### Slimper + SIM Integration

- **SIM Preproc and SIM Attn** added.
- SIM and UHM share embedding tables.
- SIM preproc strategy unchanged.
- **SIM_MAX_SEQ_LEN:** 22,000
- **SIM_MAX_NUM_MATCHES:** 64

---

## 4. Principals Driving Gains

### A. Model Architecture Revamp

- **Deeper, longer model:** More layers and longer sequence length allow richer user history/context modeling.
- **Feature engineering:** New features (raw watch time, DataFM, responsiveness) improve signal capture.

### B. Efficient Embedding & Memory Optimization

- **Embedding table splitting:** Optimizes GPU memory usage, enabling scale-up.
- **Reduced embedding dimensions:** Balances expressiveness and efficiency.

### C. SIM Integration

- **SIM Preproc & Attention:** Targeted sequence matching and attention for improved personalization.
- **Shared embedding tables:** Reduces memory footprint, improves efficiency.

### D. Training & Serving Efficiency

- **Model compile & fusion splitting:** Unblocks further scale-up and efficiency.
- **Position embedding optionality:** Flexible model configurations.

### E. Validation & Risk Management

- **Bar validation checks:** NE gains must justify resource costs.
- **Calibration:** Maintained within [0.95, 1.05] for reliable predictions.

---

## 5. SIM: What It Means

### **SIM = Search-based Interest Modeling**

#### **Architecture**

- **Two-stage process:**
    1. **General Search Unit (GSU):** Searches user’s long engagement history, selects top-K relevant items.
    2. **Exact Search Unit (ESU):** Applies attention between candidate and selected history items for precise interest modeling.

#### **Workflow**

- **Hard Search:** Uses explicit features (topic ID, author ID) for matching.
- **Soft Search:** Uses embedding similarity for nuanced matching.

#### **Purpose**

- **Scalability:** Handles very long user histories (10k–50k+ events).
- **Relevance:** Focuses on most relevant historical interactions.
- **Efficiency:** Reduces noise and unnecessary computation.

#### **References**
- [CFR Search-based Interest Modeling (SIM) Proposal Design]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1310468516911179/permalink/1420523679238995/}}
- [SIM Soft Search Proposal: Metrics & Methodology]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/3583444701719170/permalink/9492807990782782/}}
- [SIM Optimization status update]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1310468516911179/permalink/1404589910832372/}}
- [Cargo Proposal] Search-based Interest Modeling {{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1853065398160458/permalink/3611171839016463/}}
- [MC Proposal] Slimper + SIM {{#metamate_citation DOCUMENT https://fb.workplace.com/groups/23907659715519923/permalink/25473655645586981/}}

---

## 6. Evaluation & Metrics

### **Offline NE Readings (Critical Tasks)**

- **All tasks show NE improvements** for candidate vs. baseline (comment, like, share, linear_vpvd).
- **Mean p-diff:** -0.6725 (train), -0.575 (eval)
- **US/CA and US/CA YA segments:** Consistent NE improvements.

### **Negative Event NE Readings**

- **Dislike, hide_xout, show_less, skip_post:** All show NE reductions (improvements).
- **Mean p-diff:** -0.82 (train), -0.64 (eval)
- **US/CA and US/CA YA segments:** Consistent NE improvements.

### **Training Efficiency & Resource Impact**

- **Training QPS:** -16.29% (from 700K to 586K)
- **GPU Memory Impact:** +20.78%
- **Combined QPS Cost:** -29.8%

### **Serving Evaluation & Load Test**

- **Instructions provided for running serving eval and load tests.**
- **Capacity cost measurement:** Feature bytes, model cost, pipeline/storage impact.

### **O2O Accuracy & Top-line Gains**

- **IFR-specific metric gains:** 90s VVS, comment/dislike/like/reshare/xout per VPV, show more CTR, distinct VPV, photo/video shifts.
- **Top-line gains:** FB+MSGR session, Blue timespent, rVPV, global VPV, FB session, DAU.

---

## 7. Risks & Validation

- **Risks:** GPU memory increase, training QPS regression, potential load test CPS regression.
- **Bar Validation:** Calibration, NE/RMSE gains/losses, efficiency savings must justify losses.

---

## 8. References & Further Reading

- [Slimper V1 Model Arch Credit](https://fb.workplace.com/groups/352549176749191/permalink/1132371748766926/)
- [Proposal Notebook](https://www.internalfb.com/intern/anp/view/?id=8687679)
- [Diff Stack](https://www.internalfb.com/intern/diff/87850470/)
- [Serving Eval Template](https://fburl.com/vanguard/0lybcuzw)
- [Deltoid Top-line](https://fburl.com/deltoid3/725zeyfh)
- [IFR-specific Deltoid](https://fburl.com/deltoid3/gla05lif)
- [CFR Search-based Interest Modeling (SIM) Proposal Design]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1310468516911179/permalink/1420523679238995/}}
- [SIM Soft Search Proposal: Metrics & Methodology]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/3583444701719170/permalink/9492807990782782/}}
- [SIM Optimization status update]{{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1310468516911179/permalink/1404589910832372/}}
- [Cargo Proposal] Search-based Interest Modeling {{#metamate_citation DOCUMENT https://fb.workplace.com/groups/1853065398160458/permalink/3611171839016463/}}
- [MC Proposal] Slimper + SIM {{#metamate_citation DOCUMENT https://fb.workplace.com/groups/23907659715519923/permalink/25473655645586981/}}

---

## **TL;DR**

- **Slimper + SIM proposal delivers strong NE gains (+0.575%) with significant training QPS cost (-16.29%) and increased GPU memory usage.**
- **All critical and negative event tasks show NE improvements.**
- **Risks include resource usage regressions, but gains justify costs per validation bars.**
- **SIM (Search-based Interest Modeling) is a two-stage architecture for efficiently leveraging long user histories in personalization models.**

---

**Let me know if you want this further condensed, split into slides, or focused on a particular section!**