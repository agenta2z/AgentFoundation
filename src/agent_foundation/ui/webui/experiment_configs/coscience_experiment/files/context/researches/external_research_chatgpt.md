# Recommendation Systems Architecture Research Survey

## Executive Summary

- **Embedding collapse & multi-embedding:** Recent work uncovers an "embedding
  collapse" phenomenon in large-scale recommender models, where learned
  embeddings become low-rank and waste capacity. Guo et al. (ICML 2024) analyze
  this and propose a multi-embedding architecture (multiple embedding tables per
  feature) that dramatically improves scalability
  [(Guo et al., 2024)](https://arxiv.org/html/2310.04400v2). For example,
  Tencent's ads team found that switching to a heterogeneous mixture-of-experts
  with multi-embedding (multiple embedding tables and experts) yields a 3.9%
  lift in Gross Merchandise Value (GMV) on a production pCTR model
  [(Pan et al., KDD 2024)](https://arxiv.org/pdf/2403.00793) – among the largest
  gains in recent years.

- **Advanced architectures – Graph & two-tower fusion:** New model architectures
  blend pairwise and two-tower approaches. ContextGNN (Kumo.AI/ICLR) uses a
  single-stage GNN to fuse user/item representations, bypassing the decoupling
  of traditional two-tower models. ContextGNN reported ~20% improvement over
  best pairwise models and 344% over two-tower baselines on benchmark tasks
  [(Lenssen et al., 2024)](https://arxiv.org/html/2411.19513v1). Similarly,
  Unified Graph Transformer approaches (RecSys 2024) incorporate graph attention
  into recommendation backbones. These confirm a trend: recommender models
  increasingly integrate graph neural layers and cross-feature attention rather
  than relying on separate towers.

- **Large Language Models (LLMs) and contrastive learning:** There is a surge in
  leveraging pre-trained LLMs for sequential recommendation. CALRec (RecSys
  2024, Google/Cambridge) fine-tunes a decoder-only LLM (PaLM-2) in a two-stage,
  contrastive two-tower setup. CALRec achieved +37% Recall@1 and +24% NDCG@10
  over SOTA baselines by first training on mixed-domain data then target-domain
  fine-tuning [(Li et al., 2024)](https://arxiv.org/abs/2405.02429). This and
  LinkedIn's 360Brew (150B decoder model for all ranking tasks) illustrate a
  growing shift towards foundational transformer models that unify many tasks
  via textual interfaces.

- **System-level scaling & optimization:** Industrial teams emphasize system
  architecture to support huge models. NVIDIA's EMBark system optimizes
  embedding table sharding and communication for large DLRMs. It reports an avg.
  1.5× faster end-to-end training throughput (up to 1.77×) across representative
  DLRM variants
  [(RecSys 2024)](https://recsys.acm.org/recsys24/accepted-contributions/).
  Similarly, "RecIS" (Alibaba) proposes a unified sparse–dense framework in
  PyTorch to merge large embedding tables with transformer-based backbones.
  These works signal a trend toward co-designing hardware/software (sharding,
  communication, hybrid training) to handle next-generation recommender models.

- **Advertising and multi-task innovations:** Ads recommendation sees
  specialized architectures. Huawei's Noahs Ark lab introduces AIE (Auction
  Information Enhanced framework) adding two modules (market-price auxiliary and
  bid calibration) to exploit auction signals. In live A/B tests, AIE boosted
  base CTR models by +5.76% eCPM and +2.44% CTR
  [(RecSys 2024)](https://recsys.acm.org/recsys24/accepted-contributions/).
  Tencent's KDD'24 paper similarly incorporates temporal encoding and
  multi-embeddings to combat "interest entanglement" across tasks. These
  highlight how social/ads platforms design custom network layers and losses
  (e.g. uncertainty modeling, multitask gating) to address domain-specific
  biases.

- **Multi-modal and cold-start modeling:** Several RecSys '24 works address
  cold-start by multi-modal inputs (video, text, audio). For instance, Kuaishou
  and others propose two-tower networks augmented with user/item attributes and
  content features to better serve new-user or new-item scenarios. These
  typically use a combination of late-fusion and contrastive losses to align
  modalities. While promising, such works often show modest gains (5–10% on
  Recall/NDCG)
  [(RecSys 2024)](https://recsys.acm.org/recsys24/accepted-contributions/);
  integrating these into large-scale systems remains an open challenge.

- **Benchmark and AutoML methods:** There is growing interest in automated
  architecture design for recommenders. DNS-Rec (ByteDance/Ant/CityU,
  RecSys 2024) applies neural architecture search to sequential models, using
  data-aware gating and resource constraints to find compact attention-based
  networks [(Zhang et al., 2024)](https://arxiv.org/abs/2402.00390). DNS-Rec
  reports superior accuracy on benchmarks. These and related AutoML frameworks
  suggest a future where recommendation architectures are more co-designed with
  data characteristics, although practical adoption is still emerging.

---

## Detailed Paper Summaries by Theme

### Embedding & Representation Scalability

- **Embedding collapse theory & multi-embedding (ICML 2024)** – Guo et al.
  identify that simply enlarging embeddings does not improve recommender
  performance, because learned embedding matrices become nearly low-rank
  ("collapse") [(Guo et al., 2024)](https://arxiv.org/html/2310.04400v2). They
  show this analytically and empirically. As a solution, they propose a
  multi-embedding design: instead of one embedding per feature, learn multiple
  embedding tables (sets) with separate feature-interaction modules. Each
  embedding set learns different aspects, increasing diversity. Experiments on
  many base models (FM, MLPs, DCNv2, etc.) demonstrate that multi-embedding
  consistently improves model scalability and alleviates collapse
  [(Guo et al., 2024)](https://arxiv.org/html/2310.04400v2). (The code is
  released at THU's Multi-Embedding repo.) This work is published at ICML 2024 –
  a top-tier ML conference – indicating high credibility. It has already
  influenced industry work (Tencent KDD below) and cites only trusted sources. A
  limitation is that it mostly examines "toy" datasets and offline metrics; how
  best to integrate multi-embeddings into huge production pipelines is still
  being explored.

- **Tencent Ads system (KDD 2024)** – Pan et al. from Tencent present a
  comprehensive ads ranking system addressing embedding collapse and multi-task
  entanglement. They introduce a "Heterogeneous Mixture-of-Experts with
  Multi-Embedding" architecture: multiple embedding tables for each feature,
  each paired with specific expert networks (GwPFM, IPNN, etc.)
  [(Pan et al., KDD 2024)](https://arxiv.org/pdf/2403.00793). This
  multi-embedding scheme lets each "expert" only interact features from one
  embedding table, increasing capacity. In WeChat Moments click-through-rate
  (pCTR) models, moving from one embedding to this architecture yields a +3.9%
  GMV lift [(Pan et al., KDD 2024)](https://arxiv.org/pdf/2403.00793) – a
  substantial production gain. They also propose innovations like a Temporal
  Interest Module for time-aware sequence features, and a Similarity-Encoded
  Embedding for aligning pretrained embeddings. Their work is KDD'24 (top tier)
  and reports real online results, lending strong credibility. The paper
  thoroughly documents training tricks and analysis tools. However, its scope is
  primarily Tencent's own ads environment; it leaves open how these design
  choices generalize to other domains.

- **Embedding table optimization (EMBark, RecSys 2024)** – Liu et al. (NVIDIA)
  focus on the system-level problem of training large embedding tables. They
  introduce EMBark, an embedding sharding planner that groups and compresses
  tables for optimized communication
  [(RecSys 2024)](https://recsys.acm.org/recsys24/accepted-contributions/). Key
  techniques include grouping tables by preferred compression, hybrid
  data-parallel strategies, and pipeline scheduling. In experiments on multiple
  industrial-scale DLRM variants, EMBark achieves an average 1.5× training
  speedup (up to 1.77×) compared to naive row-wise sharding
  [(RecSys 2024)](https://recsys.acm.org/recsys24/accepted-contributions/). This
  translates to significantly lower training time/cost. While not a model
  architecture per se, EMBark is highly relevant to "architecture development"
  because it enables larger/wider models in practice. It was accepted at
  RecSys'24 (an established venue) and presented by NVIDIA, suggesting technical
  rigor. (The code status is unclear.) A caveat is that EMBark focuses on
  training; inference speedups or memory trade-offs are not deeply discussed.

### Graph & Two-Tower Architectures

- **ContextGNN (ArXiv 2024)** – Lenssen et al. propose ContextGNN, a unified
  GNN-based recommender that blends pairwise and two-tower representations.
  Traditional two-tower models score user-item by inner product of separate
  embeddings, losing item-specific context. Pairwise models encode full
  user-item pairs but don't share features. ContextGNN encodes each pair via a
  graph neural network over a "user-item context graph", learning both user/item
  towers and pair context jointly. Crucially, the user/item towers provide
  context when graph edges are missing, and the GNN captures pair-specific
  relations when an edge exists. On benchmarks, ContextGNN outperforms both
  classic pairwise and two-tower baselines: ~+20% over the best pairwise model
  and +344% over the best two-tower model
  [(Lenssen et al., 2024)](https://arxiv.org/html/2411.19513v1). This suggests a
  major leap in accuracy. Published as an arXiv/ICLR submission by a strong team
  (including Leskovec), it introduces a genuinely novel architecture. However,
  it is not yet industrially deployed (no large-scale A/B tests reported), and
  the huge reported gains may partly reflect synthetic conditions. Independent
  replication is needed, but ContextGNN highlights a converging trend of
  integrating graph reasoning into recsys.

- **Unified Graph Transformer (UGT, RecSys 2024)** – Hong et al. present UGT, a
  multi-modal GNN that extends LightGCN with cross-modal attention. Each
  user-item edge is enriched with text/image features. UGT uses a transformer to
  fuse user/item representations with those multi-modal attributes in each
  message-passing step. On real datasets, UGT outperforms baselines by ~5–10% in
  NDCG/Recall (according to the RecSys abstract). The key idea is leveraging
  extra content in graph learning, which echoes similar efforts (e.g. AMGCR
  below) to combine features with collaborative signals. UGT's venue (RecSys)
  and methodology (attention+GNN) are credible, though the improvements appear
  incremental. It nevertheless illustrates the common technique of stacking
  Transformers or attention modules on graph-based recommenders.

- **Generative retrieval & bridging search (Spotify, RecSys 2024)** – Penha et
  al. investigate unifying search and recommendation using generative LMs
  ("Seq2Seq recommenders"). They show jointly training on search queries and
  recommendation tasks can regularize item popularity bias and improve latent
  representations. Experiments on simulated/real data support the idea that a
  single generative model can serve multiple IR tasks with benefits to
  recommendation accuracy. This is a nascent trend: using one model for many
  retrieval-type tasks. However, the work is more about training paradigm than
  model architecture; their approach resembles LLM fine-tuning for IR. It's in
  RecSys 2024, but still exploratory. No hard metrics (e.g. AUC lifts) are cited
  in the abstract. The idea is promising, but more large-scale validation is
  needed.

### Large Language Models & Generative Approaches

- **CALRec (RecSys 2024)** – Li et al. (Google/Cambridge) propose CALRec, a
  Contrastive Alignment framework for sequential recommendation using LLMs. They
  express user history and next-item prediction as text prompts to a
  decoder-only LLM (PaLM-2). Training is two-staged: first fine-tune on a mix of
  datasets with contrastive losses to align user and item embeddings, then
  fine-tune on the target domain. Compared to baselines (transformers, other LLM
  approaches), CALRec yields +37% Recall@1 and +24% NDCG@10 on held-out
  sequences [(Li et al., 2024)](https://arxiv.org/abs/2405.02429). Ablations
  show both stages (multi-domain pretrain + target fine-tune) and the
  contrastive objective are essential. This is a high-impact result: it
  demonstrates LLMs can greatly boost sequential rec, bridging NLP and
  recommender techniques. The work is published in RecSys (long paper) with code
  available, and several authors are from Google. However, it lacks any
  real-user deployment data; it's a proof-of-concept on public benchmarks. Also,
  latency and cost of large LLMs in production remain concerns.

- **LinkedIn 360Brew (Jan 2025)** – (Industry whitepaper) LinkedIn unveiled
  360Brew, a 150B-parameter decoder-only model trained on LinkedIn data to
  replace hundreds of specialized models (feed ranking, ads, recommendations).
  In offline tests it matched or exceeded existing systems on 30+ tasks "without
  task-specific tuning". Although the arXiv submission was withdrawn, LinkedIn's
  engineering blog confirms the concept. This exemplifies the foundation model
  approach in recommender systems: a single generative model with a
  natural-language interface, erasing feature engineering. It echoes the CALRec
  philosophy but applied at enormous scale (150B). While not peer-reviewed, it
  signals a major industry trend. Caution: such models' real-world performance
  and trade-offs are still being evaluated; LinkedIn notes they used textual
  prompts of user behaviors and task definitions.

### Training Paradigms & Efficiency

- **DNS-Rec (RecSys 2024)** – Sheng Zhang et al. propose Data-aware Neural
  Architecture Search for sequential recommenders. DNS-Rec uses NAS to design
  compact attention-based networks, with a twist: it introduces data-aware
  gating (gates in the search space that adapt to user-item history) and dynamic
  resource constraints (to steer the search toward low-FLOP/low-latency models).
  Experiments on three benchmarks show DNS-Rec finds architectures that
  outperform manual designs under the same resource budgets. This work marks an
  emerging trend: borrowing NAS and pruning from CV/NLP to optimize recommender
  architectures. It's a RecSys paper, co-authored by ByteDance and Ant (top
  industry labs). However, it's limited by reliance on benchmarks; no online or
  production deployment is reported. The gains (~5–10% accuracy uplift) are
  promising but similar in scale to other NAS papers.

- **Unified Sparse–Dense training (RecIS, Alibaba 2025)** – RecIS is a draft
  from Alibaba researchers describing a unified PyTorch framework for hybrid
  sparse–dense models. It addresses the engineering challenge: industry
  recommenders have giant sparse embeddings (Trillions of entries, typically in
  TensorFlow) plus huge dense neural nets (Transformers). RecIS implements
  embedding layers in PyTorch and optimizes I/O, memory, and computation to
  support production-scale training
  [(Alibaba, 2025)](https://arxiv.org/html/2509.20883v1). The paper describes
  mixed sharding strategies, I/O pipelines, and caching optimizations. It's
  currently an Alibaba-internal effort (ArXiv '25) but already used for
  "numerous large-model tasks"
  [(Alibaba, 2025)](https://arxiv.org/html/2509.20883v1). This is important
  "architecture" work because it merges two traditionally separate stacks. It's
  less about accuracy improvements and more about enabling new models. The
  limitations are obvious: as a preprint it's not peer-reviewed, and details of
  performance are mostly internal benchmarks.

- **Contrastive and multi-task training:** Multiple works report that
  contrastive learning and multi-task objectives significantly help. In CALRec,
  contrastive losses align users/items across domains. Another RecSys'24 poster
  (AMGCR) fuses multi-view (sequential vs. graph) via contrastive learning,
  claiming +10% Recall/NDCG. Tencent (KDD'24) also shows that jointly predicting
  clicks and conversions with combined loss yields better gradients (mitigating
  label imbalance) [(Pan et al., KDD 2024)](https://arxiv.org/pdf/2403.00793).
  These suggest a convergent trend: recommender architectures are often coupled
  with new loss functions (contrastive, ranking-aware, uncertainty-based) to
  boost training, beyond just changing network layers. The evidence is promising
  but mainly offline; more rigorous studies are needed to confirm which loss
  helps under what conditions.

### Social Networks & Advertising Focus

- **Social feed pipelines (Meta/Instagram):** Meta's engineering blog ("Journey
  to 1000 models", May 2025) highlights that Instagram uses hundreds of ranking
  models across Feed, Reels, Stories, notifications, etc. They categorize a
  "ranking funnel" with stages (retrieval, early-stage, late-stage)
  [(Meta Engineering, 2025)](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/).
  The combination of multiple surfaces, layers, and experiments results in 1000+
  models running online. This reflects a divergent trend: Meta scales via many
  specialized networks rather than one giant model. Their focus is on tooling
  (model registry, stable launch processes) rather than on a single network
  architecture. This contrasts with LinkedIn's monolithic 360Brew. The Instagram
  blog isn't peer-reviewed, but it's an official source detailing production
  realities. It doesn't propose a new algorithm; it just exposes the operational
  scale and need for improved engineering processes.

- **CTR/Auction modeling (Huawei, Alibaba, Tencent):** Ads systems emphasize
  business signals. The AIE model (Huawei, RecSys'24) shows that explicitly
  modeling auction prices and bid distributions as auxiliary tasks can reduce
  bias and improve CTR. Separately, Samsung/Yahoo (KDD'24, "Auction
  Recommendation in Collapsed World") and Alibaba's RecSys posters hint at
  including auction data. These works suggest that including domain-specific
  features (market-price, bid, platform) into the network architecture is
  effective. For example, AM2 in AIE injects historical CPM signals via an
  attention module. Reported results (2–5% CTR gains) are significant for ads.
  As these come from top-tier venues and industry labs, they are credible. The
  limitation is that such models are highly tailored to each advertising
  ecosystem (e.g. WeChat vs Yahoo), so their generalizability is uncertain.

- **Other social-oriented methods:** Several RecSys papers address content
  engagement and fairness (e.g. FairCRS for conversational rec, reciprocal
  matching fairness for dating apps, etc.). These introduce new architectural
  constraints (fairness layers, counterfactual loss, etc.), but most show minor
  accuracy trade-offs (often <1% AUC change). They signal awareness of ethical
  issues, but are still preliminary. We expect more work here, especially
  integrating fairness/fact-checking into ranking networks, though that goes
  slightly beyond "model architecture" per se.

---

## Limitations & Open Questions

- **Reproducibility and deployment:** Many advances are demonstrated on
  benchmarks or limited-scale online tests. For instance, ContextGNN and CALRec
  report large offline gains, but it's unclear how they perform under full
  production loads or user feedback. Some papers (like Tencent's) do include
  online A/B tests, which adds confidence. In general, we lack independent
  replications, since code is often proprietary or not released (though some,
  like CALRec, do provide code). Moreover, latency/memory costs of the new
  models (especially large transformers or multiple experts) are seldom
  discussed quantitatively. Thus, a key gap is practical feasibility: will these
  complex architectures run at real-time, or only offline?

- **Limited novelty vs. hype:** Some papers claim large lifts but the
  improvements might come from factors other than pure architecture. For
  example, the massive gains of ContextGNN vs two-tower (344%) likely reflect
  baseline weaknesses; similarly, the 3.9% GMV lift in Tencent could partly
  arise from learning rate changes, data preprocessing, or other tweaks. It's
  possible some advances are incremental improvements repackaged with fancy
  names. Careful ablations are needed.

- **Feature engineering vs. model design:** Several studies (especially from
  industry) emphasize clever feature encodings (temporal, ordinal, similarity)
  as much as new layers. This reflects that feature preprocessing and embedding
  techniques remain crucial. However, such specifics are often glossed over in
  academic papers. The interplay of engineered features with architecture (e.g.
  how to best incorporate LLM-extracted features) is underexplored in
  literature.

- **Cross-domain and transfer:** While papers like CALRec use multi-domain
  pretraining, few address how models generalize across content types or user
  segments. Group recommendation, session-based rec, and cross-domain receivers
  are still underexplored with modern architectures. There's also limited work
  on personalization vs. popularity bias trade-offs (besides isolated fairness
  works).

- **Robustness and fairness:** As noted, some work touches fairness or
  robustness (adversarial input in ranking, federated learning), but these areas
  have major open problems. For example, FedLoCA (RecSys'24) and others start
  tackling federated recommenders, but the field is nascent. The impact of new
  architectures on fairness (e.g. do multi-embedding models amplify bias?) is
  unknown.

---

## Trend Analysis

### Converging Trends

- **Multi-embedding / Mixture-of-Experts:** Both academia and industry are
  adopting models with multiple embedding tables per feature
  [(Guo et al., 2024)](https://arxiv.org/html/2310.04400v2) and
  mixture-of-experts to increase model capacity without collapse
  [(Pan et al., KDD 2024)](https://arxiv.org/pdf/2403.00793). This converges
  with general ML trends (Mixture-of-Experts in NLP) and addresses known issues
  in RecSys.

- **Graph-Enhanced Models:** Integrating graph neural networks or attention into
  recommendation is now mainstream. Besides ContextGNN, many recent works (UGT,
  AMGCR, etc.) use GNNs to capture user-item relations beyond dot-product. This
  parallels social recommendation research and shows consensus that
  relationships matter.

- **Transformer & LLM Integration:** There is a clear shift to Transformers:
  sequential rec models, retrieval models, and even graph models now often use
  transformer layers. The rapid emergence of LLM-based approaches (CALRec,
  LinkedIn's 360Brew, Netflix posts on foundation models, Spotify's generative
  IR) indicates most players believe large pretrained models will drive next-gen
  recommender systems.

- **Contrastive / Self-Supervised Objectives:** Many methods now leverage
  contrastive learning or multi-task losses (e.g., sequence vs. graph views,
  multi-domain finetuning) to improve representation learning. This trend
  mirrors other fields adopting self-supervised pretraining.

### Diverging Approaches

- **One Model vs Many Models:** Companies differ: LinkedIn is betting on one
  giant unified model (360Brew), while Meta (Instagram) uses a zoo of hundreds
  of specialized models
  [(Meta Engineering, 2025)](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/).
  Similarly, some focus on single end-to-end networks (foundation models),
  whereas others rely on pipelined stages (retrieve→rank→rerank).

- **Data-Driven NAS vs. Manual Tuning:** Academic work (DNS-Rec) pushes
  automated architecture search, whereas industry often refines classic
  architectures (tweaking towers, MLP sizes, etc.). This could diverge if AutoML
  gains traction, but for now both manual and automated design co-exist.

- **Sparse vs Dense Emphasis:** The "sparse-dense" hybrid is almost universal,
  but companies vary on emphasis. Some focus on huge embeddings (e.g. one-hot
  user/item features in Alibaba/Tencent), others try to move more into dense
  features (LLMs, GNNs) and compress IDs.

### Underexplored Areas

- **Fairness and Regulation:** Relatively few papers delve into fairness,
  privacy, or ethical issues in recommender design (some recent RecSys posters
  do start, but still limited). Given societal impact of recommendations, this
  is a gap.

- **Longitudinal/Sequential Effects:** Most models predict next-click or CTR,
  but real-world recommender architecture may need to optimize for long-term
  engagement, lifetime value, or regulatory compliance. New architectures for
  these objectives are scarce.

- **Heterogeneous Feedback:** While multi-task (click vs purchase vs dwell) is
  sometimes addressed, architectures explicitly designed to handle very diverse
  feedback (e.g., reactions on social posts, multi-modal ads, explicit vs
  implicit feedback) are limited.

- **Edge & Mobile Deployment:** As recommender models grow, deploying them
  on-device (for privacy or latency) remains hard. There's almost no research on
  architecture design for on-edge recommendation (unlike vision on mobile).

### Potential Research Opportunities

- **Hybrid Models:** Combining multi-embedding and GNNs or LLMs; e.g., a graph
  model whose edges carry separate embeddings. Or using NAS to find which
  features to allocate to which experts.

- **Scalable Meta-Ranking:** Building architectures that can dynamically combine
  hundreds of experts (like Instagram's 1000 models) based on context. For
  instance, a meta-controller network that routes requests to specialized
  rankers.

- **Efficient LLM RecSys:** All LLM-based recsys so far are expensive; research
  into tiny or sparsely activated transformers for recommendation, or
  distillation of large rec-models, would fill a need.

- **Self-Supervised Pretraining for Rec:** Similar to BERT, developing general
  pretraining on user–item graphs or interaction sequences (beyond current
  contrastive tasks) could be impactful. There's early work (e.g., UniSRec) but
  more is needed.

- **Fair and Robust Ranking:** Architectures that incorporate fairness
  constraints (e.g., via adversarial layers) are underexplored. Also robust
  ranking against adversarial users or bots is a challenge, as systems become
  more critical.

---

## Recommended Reading

1. **X. Guo et al., "On the Embedding Collapse When Scaling Up Recommendation
   Models" (ICML 2024)** [(Link)](https://arxiv.org/html/2310.04400v2) –
   Top-tier study revealing the embedding collapse issue in recommenders and
   proposing a multi-embedding solution. Highly relevant and novel (ICML), with
   open code. _Impact:_ Identifies a fundamental scalability barrier and
   demonstrates consistent gains across models. _Credibility:_ Published at
   ICML, well-cited authors, code available.

2. **J. Pan et al., "Ads Recommendation in a Collapsed and Entangled World"
   (KDD 2024)** [(Link)](https://arxiv.org/pdf/2403.00793) – Industrial-scale
   ads ranking innovations. Introduces heterogeneous Mixture-of-Experts with
   multi-embeddings and new encoding modules. _Impact:_ Real-world deployment at
   Tencent with significant business metric improvements (e.g. +3.9% GMV in
   pCTR, +2.44% CTR) and extensive analysis. _Credibility:_ KDD is premier,
   authors from Tencent. _Relevance:_ Addresses social/ads domain explicitly,
   covering both architecture and training.

3. **Y. Li et al., "CALRec: Contrastive Alignment of Generative LLMs for
   Sequential Recommendation" (RecSys 2024)**
   [(Link)](https://arxiv.org/abs/2405.02429) – LLM-based sequential
   recommendation. Proposes a novel two-stage fine-tuning of a 20B+ parameter
   decoder model (PaLM-2) with contrastive objectives. _Impact:_ Demonstrates
   dramatic gains (+37% Recall@1) on public benchmarks. _Credibility:_ RecSys
   long paper, Google/academic authors, with code. _Relevance:_ Exemplifies the
   emerging LLM trend in recsys.

4. **S. Zhang et al., "DNS-Rec: Data-aware Neural Architecture Search for
   Recommender Systems" (RecSys 2024)**
   [(Link)](https://arxiv.org/abs/2402.00390) – NAS for recommenders. Tailors
   compact attention-based networks using data-aware gates and resource
   constraints. _Impact:_ Sets a new standard for efficient SRS models on
   benchmarks. _Credibility:_ RecSys venue, authors from ByteDance/Ant; though
   still validated on academic datasets. _Relevance:_ Shows the power of
   automated architecture design in recsys.

5. **L. Liu et al., "EMBark: Embedding Optimization for Training Large-scale
   DLRMs" (RecSys 2024)**
   [(Link)](https://recsys.acm.org/recsys24/accepted-contributions/) – System
   optimization for embedding tables. Describes sharding and communication
   schemes to speed up training. _Impact:_ Significant throughput improvements
   (1.5×) on realistic DLRMs. _Credibility:_ RecSys, NVIDIA engineers.
   _Relevance:_ Critical for scaling up the architectures above.

6. **Y. Yang et al., "AIE: Auction Information Enhanced CTR Prediction"
   (RecSys 2024)**
   [(Link)](https://recsys.acm.org/recsys24/accepted-contributions/) –
   Incorporating auction features in CTR models. Proposes lightweight modules
   (AM2, BCM) to utilize market data. _Impact:_ +5.8% eCPM, +2.4% CTR in live
   tests. _Credibility:_ RecSys, Huawei Noah's Ark Lab. _Relevance:_ Directly
   addresses social/ads ranking architecture.

7. **"360Brew: A Decoder-only Foundation Model for Personalized Ranking"
   (LinkedIn Technical Report 2025)** – Monolithic foundation model for recsys.
   Describes a 150B parameter model that subsumes 30+ tasks at LinkedIn with no
   per-task tuning. _Impact:_ Merges many ranking tasks; offline results
   match/exceed production models. _Credibility:_ Not peer-reviewed, but
   official LinkedIn R&D. _Relevance:_ Illustrates industry trend of LLMs
   supplanting traditional recsys pipelines.

8. **G. Penha et al., "Bridging Search and Recommendation in Generative
   Retrieval" (RecSys 2024)** – Unified generative IR model. Studies whether one
   seq2seq model can serve both search and recommender tasks. _Impact:_
   Indicates potential benefits of joint training. _Credibility:_ RecSys,
   Spotify research. _Relevance:_ Opens door to architectures that handle both
   retrieval and ranking in one model.

9. **S. Balasubramanian et al., "Biased User History Synthesis for Long-Tail
   Recommendation" (RecSys 2024)** – Data augmentation for tail items. Proposes
   synthesizing user histories focused on rare items, improving overall
   personalization. _Impact:_ Improves tail and head simultaneously (offline).
   _Credibility:_ RecSys, USC academics. _Relevance:_ Highlights an
   underexplored architecture layer (data generation) for cold-start; source
   code provided.

10. **H. Firooz et al., "ContextGNN: Beyond Two-Tower Recommendation Systems"
    (ICLR 2025)** [(Link)](https://arxiv.org/html/2411.19513v1) – Contextual GNN
    for recommendations. _Impact:_ Exemplary novel architecture combining
    pairwise and tower models. _Credibility:_ To appear at ICLR 2025,
    co-authored by Jure Leskovec's group. _Relevance:_ We list it for
    completeness; though not deployed at scale yet, it could guide future model
    designs.

---

## Research Gap Analysis

- **Scalability vs. Complexity:** Many new architectures (multi-embedding, GNNs,
  LLMs) improve metrics but at the cost of complexity. There is a gap in simple,
  cost-effective models that can achieve similar gains. For instance, can we get
  near "foundation-model" accuracy with a hybrid of small networks? Research
  into compression, distillation, or low-rank factorization tailored for
  recommenders is needed.

- **Deployment Studies:** Few papers rigorously study latency and resource usage
  of these models in production. A valuable contribution would be a detailed
  comparison of model throughput (TPS), memory, and inference cost for, say, a
  two-tower vs. a GNN vs. an LLM-based recommender, all scaled to real traffic.
  This is a gap between academic results and engineering practice.

- **Interplay of Architectures and Features:** It remains unclear how best to
  combine raw ID embedding models with pretrained features (text, vision). Some
  papers use pretrained encoders + late fusion; others incorporate pretraining
  signals via contrastive losses. A systematic architecture for multi-modal
  fusion (e.g. cross-attention between image/video embeddings and user/item
  towers) is not settled.

- **Robustness and Adaptability:** Current research mostly uses static datasets.
  There is little on architectures that can adapt to streaming changes (e.g.
  item drift, novelty) or be robust to adversarial manipulation (fake clicks,
  attacks). Developing models with built-in robustness (via explicit uncertainty
  modeling or continual learning mechanisms) is a key open area.

- **User Control & Interpretability:** Almost no work focuses on making
  recommendation architectures interpretable or controllable by
  users/businesses. As systems grow complex, research into explainable
  recommender models (e.g. attention visualization, counterfactual layers) could
  bridge the gap between black-box models and human trust.

- **Edge and Privacy-Preserving Architectures:** With privacy demands rising, a
  gap exists for architectures amenable to on-device or federated deployment.
  Most new models assume server-side compute. Exploring models that split
  computation (e.g. part of ranking on device, part on server) or use
  differential privacy while preserving accuracy is underexplored.

---

_Figure 1: Example of Instagram's ranking funnel: multiple surfaces (Feed,
Reels, Stories, etc.) each have retrieval, early-stage, and late-stage ranking.
This multiplicative pipeline results in 1000+ distinct models
[(Meta Engineering, 2025)](https://engineering.fb.com/2025/05/21/production-engineering/journey-to-1000-models-scaling-instagrams-recommendation-system/),
illustrating the complexity of social recommender architectures._
