# The Generative Turn in Recommendation Systems: Architectural Evolution, Scaling Laws, and Multi-Task Paradigms (2024–2025)

## 1. Introduction: The End of the "Two-Tower" Era

The period between 2024 and 2025 marks a definitive inflection point in the history of recommendation systems (RecSys). For nearly a decade, the industry standard has been dominated by the "Two-Tower" paradigm—separating retrieval (candidate generation) and ranking (discriminative scoring)—and Deep Learning Recommendation Models (DLRMs) relying on massive embedding tables and feature interaction layers. However, recent developments from major research institutions and technology giants indicate that this architecture is approaching its asymptotic limit. The field is undergoing a fundamental metamorphosis toward Generative Recommendation (GenRec), driven by the adoption of sequential transduction architectures, the validation of scaling laws analogous to those in Large Language Models (LLMs), and the integration of unified multi-task learning frameworks.

This transformation is not merely an academic exercise but an industrial necessity. As user interaction histories grow exponentially and item catalogs expand into the billions, traditional discriminative models struggle to capture long-term dependencies and high-order feature interactions without incurring prohibitive computational costs. The emergence of architectures like Meta's HSTU (Hierarchical Sequential Transduction Unit), Kuaishou's OneRec, and Pinterest's PinRec demonstrates a collective shift toward reformulating recommendation not as a classification problem—predicting the probability of a click—but as a generative sequence modeling task.

This report provides an exhaustive analysis of these developments. We synthesize findings from top-tier conferences including NeurIPS, ICML, KDD, and RecSys, alongside engineering reports from Meta AI, Google DeepMind, Alibaba, Tencent, Shopee, and others. The analysis focuses on three core pillars: the architectural shift to generative sequential modeling, the prioritization of ID-based large-scale multi-task learning, and the infrastructural engineering required to serve these massive models under strict latency constraints.

## 2. Architectural Paradigm Shift: From Prediction to Transduction

The most significant theoretical advancement in the 2024–2025 timeframe is the transition from predictive modeling to sequential transduction. Traditional DLRMs treat user history as a "bag of features" or use simple pooling mechanisms, losing the rich temporal causality of user behavior. The new wave of architectures treats user actions as a modality equivalent to language tokens, enabling the application of Generative AI principles to interaction data.

### 2.1 Meta's Hierarchical Sequential Transduction Unit (HSTU)

The introduction of HSTU by Meta AI represents a watershed moment in distinguishing generative recommendation from standard Natural Language Processing (NLP) transformers. While the success of Transformers in NLP inspired early adoption in RecSys (e.g., SASRec, BERT4Rec), these models suffered from the quadratic complexity ($O(N^2)$) of softmax self-attention, making them computationally prohibitive for the long user sequences typical in industrial settings (e.g., thousands of historical clicks).

Meta's research, specifically the work titled "Actions Speak Louder than Words," validates a new architectural approach. HSTU replaces the traditional softmax attention mechanism with a novel pointwise aggregation scheme. This mechanism reduces the computational complexity from quadratic to linear ($O(N)$) while maintaining the ability to model high-order interactions. [(Zhai et al., ICML 2024)](https://proceedings.mlr.press/v235/zhai24a.html)

#### 2.1.1 Mechanism of Action

The core innovation of HSTU lies in how it processes the input sequence. Unlike standard transformers that attend to all previous tokens via a dense attention matrix, HSTU aggregates key-value pairs cumulatively. The output at time step $t$, denoted as $Y_t$, is computed via a normalized aggregation of interactions between keys ($K$) and values ($V$) up to that point, modulated by the current query ($Q_t$). This design allows the model to "transduce" the sequence of user actions into a latent representation that updates efficiently with each new interaction. [(Zhai et al., 2024 - GitHub)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhai24a/zhai24a.pdf)

Crucially, HSTU processes an interleaved sequence of items and actions. A user's history is not just a sequence of items ($Item_1, Item_2, \dots$) but a causal chain of items and their associated engagement types ($Item_1, Action_{click}, Item_2, Action_{skip}, \dots$). By treating the action type as a distinct token that modifies the latent state of the item, HSTU models the joint distribution of user behaviors rather than just the conditional probability of the next item. [(DFGR, arXiv)](https://arxiv.org/html/2505.16752v3)

#### 2.1.2 Performance and Efficiency

Empirical evaluations demonstrate that HSTU significantly outperforms FlashAttention2-based Transformers. On sequences of length 8192—a scale necessary to capture long-term user interests—HSTU achieves speedups ranging from 5.3x to 15.2x. [(Zhai et al., ICML 2024)](https://proceedings.mlr.press/v235/zhai24a.html) This efficiency gain is not merely for speed; it unlocks the ability to train on orders of magnitude more data. Meta reports that HSTU-based Generative Recommenders (GRs) with up to 1.5 trillion parameters have been deployed, improving online metrics by 12.4% compared to highly optimized DLRM baselines. [(Zhai et al., ICML 2024)](https://proceedings.mlr.press/v235/zhai24a.html)

### 2.2 The "Wukong" Architecture: Scaling Laws for Feature Interaction

While HSTU addresses sequential modeling, the problem of modeling complex feature interactions (e.g., User $\times$ Context $\times$ Item) in non-sequential contexts remains critical. Traditional approaches utilize Multi-Layer Perceptrons (MLPs) or explicitly engineered Cross Networks (DCN). However, Meta's research into "Wukong" highlights a critical deficiency: standard MLPs do not exhibit favorable scaling laws. Simply increasing the depth or width of an MLP does not yield consistent performance gains in recommendation tasks. [(Zhang et al., arXiv)](https://arxiv.org/abs/2403.02545)

Wukong addresses this by proposing Stacked Factorization Machines (SFMs). A standard Factorization Machine (FM) captures second-order interactions (pairs of features). Wukong stacks these FMs hierarchically, where the output of one FM layer serves as the input to the next. Mathematically, if Layer 1 captures 2nd-order interactions, Layer 2 captures interactions of those interactions (effectively 4th-order), and Layer $L$ captures $2^L$-th order interactions. [(Zhang et al., arXiv PDF)](https://arxiv.org/pdf/2403.02545)

This architecture establishes a scaling law for feature interaction models. Meta's experiments across six datasets demonstrate that Wukong's performance improves as a power-law function of compute (GFLOPs), extending beyond 100 GFLOPs/example where prior arts like MaskNet and xDeepFM saturate. [(Zhang et al., arXiv PDF)](https://arxiv.org/pdf/2403.02545) This finding is pivotal, as it suggests that recommendation quality can be predictably improved by investing in larger models and more compute, mirroring the "scaling hypothesis" that drives LLM development.

### 2.3 Kuaishou's OneRec: The End-to-End Generative Framework

Traditional industrial systems operate as a cascade: a retrieval model selects a candidate set (e.g., 1,000 items), and a ranking model scores them. This disconnect creates an information bottleneck—the ranking model can only score what the retrieval model finds. Kuaishou's "OneRec" (2025) challenges this paradigm by proposing a unified, end-to-end generative framework that performs retrieval and ranking simultaneously. [(OneRec-V2, arXiv)](https://arxiv.org/abs/2508.20900)

#### 2.3.1 The Lazy Decoder Architecture

A major challenge in applying generative models to recommendation is efficiency. In a standard Encoder-Decoder architecture, the encoder processes the user's history. If this is done for every recommendation request, the computational cost is prohibitive. OneRec V2 introduces a "Lazy Decoder-Only" architecture. This design decouples the heavy encoding of user history from the lightweight decoding of recommendations. The history is encoded once, and the "lazy" decoder utilizes this context to generate a sequence of items. This architectural change reduced the total computation by 94% compared to standard generative approaches, enabling the scaling of the model to 8 billion parameters. [(OneRec-V2, arXiv)](https://arxiv.org/abs/2508.20900)

#### 2.3.2 Direct Preference Optimization (DPO) in RecSys

OneRec also pioneers the application of Direct Preference Optimization (DPO)—a technique originally developed for aligning LLMs—to recommendation systems. In standard training, models are optimized using Cross-Entropy Loss (predicting the next clicked item). However, this creates a bias toward "clickbait" content that users click but regret. OneRec utilizes DPO to align the generative model with true user satisfaction signals (e.g., watch time or retention). By constructing preference pairs (e.g., "User preferred Video A over Video B"), the model is fine-tuned to maximize the margin between preferred and non-preferred content, effectively aligning the generator with complex, non-differentiable business metrics. [(OneRec, arXiv)](https://arxiv.org/abs/2502.18965)

## 3. Large-Scale Generative Retrieval: Methodologies and Mechanisms

The retrieval stage is undergoing a radical shift from "similarity search" (finding items close to a user vector) to "generative retrieval" (directly generating item identifiers). This shift is driven by the need to capture more complex semantic relationships than simple dot-product spaces can allow.

### 3.1 Pinterest's PinRec: Outcome-Conditioned Retrieval

Pinterest's deployment of PinRec [(PinRec, arXiv)](https://arxiv.org/html/2504.10507v1) serves as a primary case study for industrial generative retrieval. Serving over 550 million users, Pinterest faces a unique challenge: balancing diverse engagement objectives (e.g., "saving" a pin vs. "clicking" a pin). A standard generative model predicting the "next likely action" conflates these signals.

**Outcome-Conditioned Generation:**

PinRec solves this by introducing outcome tokens into the generation process. During training, the input sequence includes the specific action taken by the user (e.g., `<user_history> <action_save> -> <item_id>`). During inference, the system can explicitly "steer" the generation by forcing specific outcome tokens. If the system aims to drive saves, it prefixes the generation with `<action_save>`; if it aims to drive traffic, it uses `<action_click>`. This controllability allows the retrieval engine to dynamically adjust the mix of content based on real-time business logic. [(PinRec v4, arXiv)](https://arxiv.org/html/2504.10507v4)

**Windowed Multi-Token Generation:**

To mitigate the latency of autoregressive decoding, PinRec utilizes a windowed generation strategy. Rather than generating one item at a time, the model generates multiple tokens or items within a specific window, parallelizing the process where possible. This balances the high fidelity of autoregressive modeling with the strict latency requirements of a real-time feed. [(PinRec v4, arXiv)](https://arxiv.org/html/2504.10507v4)

### 3.2 Alibaba's TBGRecall: Next Session Prediction

Alibaba's TBGRecall [(TBGRecall, arXiv)](https://arxiv.org/abs/2508.11977) critiques the standard "Next Item Prediction" task for e-commerce retrieval. In a shopping session, a user often clicks multiple items that are not causally dependent on each other (e.g., opening multiple tabs for comparison) but are jointly dependent on the user's intent. Standard autoregressive models force a false order on these intra-session actions, introducing noise.

**Mechanism:**

TBGRecall reformulates the task as Next Session Prediction (NSP). Instead of predicting $i_{t+1}$, the model predicts the set of items $\{i_a, i_b, \dots\}$ that will constitute the next session. The input architecture partitions the sequence into multi-session segments, separating long-term inter-session dependencies from short-term intra-session noise. This approach has shown significant improvements in Gross Merchandise Volume (GMV) by better capturing the user's broader shopping intent rather than just their immediate next click. [(TBGRecall v2, arXiv)](https://arxiv.org/html/2508.11977v2)

### 3.3 ByteDance's LongRetriever: Ultra-Long Sequence Modeling

ByteDance addresses the context length limitation with LongRetriever. [(LongRetriever, arXiv)](https://arxiv.org/abs/2508.15486) While ranking models often use complex attention mechanisms, retrieval models have historically been limited to short user histories due to latency constraints. LongRetriever introduces "in-context training" and "multi-context retrieval" to the candidate matching stage.

The system employs a search-based mechanism to filter relevant subsequences from a user's lifelong behavior sequence. By retrieving only the most relevant historical segments for a given context, the model can effectively process ultra-long sequences without incurring the quadratic cost of full attention. This enables the retrieval stage to access the same depth of user history as the ranking stage, significantly reducing the "information gap" between the two layers. [(LongRetriever v2, arXiv)](https://arxiv.org/html/2508.15486v2)

### 3.4 Tencent's RARE: LLM-Generated Commercial Intentions

While many systems use opaque IDs, Tencent's RARE (Real-time Ad REtrieval) framework [(RARE, arXiv)](https://arxiv.org/abs/2504.01304) integrates Large Language Models (LLMs) to generate semantic bridges. RARE utilizes a customized LLM to generate Commercial Intentions (CIs)—short text descriptors of user intent (e.g., "summer floral dress")—based on the query and user history. These CIs serve as an intermediate semantic representation.

The system maintains a dynamic index mapping CIs to ads. During inference, the LLM generates CIs, which are then used to retrieve ads. This approach leverages the reasoning and world knowledge of the LLM to understand abstract intent, bypassing the limitations of keyword matching or ID-based retrieval, especially for new or tail queries. [(RARE, ACL Anthology)](https://aclanthology.org/2025.emnlp-main.1473.pdf)

### 3.5 Comparative Analysis of Retrieval Architectures

Table 1 summarizes the key architectural differences among the leading generative retrieval systems in 2025.

**Table 1: Comparative Analysis of State-of-the-Art Generative Retrieval Architectures (2024–2025)**

| System | Institution | Core Task | Architecture | Key Innovation |
|--------|-------------|-----------|--------------|----------------|
| PinRec [(Link)](https://arxiv.org/html/2504.10507v1) | Pinterest | Generative Retrieval | Transformer (Decoder) | Outcome Conditioning: Steers generation toward specific business metrics (Save vs. Click). |
| TBGRecall [(Link)](https://arxiv.org/abs/2508.11977) | Alibaba | Next Session Prediction | Session-Partitioned Transformer | NSP: Predicts item sets per session to handle non-causal intra-session clicks. |
| OneRec [(Link)](https://arxiv.org/abs/2508.20900) | Kuaishou | Unified Retrieve & Rank | Lazy Decoder-Only (MoE) | DPO Integration: Aligns generation with user preference rewards via RLHF-style training. |
| LongRetriever [(Link)](https://arxiv.org/abs/2508.15486) | ByteDance | Lifelong Sequence Retrieval | Search-based Attention | Multi-Context Retrieval: Efficiently processes ultra-long histories via selective subsequence retrieval. |
| RARE [(Link)](https://arxiv.org/abs/2504.01304) | Tencent | Commercial Intent Retrieval | LLM + Dynamic Index | Commercial Intentions: Uses LLM-generated text as an explicit semantic bridge for retrieval. |

## 4. Multi-Task Learning (MTL) and Optimization

As platforms evolve into "Super Apps" (e.g., WeChat, TikTok, Uber), recommendation systems must optimize for multiple conflicting objectives (e.g., Click-Through Rate vs. Retention vs. Revenue) across multiple scenarios (e.g., Feed, Shorts, Mall). 2024–2025 has seen a shift from simple parameter sharing to complex Pareto-optimized and scenario-aware architectures.

### 4.1 Pareto-Ensembled Multi-Task Learning (PEMBOT)

Multi-Task Learning (MTL) often suffers from the "seesaw phenomenon," where improving one task degrades another. Amazon's PEMBOT (Pareto-Ensembled Multi-Task Boosted Trees) [(Amazon Science, KDD 2024)](https://www.amazon.science/conferences-and-events/kdd-2024) introduces a rigorous theoretic approach to this problem. Instead of seeking a single optimal model with fixed task weights, PEMBOT generates a set of solutions distributed along the Pareto frontier.

The architecture utilizes a Gradient Boosted Tree framework to train multiple trees, each optimizing a different trade-off point between tasks. During inference, an ensemble of these Pareto-optimal trees is used. This allows the system to dynamically navigate the trade-off space without retraining, ensuring that no single metric is sacrificed disproportionately. This approach has been deployed in e-commerce logistics to balance competing objectives like shipping speed and defect rates. [(PEMBOT Paper, Amazon Science)](https://assets.amazon.science/e1/a8/3dea9102495d825d23fb4819641d/pembot-pareto-ensembled-multi-task-boosted-trees.pdf)

### 4.2 Scenario-Wise Rec and Architecture Adaptation

The challenge of Multi-Scenario Recommendation (MSR)—serving recommendations across distinct product surfaces—has historically lacked standardized benchmarks. The introduction of the Scenario-Wise Rec benchmark [(Scenario-Wise Rec, arXiv)](https://arxiv.org/html/2412.17374v1) provides a unified pipeline for evaluating MSR models.

Recent innovations in this space include MTmixAtt (Multi-Task Mixture-of-Attention). [(MTmixAtt, arXiv)](https://arxiv.org/html/2510.15286v1) Unlike static shared-bottom models, MTmixAtt employs a Mixture-of-Experts (MoE) backbone where experts are dynamically routed. Some experts are shared across all scenarios to capture universal user preferences, while others are scenario-specific. A gating network explicitly models the differences between scenarios, allowing the model to transfer knowledge where appropriate while preventing negative transfer (interference) between dissimilar scenarios. [(MTmixAtt, arXiv)](https://arxiv.org/html/2510.15286v1)

### 4.3 AEM2TL: Adaptive Transfer Learning

In complex conversion funnels (e.g., Impression $\rightarrow$ Click $\rightarrow$ Cart $\rightarrow$ Buy), data sparsity becomes acute for deeper tasks. AEM2TL (Adaptive Entire-space Multi-scenario Multi-task Transfer Learning) [(AEM2TL, IEEE)](https://www.computer.org/csdl/journal/tk/2025/04/10858412/23VPw4QCkrm) addresses this by modeling the entire space of samples (impressions) rather than just the clicked samples. It introduces a Scenario-Customized Gate Control (Scenario-CGC) and a Task-Customized Gate Control (Task-CGC). These modules adaptively control the flow of information from abundant tasks (clicks) to sparse tasks (conversions) and across scenarios, explicitly handling Sample Selection Bias (SSB). [(AEM2TL, IEEE)](https://www.computer.org/csdl/journal/tk/2025/04/10858412/23VPw4QCkrm)

## 5. Bridging the Semantic Gap: LLMs, Semantic IDs, and Quantization

A critical friction point in the convergence of LLMs and RecSys is the representation of items. LLMs operate on semantic tokens (words), while recommender systems operate on discrete, often arbitrary, IDs. 2025 has seen the crystallization of two distinct approaches: adapting LLMs to understand IDs (PLUM) and generating parallel semantic representations (RPG).

### 5.1 PLUM: Pre-trained Language Models for Unified Recommendations

Google DeepMind's PLUM framework [(PLUM, arXiv)](https://arxiv.org/abs/2510.07784) represents the "adaptation" school of thought. It tackles the Memory Wall problem of DLRMs, where embedding tables for billions of items consume terabytes of memory.

**Semantic IDs (SIDs):**

PLUM replaces random integer IDs with Semantic IDs. These are generated using hierarchical quantization methods like RQ-VAE (Residual Quantized Variational Autoencoder). An item is represented as a tuple of discrete tokens (e.g., `<cluster_12> <subcluster_5> <rank_9>`). Critically, these IDs preserve semantic locality: items with similar content or collaborative history share prefixes in their IDs. [(PLUM, ResearchGate)](https://www.researchgate.net/publication/396373037_PLUM_Adapting_Pre-trained_Language_Models_for_Industrial-scale_Generative_Recommendations)

**Continued Pre-Training (CPT):**

PLUM adapts a standard LLM (e.g., T5) via Continued Pre-Training on user behavior sequences composed of these Semantic IDs. This phase teaches the LLM the "grammar" of user behavior. The model learns that the sequence SID_A, SID_B implies a high probability of SID_C. By shifting the parameter load from memory-intensive embedding tables to compute-intensive transformer weights, PLUM achieves superior sample efficiency and aligns better with hardware accelerators designed for dense matrix multiplication. [(PLUM v1, arXiv)](https://arxiv.org/html/2510.07784v1)

### 5.2 RPG: Solving the Latency of Semantic IDs

A major critique of Semantic IDs is inference latency. If an item is represented by 4 tokens, a standard autoregressive model must execute 4 generation steps to recommend a single item. RPG (Recommendation with Parallel Generation) [(RPG, ResearchGate)](https://www.researchgate.net/publication/392514863_Generating_Long_Semantic_IDs_in_Parallel_for_Recommendation) proposes a solution: parallel decoding.

RPG treats the Semantic ID not as a rigid sequence but as a structured set. It trains independent prediction heads for each token position or utilizes a parallel decoding scheme that generates all constituent tokens of an ID simultaneously. This reduces the inference complexity from $O(L)$ (where $L$ is ID length) to $O(1)$, making Semantic ID-based generation feasible for high-throughput serving systems. [(RPG, ResearchGate)](https://www.researchgate.net/publication/392514863_Generating_Long_Semantic_IDs_in_Parallel_for_Recommendation)

## 6. Emerging Trends in Generative Recommendation: Reasoning and Unified Frameworks

The 2025 landscape highlights a significant move towards not just generating recommendations, but reasoning about why a recommendation is made, and unifying disparate system stages into cohesive generative frameworks.

### 6.1 Shopee's OnePiece: Context Engineering and Reasoning

Shopee's OnePiece framework introduces LLM-style context engineering and multi-step reasoning into industrial cascade ranking systems. Unlike LLMs which naturally use prompts, ranking models have historically lacked structured context.

**Structured Context Engineering:**

OnePiece augments standard interaction history with preference anchors (e.g., top-clicked items for a query) and situational descriptors (e.g., query context). For ranking, it makes candidate items jointly visible to each other, allowing the model to capture cross-candidate interactions—effectively "comparing" items before scoring. [(OnePiece, arXiv)](https://arxiv.org/html/2509.18091v1)

**Block-Wise Latent Reasoning:**

Instead of a direct mapping from input to score, OnePiece employs block-wise latent reasoning. The model progressively refines its hidden states across multiple blocks, creating a "chain of thought" in latent space. This is trained via a progressive multi-task strategy, leveraging feedback chains (exposure $\rightarrow$ click $\rightarrow$ purchase) to supervise the reasoning steps. Deployed at Shopee, OnePiece achieved over +2% GMV per user.

### 6.2 Kuaishou's OneRec-Think: Explicit Reasoning and Dialogue

While OnePiece focuses on latent reasoning, Kuaishou's OneRec-Think brings explicit reasoning to the forefront. It addresses the limitation of generative models acting as "implicit predictors" by enabling them to generate natural language rationales.

**Think-Ahead Architecture:**

The framework operates in two stages. First, a Reasoning-Guided Prefix Generation stage (often offline or cached) generates a "thought" or rationale explaining why a user might want specific content, followed by initial item tokens. Second, a lightweight online stage uses these tokens as a constrained prefix to rapidly generate the final recommendation. This "Think-Ahead" approach balances the heavy compute of reasoning with the low latency required for serving.

**Itemic Alignment:**

To ground the reasoning, OneRec-Think aligns item IDs with textual descriptions through Itemic Alignment, ensuring the model understands the semantic content of the items it recommends. It was deployed on Kuaishou, yielding a 0.159% gain in App Stay Time.

### 6.3 Meta's GEM: A Foundation Model for Ads

Meta has introduced GEM (Generative Ads Model), a massive foundation model designed to power its ads ecosystem. GEM is not just a ranker but a "central brain" that learns from diverse signals across Facebook and Instagram.

**Dual-Attention Architecture:**

GEM processes two distinct types of inputs: sequence features (long-term user history) and non-sequence features (user/ad attributes). It applies customized attention mechanisms to each stream independently before fusing them via cross-feature learning. This preserves the fidelity of long interaction sequences while capturing complex attribute interactions.

**Knowledge Transfer:**

A key innovation is its post-training knowledge transfer. GEM acts as a teacher, distilling its learned representations to lighter, surface-specific models. Meta reports this transfer is 2x more effective than standard distillation, driving a 5% increase in ad conversions on Instagram.

### 6.4 Kuaishou's OneSearch and OneSug: Unified End-to-End Frameworks

Kuaishou has further pushed the "unified" paradigm with OneSearch and OneSug.

**OneSearch (E-commerce Search):** Replaces the fragmented recall-ranking cascade with a single generative model. It utilizes Keyword-enhanced Hierarchical Quantization Encoding (KHQE) to create semantically rich item IDs that preserve hierarchical category information. A multi-view behavior sequence injection strategy allows the model to "see" the user's history from multiple perspectives (e.g., clicks vs. buys). Online A/B tests showed a +3.22% increase in order volume.

**OneSug (Query Suggestion):** Unifies the query suggestion pipeline using a Prefix2Query enhancement module. This module enriches short user prefixes with semantically related queries before generation. It employs reward-weighted ranking to align suggestions with business metrics like order volume, achieving a 2.04% increase in orders.

## 7. Serving Infrastructure and Latency Engineering

The shift to generative models introduces a massive latency penalty. Inference for a billion-parameter Transformer is orders of magnitude slower than the dot-product lookups of Two-Tower models. Consequently, 2024–2025 has seen intense innovation in "serving infrastructure" to make GenRec viable.

### 7.1 Meituan's Dual-Flow Generative Ranking (DFGR)

Meituan's DFGR [(DFGR, arXiv)](https://arxiv.org/html/2505.16752v3) offers a direct architectural critique of Meta's HSTU. The authors observed that HSTU's approach of interleaving items and actions doubles the input sequence length ($2N$). Since attention complexity is quadratic (or linear with significant constants), doubling the sequence length quadruples the computational load ($4N^2$ in standard attention).

**Architecture:**

DFGR decouples the input into two parallel streams: an Item Flow ($[Item_1, Item_2, \dots]$) and an Action Flow ($[Action_1, Action_2, \dots]$). These flows are processed by separate transformer stacks that communicate via cross-attention gates at each layer. This design maintains the sequence length at $N$, effectively halving the computational complexity compared to the interleaved approach. Meituan reports that DFGR achieves 2x faster training and 4x faster inference than HSTU while matching its accuracy. [(DFGR, Shaped.ai)](https://www.shaped.ai/blog/action-is-all-you-need-dual-flow-generative-ranking-network-for-recommendation)

### 7.2 Prism: Resource Disaggregation for DLRMs

For massive DLRMs, the bottleneck is often memory bandwidth and capacity rather than pure FLOPs. Prism [(Prism, USENIX NSDI'25)](https://www.usenix.org/system/files/nsdi25-yang.pdf) is a production serving system that implements Resource Disaggregation.

Prism splits the recommendation model graph into two distinct sub-graphs:

- **Memory-Intensive Sub-graph:** Contains the massive embedding tables. This runs on CPU Nodes (CNs) equipped with terabytes of RAM.
- **Compute-Intensive Sub-graph:** Contains the dense interaction layers (MLPs/Transformers). This runs on GPU Nodes (HNs).

The nodes communicate via high-speed RDMA (Remote Direct Memory Access). This architecture eliminates "hardware fragmentation"—the common scenario where a GPU server runs out of memory (due to embeddings) while its compute cores sit idle. Prism allows independent scaling of memory and compute resources, reducing total cost of ownership by over 50%. [(Prism, USENIX NSDI'25)](https://www.usenix.org/system/files/nsdi25-yang.pdf)

### 7.3 Speculative Decoding (AtSpeed)

Borrowed from LLM serving, AtSpeed [(AtSpeed, ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4bf5c3245fd92a4554a16af9803b757-Paper-Conference.pdf) applies Speculative Decoding to recommendation. In this paradigm, a smaller, faster "draft model" generates a candidate list of $K$ items. The massive "target model" then verifies these candidates in parallel. This reduces the number of serial passes required by the large model. Experiments show that AtSpeed can accelerate decoding by approximately 2x with negligible accuracy loss. [(AtSpeed, ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4bf5c3245fd92a4554a16af9803b757-Paper-Conference.pdf)

## 8. Emerging Frontiers: Uncertainty and Universal Representations

### 8.1 Epistemic Neural Networks (ENN) for Cold Start

A persistent failure mode of Deep Learning in RecSys is overconfidence. A standard neural network will output a low score for a new item simply because it lacks training data, preventing the item from ever gaining impressions (the "Cold Start" problem). Meta has begun deploying Epistemic Neural Networks (ENNs) [(ENN, arXiv)](https://arxiv.org/html/2412.04484v1) to address this.

**Mechanism:**

An ENN augments the base network with an "Epinet." The Epinet takes the input features plus a random noise vector $z$ (the epistemic index).

- For known items (where data is abundant), the Epinet learns to output zero, regardless of $z$. The model is confident.
- For new items (sparse data), the Epinet outputs values that vary wildly as $z$ changes.

This variance quantifies epistemic uncertainty (uncertainty due to lack of knowledge). By sampling $z$, the system generates a distribution of scores. This enables the implementation of Deep Thompson Sampling: the system can optimistically sample a high score from the distribution to "explore" a new item. If the user engages, the model learns; if not, the uncertainty (and score) collapses. Deployment on Facebook Reels resulted in statistically significant gains in fresh content discovery. [(ENN, arXiv)](https://arxiv.org/html/2412.04484v1)

### 8.2 Universal User Representations

The RecSys Challenge 2025 focused on developing Universal Behavioral Profiles (UBPs). [(RecSys Challenge 2025, arXiv)](https://arxiv.org/html/2508.06970v1) The goal was to create a single user embedding capable of powering multiple downstream tasks (churn prediction, propensity scoring, recommendation) without task-specific retraining.

The winning solution employed Contrastive Learning to align a user's past behavior with their future actions in a shared latent space. By training a Transformer encoder to maximize the similarity between the "Past Embedding" and "Future Embedding" of the same user (while minimizing it for different users), the model learns a robust, task-agnostic representation of user intent. This trend points toward the commoditization of user modeling, where a central "Foundation User Model" serves embeddings to all downstream applications. [(RecSys Challenge 2025, Medium)](https://medium.com/@buildsomethinggreat/inside-recsys-challenge-2025-building-universal-user-representations-for-the-future-of-081813479e64)

## 9. Conclusion: The Agentic Future

The 2024–2025 period represents a decisive maturation of the "Generative Era" in recommendation systems. The industry has successfully validated Scaling Laws for interaction data, proving that investments in compute and model size yield predictable returns in quality (Wukong, HSTU, GEM). This has justified the massive infrastructural shift toward Generative Recommendation (OneRec, PinRec, OneSearch), which captures the joint distribution of user behaviors with a fidelity that discriminative models cannot match.

However, the "free lunch" of scaling is paid for in latency. This constraint has driven the most creative engineering of the period: Dual-Flow Architectures (DFGR) to optimize attention, Lazy Decoders (OneRec) to minimize redundant compute, and Resource Disaggregation (Prism) to break the memory wall.

Looking forward, the integration of LLMs is moving beyond simple feature extraction toward Agentic Recommendation. Systems like Tencent's RARE, Kuaishou's OneRec-Think, and Shopee's OnePiece are beginning to exhibit reasoning capabilities—understanding why a user might want an item and generating intermediate semantic goals (Commercial Intentions, Rationales) to fulfill that want. The next frontier lies in systems that do not merely rank static items but actively plan, reason, and converse to fulfill complex, long-term user intents, completing the transition from "Recommender Systems" to "Personal AI Agents."

---

## Works Cited

1. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://proceedings.mlr.press/v235/zhai24a.html) - Proceedings of Machine Learning Research
2. [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (PDF)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhai24a/zhai24a.pdf) - GitHub
3. [Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation](https://arxiv.org/html/2505.16752v3) - arXiv
4. [Actions Speak Louder than Words (Abstract)](https://arxiv.org/abs/2402.17152) - arXiv
5. [Wukong: Towards a Scaling Law for Large-Scale Recommendation (Abstract)](https://arxiv.org/abs/2403.02545) - arXiv
6. [Wukong: Towards a Scaling Law for Large-Scale Recommendation (PDF)](https://arxiv.org/pdf/2403.02545) - arXiv
7. [OneRec-V2 Technical Report (Abstract)](https://arxiv.org/abs/2508.20900) - arXiv
8. [OneRec-V2 Technical Report](https://arxiv.org/html/2508.20900v1) - arXiv
9. [OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965) - arXiv
10. [PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems (v1)](https://arxiv.org/html/2504.10507v1) - arXiv
11. [PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems (v4)](https://arxiv.org/html/2504.10507v4) - arXiv
12. [TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios (Abstract)](https://arxiv.org/abs/2508.11977) - arXiv
13. [TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios](https://arxiv.org/html/2508.11977v2) - arXiv
14. [LongRetriever: Towards Ultra-Long Sequence based Candidate Retrieval for Recommendation (Abstract)](https://arxiv.org/abs/2508.15486) - arXiv
15. [LongRetriever: Towards Ultra-Long Sequence based Candidate Retrieval for Recommendation](https://arxiv.org/html/2508.15486v2) - arXiv
16. [LongRetriever (PDF)](https://www.arxiv.org/pdf/2508.15486) - arXiv
17. [Real-time Ad retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising (Abstract)](https://arxiv.org/abs/2504.01304) - arXiv
18. [Real-time Ad Retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising (PDF)](https://aclanthology.org/2025.emnlp-main.1473.pdf) - ACL Anthology
19. [Real-time Ad retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising](https://arxiv.org/html/2504.01304v1) - arXiv
20. [KDD 2024](https://www.amazon.science/conferences-and-events/kdd-2024) - Amazon Science
21. [PEMBOT: Pareto-Ensembled Multi-task Boosted Trees (PDF)](https://assets.amazon.science/e1/a8/3dea9102495d825d23fb4819641d/pembot-pareto-ensembled-multi-task-boosted-trees.pdf) - Amazon Science
22. [Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark](https://arxiv.org/html/2412.17374v1) - arXiv
23. [MTmixAtt: Integrating Mixture-of-Experts with Multi-Mix Attention for Large-Scale Recommendation](https://arxiv.org/html/2510.15286v1) - arXiv
24. [An Adaptive Entire-Space Multi-Scenario Multi-Task Transfer Learning Model for Recommendations](https://www.computer.org/csdl/journal/tk/2025/04/10858412/23VPw4QCkrm) - IEEE Computer Society
25. [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations (Abstract)](https://arxiv.org/abs/2510.07784) - arXiv
26. [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://www.researchgate.net/publication/396373037_PLUM_Adapting_Pre-trained_Language_Models_for_Industrial-scale_Generative_Recommendations) - ResearchGate
27. [PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/html/2510.07784v1) - arXiv
28. [Generating Long Semantic IDs in Parallel for Recommendation](https://www.researchgate.net/publication/392514863_Generating_Long_Semantic_IDs_in_Parallel_for_Recommendation) - ResearchGate
29. [Generating Long Semantic IDs in Parallel for Recommendation (PDF)](https://arxiv.org/pdf/2506.05781) - arXiv
30. [OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System](https://arxiv.org/html/2509.18091v1) - arXiv
31. [Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation](https://www.shaped.ai/blog/action-is-all-you-need-dual-flow-generative-ranking-network-for-recommendation) - Shaped.ai
32. [GPU-Disaggregated Serving for Deep Learning Recommendation Models at Scale](https://www.usenix.org/system/files/nsdi25-yang.pdf) - USENIX
33. [Efficient Inference for Large Language Model-based Generative Recommendation](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4bf5c3245fd92a4554a16af9803b757-Paper-Conference.pdf) - ICLR Proceedings
34. [Epinet for Content Cold Start](https://arxiv.org/html/2412.04484v1) - arXiv
35. [Epistemic Artificial Intelligence is Essential for Machine Learning Models to Truly 'Know When They Do Not Know'](https://starlab.ewi.tudelft.nl/papers/r47.pdf) - STAR Lab
36. [Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025](https://arxiv.org/html/2508.06970v1) - arXiv
37. [Inside RecSys Challenge 2025: Building Universal User Representations for the Future of Recommendations](https://medium.com/@buildsomethinggreat/inside-recsys-challenge-2025-building-universal-user-representations-for-the-future-of-081813479e64) - Medium