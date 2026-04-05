From Static Identifiers to Dynamic Semantics: The Architectural Evolution of Context-Aware Recommender Systems
1. Introduction: The Semantic Gap
The trajectory of machine learning for natural language processing (NLP) over the last decade has been defined by a singular, pivotal transition: the abandonment of static representations in favor of dynamic, context-dependent function approximation. For nearly thirty years, the field struggled with the limitations of assigning fixed vectors to variable linguistic units. The resolution of this struggle—culminating in the development of ELMo, BERT, and GPT—did not merely improve benchmarks; it fundamentally redefined the unit of analysis from the "word" to the "token-in-context."
In stark contrast, the field of Recommender Systems (RecSys) remains largely entrenched in the architectural paradigms of the pre-BERT era. Despite the massive proliferation of deep learning within recommendation pipelines, the foundational atomic unit of the vast majority of industrial systems remains the static Item ID embedding. Whether in Matrix Factorization, Neural Collaborative Filtering, or Two-Tower architectures, items are typically treated as immutable points in a latent vector space. This architectural inertia persists despite the fact that items, much like words, exhibit profound polysemy. A single item—be it a movie, a news article, or a consumer product—can embody distinct semantic roles depending entirely on the user interacting with it and the temporal context of that interaction.
This report serves as a comprehensive investigation into this semantic gap. It begins by rigorously deconstructing the mechanisms by which NLP overcame the polysemy of language, using this historical lens to diagnose the current limitations of Recommender Systems. It documents the technical constraints—specifically latency, memory bandwidth, and the indexability requirement—that have forced RecSys to cling to static embeddings. Subsequently, it surveys the emerging frontier of Hypernetworks as a mechanism for dynamic parameter generation, analyzing their potential to bridge the semantic gap and their inherent scalability challenges. Finally, responding to the need for a system capable of handling 100 million items with robust cold-start handling, the report proposes a novel, hybrid architecture: the User-Conditioned Generative Hypencoder (UCGH). This design synthesizes hierarchical semantic quantization (TIGER) with hypernetwork-driven dynamic reranking to create a system where item representations are fluid, personalized, and theoretically infinite.
2. The Resolution of Polysemy in Natural Language Processing
To understand the future of item representation, one must first dissect the resolution of word polysemy in NLP. This transition was not a linear improvement but a paradigm shift in how meaning is encoded in high-dimensional spaces.
2.1 The Era of Static Isotropy: Word2Vec and GloVe
Prior to 2018, the dominant methodology for representing language involved static word embeddings, exemplified by Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014). These models relied on the distributional hypothesis—that words appearing in similar contexts share semantic meaning—to map the vocabulary $V$ to a real-valued vector space $\mathbb{R}^d$.1
2.1.1 The Manifold Hypothesis and Isotropy
The underlying assumption of static embeddings is that the semantic manifold is static. The word "bank" is assigned a single vector $v_{bank} \in \mathbb{R}^{300}$. This vector represents the weighted centroid of all observed usages of the word in the training corpus. If "bank" appears 50% of the time in the context of "river" and 50% in the context of "money," the resulting vector $v_{bank}$ resides in a semantic "no-man's land" halfway between the geological and financial clusters.
This forced isotropy creates a fundamental bottleneck. In downstream tasks, a model receiving $v_{bank}$ must expend capacity to disambiguate which sense is active, often relying on the surrounding bag-of-words. Research indicates that in these static models, the vector space is "crowded" in the middle, and less than 5% of the variance in a word's usage is explained by its static embedding.1 The model effectively compresses the richness of human language into a lookup table, discarding the nuance of polysemy in favor of computational efficiency.
2.2 ELMo: The Concatenation of Sequential States
The first rupture in the static paradigm came with ELMo (Embeddings from Language Models) in 2018. ELMo challenged the lookup table by asserting that a word embedding should not be a stored parameter, but a computed function of the input sentence.1
2.2.1 Deep Bidirectional LSTM Mechanics
ELMo utilizes a deep bidirectional Long Short-Term Memory (biLSTM) architecture. For a given token sequence $t_1,..., t_N$, ELMo computes a context-dependent representation by concatenating the hidden states of a forward LSTM (reading $t_1 \rightarrow t_N$) and a backward LSTM (reading $t_N \rightarrow t_1$).


$$R_k = \{x_k^{LM}, \vec{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} | j=1,...,L\}$$

The final embedding is a scalar combination of these layers.1
2.2.2 Polysemy Resolution via State Evolution
The mechanism of polysemy resolution in ELMo is sequential state evolution. When the forward LSTM processes "The river bank," the hidden state at the step "bank" carries the accumulated memory of "The" and "river." This biases the vector representation heavily toward the geological sense. Conversely, in "The bank deposit," the hidden state carries the signal of "deposit." By creating representations that are structurally coupled to their history, ELMo ensures that the vector for "bank" is effectively unique in every sentence. Empirically, the nearest neighbors of "bank" in ELMo's space shift dynamically depending on the sentence, a property impossible in Word2Vec.4
2.3 BERT: Bidirectional Attention and Anisotropy
While ELMo introduced context, it was limited by the sequential nature of LSTMs, which struggle with long-range dependencies. BERT (Bidirectional Encoder Representations from Transformers) replaced recurrence with the Transformer architecture, utilizing self-attention to capture global context simultaneously.5
2.3.1 The Attention Mechanism as Polysemy Solver
The core engine of BERT is the Scaled Dot-Product Attention:


$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In this framework, every token in the sequence acts as a Query ($Q$) that attends to all other tokens as Keys ($K$). The resulting attention weights determine how much information to aggregate from the Values ($V$).
This mechanism allows for non-local polysemy resolution. In the sentence "The bank, which had been closed for renovations since the flood, finally reopened," the word "bank" can attend directly to "flood" (resolving the geological/structural sense) despite the long distance, without the signal degradation inherent in LSTMs.4
2.3.2 Masked Language Modeling (MLM) and Anisotropy
BERT's training objective, Masked Language Modeling (MLM), forces the model to predict missing words based on context. This objective is the ultimate teacher of polysemy. To predict "deposit" in "He went to the to make a deposit," the model must construct a representation of the context that specifically demands a financial institution.
Research analyzing the geometry of BERT's latent space reveals that it is highly anisotropic.1 Unlike the spherical clusters of Word2Vec, BERT's contextual embeddings form narrow cones. The representation of a word is pulled drastically toward the specific meaning required by the sentence. The cosine similarity between the "bank" (finance) and "bank" (river) vectors in BERT is low, often lower than the similarity between "bank" (finance) and "money." This confirms that BERT does not just "handle" polysemy; it effectively treats distinct senses of a word as distinct words.1
2.4 GPT: Autoregressive Contextualization
GPT (Generative Pre-trained Transformer) adopts a decoder-only architecture. While similar to BERT in its use of self-attention, its constraint is unidirectional (it can only attend to previous tokens).4
Mechanism: GPT resolves polysemy through predictive coherence. By learning to predict the next token, the model implicitly learns to maintain a coherent narrative state. If the previous context describes a financial crisis, the embedding for the next occurrence of "bank" is conditioned by that narrative arc.
Scale: GPT demonstrated that polysemy resolution scales with model depth. In deeper layers of GPT-2 and GPT-3, the contextualization is so strong that the original "static" meaning of the word contributes less than 5% to the variance of the vector, with the remaining 95% determined by the context.1
Conclusion on NLP: The field solved polysemy by accepting that a static vector $v_w$ is an insufficient compression of information. The solution was to replace $v_w$ with a function $f_\theta(w, c)$, where $\theta$ are the weights of a massive neural network and $c$ is the context. This shift allows for infinite representational capacity for a finite vocabulary.
3. The Static Stagnation of Recommender Systems
While NLP has embraced dynamic contextualization, the vast majority of industrial recommender systems remain anchored in the "Static Era." The standard workflow—exemplified by Collaborative Filtering (CF), Matrix Factorization (MF), and Two-Tower architectures—mirrors the logic of Word2Vec, not BERT.6
3.1 Defining "Item Polysemy" in Recommendation
The concept of polysemy applies directly to items in a catalog. An item is rarely a monolithic semantic entity; it is a bundle of latent attributes that manifest differently depending on the observer.8
The Multifaceted Object: Consider a film like Inception. To User A (a sci-fi fan), the item represents "Hard Sci-Fi/Space." To User B (a fan of the lead actor), the item represents "Leonardo DiCaprio." To User C (an intellectual thriller fan), it represents "Complex Plot."
The Interaction Polysemy: The "meaning" of the item is defined by the intent of the interaction. A user buying a tent for a backyard sleepover interacts with a "toy." A user buying the same tent for an Everest expedition interacts with "survival gear."
The Static Failure: A static embedding $v_{inception}$ must minimize the reconstruction loss across all these users. Mathematically, this forces the vector to be the centroid of these divergent clusters. The resulting representation captures the "average" meaning, diluting the specific signals that drove the interaction for any individual user.9 This is identical to the Word2Vec "bank" problem.
3.2 Why RecSys Still Uses Static Embeddings
The persistence of static embeddings is not due to a lack of theoretical understanding but due to severe constraints on latency, throughput, and scale.
3.2.1 The Latency-Scale Trade-off
NLP models typically operate on sequences of 512 to 8,000 tokens. A recommender system, however, must select the best item from a catalog of 100 million to billions of items.
The Indexing Requirement: To retrieve items in milliseconds, RecSys relies on Maximum Inner Product Search (MIPS). This requires representing all items as fixed vectors in a metric space, which are then indexed using Approximate Nearest Neighbor (ANN) structures like HNSW (Hierarchical Navigable Small World graphs) or Faiss.11
The MIPS Bottleneck: MIPS fundamentally assumes that the item vector $v_i$ is constant. If $v_i$ changes depending on the user $u$, the index becomes invalid. We cannot rebuild an HNSW index for every user query (a process that takes hours). Therefore, dynamic embeddings are incompatible with standard ANN retrieval, forcing systems to rely on static IDs for the candidate generation phase.6
3.2.2 The Memory Bottleneck of Embedding Tables
Industrial RecSys models are characterized by massive embedding tables. A system with 100 million items and an embedding dimension of 256 requires storing a lookup table of size $100 \times 10^6 \times 256 \times 4 \text{ bytes} \approx 100 \text{ GB}$.
Scalability Limit: This table must reside in GPU memory or high-bandwidth parameter servers for fast training. The sheer size of this static table is a primary bottleneck in scaling recommender systems.14 Moving to a dynamic model (where vectors are computed) could theoretically alleviate this if the computation is cheaper than the memory bandwidth—but typically, the computation of a BERT-level model for 100M items is exorbitantly expensive.15
3.2.3 The False Dawn of Sequential Recommendation
Models like SASRec (Self-Attentive Sequential Recommendation) and BERT4Rec utilize Transformer architectures, leading to a common misconception that they have solved the polysemy problem.16
The Misconception: These models apply the Transformer to the User History, not the Item Representation. They contextualize the sequence of interactions to produce a dynamic User Vector $v_u$.
The Reality: While the user vector is dynamic, the target item vectors $v_i$ against which $v_u$ is dot-producted are still drawn from a static embedding table.18 The model learns to match a dynamic user state to a static item state. It does not allow the item itself to morph. If Inception is represented as a "Sci-Fi" vector in the static table, the SASRec model can only match it with users looking for Sci-Fi, failing to effectively match it with the user looking for "DiCaprio" if that signal was lost in the centroid averaging of the static item embedding.20
4. Hypernetworks: The Mechanism of Dynamic Semantics
To break the static deadlock and bring NLP-level polysemy resolution to RecSys, researchers are increasingly turning to Hypernetworks. This class of architecture offers a mathematical pathway to generate parameters dynamically, effectively creating a "function-based" representation similar to ELMo/BERT but adapted for the constraints of recommendation.
4.1 Theoretical Foundations of Hypernetworks
A Hypernetwork is defined as a neural network $H$ that generates the weights $\theta$ for another neural network, referred to as the "Primary" or "Target" network $M$.21
Formally, if the primary network computes $y = M(x; \theta)$, a standard approach learns $\theta$ as static parameters. A hypernetwork approach learns $\phi$ (the weights of $H$) such that:


$$\theta = H(c; \phi)$$

$$y = M(x; H(c; \phi))$$

Where $c$ is a context vector. In RecSys, $c$ is typically the user embedding. This means the weights of the recommendation model are not fixed; they are generated on the fly for each user.21
4.2 Hypernetworks in Recommendation Literature
Several key approaches have applied this principle to address varying aspects of the static limitation.
4.2.1 HyperRS: Solving the Parameter Cold Start
HyperRS (Hypernetwork-based Recommender System) utilizes hypernetworks to address the user cold-start problem.23
Mechanism: In standard meta-learning (e.g., MAML), the model attempts to find a good initialization that can be fine-tuned. HyperRS goes further by using a hypernetwork to generate the user-specific weights of a recommendation model based on a very small support set of interactions.
Polysemy Implication: By generating user-specific weights, HyperRS effectively tailors the decision boundary of the model to the user's specific interpretation of items. The hypernetwork learns the manifold of possible user preference functions and maps the specific user to a point on that manifold.23
4.2.2 The Hypencoder: Dynamic Retrieval Functions
The "Hypencoder" (Hyper-Encoder) represents a radical departure from the "Two-Tower" dot product paradigm. Instead of encoding the query (user) and document (item) into a shared vector space, the Hypencoder uses the query to generate a scoring function.25
Architecture:
Query Encoder: A Transformer processes the user/query history.
Hyperhead: A projection layer takes the query embedding and outputs the weights $W_q, b_q$ for a small MLP.
Inference: This generated MLP is applied to the document/item embedding: $Score = \text{MLP}(v_{doc}; W_q, b_q)$.
Resolving Polysemy: This architecture allows the user to fundamentally alter how the item is perceived. If the generated MLP has high weights for dimensions 10-20 (visual features) and low weights for dimensions 30-40 (textual features), the system is dynamically prioritizing the visual aspect of the item for that specific user. The item's "meaning" (its score) is a non-linear interaction between the user's generated function and the item's features.25
4.3 Scalability Challenges of Hypernetworks
While Hypernetworks solve the representational rigidity, they introduce significant scalability hurdles that have prevented their widespread adoption in 100M+ item systems.11

Challenge
Description
Implication for 100M Items
Inference Latency
Applying a generated MLP to an item is computationally heavier than a dot product ($O(d^2)$ vs $O(d)$).
Scoring 100M items with an MLP is infeasible within the 100ms latency budget of real-time systems.
Indexing Incompatibility
Because the scoring function $f_u(v_i)$ is non-linear and user-dependent, the triangle inequality does not hold.
Standard ANN algorithms (HNSW) cannot be used. This forces a brute-force scan, which is $O(N)$.
Memory Bandwidth
The Hypernetwork must generate and transfer a weight matrix (e.g., $64 \times 64$) for every user.
High memory bandwidth consumption during batched inference, potentially bottlenecking GPU throughput.
Optimization Instability
Hypernetworks are essentially learning a "loss landscape." They are prone to mode collapse or generating "average" weights that fail to specialize.
Training requires careful normalization (e.g., LayerNorm on generated weights) and specialized initialization schemes.27

4.4 The "Late Interaction" Alternative (ColBERT)
A parallel development in resolving polysemy without full hypernetworks is the "Late Interaction" paradigm, popularized by ColBERT.28
Multi-Vector Representation: Instead of compressing an item into one vector, ColBERT represents it as a bag of vectors (one per token/patch).
MaxSim Retrieval: Interaction is computed by summing the maximum similarity of each query token to the document tokens.
Relation to Polysemy: This preserves the discrete semantic units of the item. A "sci-fi" token in the item can match the "sci-fi" token in the user query, while the "romance" token remains ignored. It avoids the centroid averaging problem.
Scalability: While ColBERT scales better than full BERT cross-encoders, the storage cost of multi-vector embeddings is massive (increasing index size by 100x), making it challenging for 100M items without extreme quantization.30
5. Architectural Design: The User-Conditioned Generative Hypencoder (UCGH)
To satisfy the dual requirements of resolving item polysemy via user-conditioning and scaling to 100M+ items, we cannot rely on a single existing architecture. We must synthesize the scalable indexing of Generative Retrieval with the dynamic expressivity of Hypernetworks.
We propose the User-Conditioned Generative Hypencoder (UCGH). This architecture eliminates the static embedding table entirely, replacing it with a content-derived generative index and a hypernetwork-driven reranker.
5.1 System Overview
The UCGH consists of three coupled subsystems:
The Semantic Item Encoder (SIE): Handles cold start and compression using hierarchical quantization (Semantic IDs).
The Generative Retrieval Core (TIGER): Handles the 100M scale by narrowing the search space via autoregressive decoding.
The Hypernetwork Reranker (Hypencoder): Handles item polysemy by generating user-specific scoring functions for the retrieved candidates.
5.2 Component 1: The Semantic Item Encoder (SIE)
To handle 100M items and solve the cold-start problem, we abandon the concept of "Atomic IDs" (random integers assigned to items). Instead, we adopt Semantic IDs derived from the item's content (text, image, metadata).32
5.2.1 Content-Conditioned Initialization
Unlike traditional CF which initializes embeddings randomly, the SIE starts with raw content.
Encoders: We utilize frozen, pre-trained encoders: a Vision Transformer (ViT) for product images and a Sentence-BERT (SBERT) for textual descriptions.
Fusion: These representations are concatenated and projected to a dense vector $v_{content} \in \mathbb{R}^{d}$.
Cold Start Solution: Because $v_{content}$ is derived solely from metadata, a brand-new item immediately possesses a high-fidelity representation before a single user interaction occurs. This creates a "Zero-Shot" recommendation capability.15
5.2.2 Hierarchical Quantization (RQ-VAE)
Storing 100M dense vectors ($100 \times 10^6 \times 768 \times 4$ bytes $\approx 300$ GB) is inefficient. We employ a Residual-Quantized Variational AutoEncoder (RQ-VAE) to compress this space.32
Mechanism: The RQ-VAE learns a codebook $C$ of discrete latent vectors. It recursively quantizes the content vector $v_{content}$ into a tuple of discrete indices $(c_1, c_2, c_3)$.
$c_1$: Represents the coarse semantic cluster (e.g., "Electronics").
$c_2$: Represents the sub-category (e.g., "Headphones").
$c_3$: Represents the fine-grained detail (e.g., "Noise-canceling").
The Semantic ID: The item is now identified by the tuple (124, 55, 12). This tuple is the embedding. Storing 100M tuples of 3 integers is trivial ($\approx 1.2$ GB), solving the memory bottleneck.
5.3 Component 2: Scalable Retrieval via TIGER (Generative Retrieval)
To retrieve from 100M items without an $O(N)$ scan, we use the TIGER (Transformer Index for Generative Recommenders) paradigm.32
User Encoding: A Transformer (SASRec backbone) processes the user's history of Semantic IDs.
Generative Decoding: Instead of predicting a dot product, the model autoregressively predicts the next Semantic ID tuple.
Step 1: Predict $P(c_1 | H_{user})$. The model predicts the broad category the user is interested in.
Step 2: Predict $P(c_2 | H_{user}, c_1)$. The model refines the intent.
Beam Search Retrieval: By performing beam search (e.g., beam width = 20), the model navigates the hierarchical semantic tree. It effectively "hallucinates" the ID of the item the user wants.
The Cluster Effect: The output of this phase is not a single item, but a set of Semantic ID prefixes (e.g., "Items starting with 124, 55"). This effectively retrieves a candidate set of perhaps 10,000 items that share these semantic characteristics.
Scalability: This retrieval is $O(\log N)$ ( logarithmic with respect to catalog size due to the tree structure of Semantic IDs), making it feasible for 100M items.
5.4 Component 3: Dynamic Polysemy Resolution via Hypencoder
Once the candidate set is reduced to ~10,000 items, we apply the User-Conditioned Hypernetwork to resolve polysemy and rank the items.25
5.4.1 The Hypernetwork Projection
We utilize the same User Transformer state $u_{state}$ from the retrieval phase.
Weight Generation: A projection head (Hypernet) maps $u_{state}$ to the parameters of a small scoring MLP (e.g., two layers, hidden dim 64).

$$\theta_{user} = \text{HyperNet}(u_{state})$$

To ensure efficiency, we use Low-Rank Factorization for the weight generation. Instead of generating a $d \times d$ matrix, we generate vectors that form the rank-1 or rank-2 approximation of the weights, significantly reducing memory bandwidth.22
5.4.2 Dynamic Scoring
For each of the 10,000 candidate items:
Reconstruction: Look up the codebook vectors for the item's Semantic ID $(c_1, c_2, c_3)$ and sum them to approximate the content vector $\hat{v}_{content}$.
Conditional Execution: Apply the generated MLP:

$$Score_{i,u} = \text{MLP}(\hat{v}_{content}; \theta_{user})$$
Ranking: Sort by score.
Why this solves Polysemy: The MLP weights $\theta_{user}$ constitute a user-specific "lens." If the user prefers visual aesthetics, the Hypernetwork will generate weights that amplify the visual dimensions of the content vector $\hat{v}_{content}$. If the user prefers textual/narrative depth, the weights will shift to amplify SBERT-derived dimensions. The item's representation is no longer static; it is a fluid variable in the user's generated function.
5.5 Comparison with Traditional Architectures
The following table contrasts the proposed UCGH architecture with standard approaches across key dimensions.
Feature
Two-Tower (Current Standard)
TIGER (Base Generative)
Proposed UCGH Architecture
Item Representation
Static Atomic ID Vector
Semantic ID (Tuple)
Semantic ID + Dynamic Function
Polysemy Handling
None (Centroid averaging)
Implicit (Sequence context)
Explicit (User-generated weighting)
Retrieval Mechanism
MIPS (HNSW/Faiss)
Autoregressive Beam Search
Generative Recall + Hypernet Rank
100M Scalability
Low (Huge Embedding Tables)
High (Quantized Codebooks)
High (Quantized + Local Ranking)
Cold Start
Fails (Needs retraining)
Good (Content-based)
Excellent (Zero-shot + Adapted)
Inference Cost
Low ($O(1)$)
Medium (Beam Search)
High but Manageable (Search + MLP)

6. Implementation Strategy and Future Outlook
Implementing the UCGH requires a shift in infrastructure from "Storage-Heavy" (serving massive tables) to "Compute-Heavy" (inference-time generation).
6.1 Training Pipeline
The system requires a two-stage training process:
Stage 1: Semantic Quantization. Train the RQ-VAE on the corpus of item content (images/text) to learn the codebooks and Semantic IDs. This effectively indexes the 100M items into the hierarchical structure.
Stage 2: End-to-End Generative Modeling. Train the User Transformer and Hypernetwork jointly. The loss function is a hybrid of:
Generative Loss: Negative Log-Likelihood of predicting the correct Semantic ID tokens (for Retrieval).
Ranking Loss: Contrastive loss (e.g., InfoNCE) using the Hypernetwork-generated scores to distinguish the ground-truth item from in-batch negatives (for Reranking).20
6.2 Hardware Implications
The shift to UCGH moves the bottleneck from Memory Capacity (VRAM size for embedding tables) to Compute Capability (TFLOPs for Transformer/Hypernet inference).
TPU/GPU Utilization: Traditional sparse embedding lookups utilize GPUs poorly (memory-bound). The UCGH is dense and compute-bound, allowing for much higher utilization of modern hardware like NVIDIA H100s or TPUs, which excel at matrix multiplications.15
6.3 Conclusion
The "static stagnation" of Recommender Systems is a solvable artifact of legacy constraints. By analyzing the trajectory of NLP—from Word2Vec's static isotropy to BERT's dynamic anisotropy—we identify the necessary evolution for RecSys: the transition to user-conditioned item representations.
The architecture proposed herein, the User-Conditioned Generative Hypencoder, bridges this gap. It conquers the scale of 100M items through the compression of Semantic IDs (TIGER) and conquers the nuance of item polysemy through the plasticity of Hypernetworks. This design does not merely improve accuracy; it fundamentally realigns the mathematical structure of recommendation with the complex, multifaceted nature of human preference, moving from a rigid matching of IDs to a dynamic generation of meaning.
Works cited
How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings - ACL Anthology, accessed January 5, 2026, https://aclanthology.org/D19-1006.pdf
Natural Language Processing • Word Vectors/Embeddings - aman.ai, accessed January 5, 2026, https://aman.ai/primers/ai/word-vectors/
From Word Vectors to Multimodal Embeddings: Techniques, Applications, and Future Directions For Large Language Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2411.05036v2
BERT: A Comprehensive Guide to the Groundbreaking NLP Framework - Analytics Vidhya, accessed January 5, 2026, https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/
BERT (language model) - Wikipedia, accessed January 5, 2026, https://en.wikipedia.org/wiki/BERT_(language_model)
Embedding in Recommender Systems: A Survey - arXiv, accessed January 5, 2026, https://arxiv.org/html/2310.18608v3
Embedding in Recommender Systems: A Survey - arXiv, accessed January 5, 2026, https://arxiv.org/html/2310.18608v2
The Representation of Polysemy: MEG Evidence - PMC - NIH, accessed January 5, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC1351340/
Node-Polysemy Aware Recommendation by Matrix Completion with Side Information, accessed January 5, 2026, https://par.nsf.gov/servlets/purl/10331959
CONTEXTGNN: Beyond Two-Tower Recommendation Systems - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/2411.19513
How can you handle scalability issues in recommender systems? - Milvus, accessed January 5, 2026, https://milvus.io/ai-quick-reference/how-can-you-handle-scalability-issues-in-recommender-systems
What is nearest neighbor search in embeddings? - Milvus, accessed January 5, 2026, https://milvus.io/ai-quick-reference/what-is-nearest-neighbor-search-in-embeddings
Accelerating Approximate Nearest Neighbor Search in Hierarchical Graphs: Efficient Level Navigation with Shortcuts - VLDB Endowment, accessed January 5, 2026, https://www.vldb.org/pvldb/vol18/p3518-chen.pdf
Training a recommendation model with dynamic embeddings - The TensorFlow Blog, accessed January 5, 2026, https://blog.tensorflow.org/2023/04/training-recommendation-model-with-dynamic-embeddings.html
From IDs to Meaning: The Case for Semantic Embeddings in Recommendation - CueZen, accessed January 5, 2026, https://cuezen.com/from-ids-to-meaning-the-case-for-semantic-embeddings-in-recommendation/
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2502.08132v1
Beyond Static Preferences: Understanding Sequential Models in Recommendation Systems (N-Gram, SASRec, BERT4Rec & Beyond) - Shaped.ai, accessed January 5, 2026, https://www.shaped.ai/blog/beyond-static-preferences-understanding-sequential-models-in-recommendation-systems-n-gram-sasrec-bert4rec-beyond
Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? - alphaXiv, accessed January 5, 2026, https://www.alphaxiv.org/overview/2309.07602v1
RecSys Transformer Models Tutorial - RecTools documentation, accessed January 5, 2026, https://rectools.readthedocs.io/en/latest/examples/tutorials/transformers_tutorial.html
Sequential retrieval using SASRec - Keras, accessed January 5, 2026, https://keras.io/keras_rs/examples/sas_rec/
Hypernetworks — A novel way to initialize weights | by Void - Medium, accessed January 5, 2026, https://medium.com/@atulit23/hypernetworks-a-novel-way-to-initialize-weights-e7584385488d
Casper : Cascading Hypernetworks for Scalable Continual Learning - OpenReview, accessed January 5, 2026, https://openreview.net/forum?id=Vhlwxfs3pH
(PDF) HyperRS: Hypernetwork-Based Recommender System for the User Cold-Start Problem - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/367101049_HyperRS_Hypernetwork-based_Recommender_System_for_the_User_Cold-Start_Problem
HyperRS: Hypernetwork-Based Recommender ... - IEEE Xplore, accessed January 5, 2026, https://ieeexplore.ieee.org/iel7/6287639/10005208/10015724.pdf
arxiv.org, accessed January 5, 2026, https://arxiv.org/html/2502.05364v2
Hypencoder: Hypernetworks for Information Retrieval - arXiv, accessed January 5, 2026, https://arxiv.org/html/2502.05364v1
HyperVLA: Efficient Inference in Vision-Language-Action Models via Hypernetworks, accessed January 5, 2026, https://openreview.net/forum?id=bsXkBTZjgY
An Overview of Late Interaction Retrieval Models: ColBERT, ColPali, and ColQwen, accessed January 5, 2026, https://weaviate.io/blog/late-interaction-overview
The Late Interaction Paradigm Explained: ColBERT's Secret to ..., accessed January 5, 2026, https://medium.com/@pepito_45426/the-late-interaction-paradigm-explained-colberts-secret-to-efficient-and-scalable-rag-f72f79d931ac
PyLate: Flexible Training and Retrieval for Late Interaction Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2508.03555v1
ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | Continuum Labs, accessed January 5, 2026, https://training.continuumlabs.ai/knowledge/vector-databases/colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert
TIGER: Transformer Index for Generative Recommenders - Emergent Mind, accessed January 5, 2026, https://www.emergentmind.com/topics/transformer-index-for-generative-recommenders-tiger
Recommender Systems with Generative Retrieval - Shashank Rajput, accessed January 5, 2026, https://shashankrajput.github.io/Generative.pdf
Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/2411.11225
Recommender Systems with Generative Retrieval | ChatSlide, accessed January 5, 2026, https://www.chatslide.ai/shared/recommender-systems-with-generative-retriev-RKOOjC
