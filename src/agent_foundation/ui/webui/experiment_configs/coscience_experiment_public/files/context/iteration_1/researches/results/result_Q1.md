Efficient Sequence Modeling in Recommender Systems: A Comprehensive Analysis of Linear-Time Architectures
1. Introduction
The paradigm of sequential recommendation has fundamentally shifted the focus of recommender systems (RS) from static user-item matrix completion to dynamic sequence modeling. By treating user interactions as ordered sequences—$S_u = \{i_1, t_1, i_2, t_2,..., i_N, t_N\}$—modern systems aim to capture the evolving preferences, cyclical patterns, and immediate intent of users. For the better part of the last decade, the Transformer architecture, underpinned by the self-attention mechanism, has established itself as the hegemonic standard for this task. Models such as SASRec (Self-Attentive Sequential Recommendation) and BERT4Rec (Bidirectional Encoder Representations from Transformers) leverage the global receptive field of attention to model dependencies between any two interactions, regardless of their temporal distance.1
However, the success of the Transformer is inextricably linked to a significant computational liability: the quadratic complexity of the attention mechanism. The calculation of the attention matrix $A = \text{softmax}(QK^T / \sqrt{d})$ requires $O(N^2)$ time and memory with respect to the sequence length $N$. This quadratic scaling imposes a "context ceiling" in production environments. To meet strict latency service-level agreements (SLAs) (often $<10$ms for inference), engineering teams routinely truncate user histories to the most recent 50 or 100 interactions, discarding potentially valuable long-term signals such as seasonal interests or multi-year preference evolution.3 Furthermore, the $O(N)$ memory footprint of the Key-Value (KV) cache during autoregressive generation grows linearly, consuming substantial GPU VRAM and limiting batch sizes during inference.1
A recent surge in efficient sequence modeling research has challenged this quadratic status quo. A new class of architectures, characterized by linear time complexity $O(N)$ and constant inference memory $O(1)$, promises to match or exceed Transformer quality while enabling the processing of sequences spanning thousands or millions of tokens. This report investigates these breakthroughs, focusing on State Space Models (SSMs) (specifically Mamba and Mamba-2), Linear Attention variants, xLSTM, and RWKV. We analyze their theoretical underpinnings, specifically the Structured State Space Duality (SSD) that unifies them, and provide a detailed survey of their adaptation into recommender systems through models like Mamba4Rec, SIGMA, SSD4Rec, TiM4Rec, and SS4Rec. Finally, we propose architectural strategies for the production deployment of these techniques to replace legacy Transformer backbones.
2. Theoretical Foundations: From Recurrence to Duality
To understand the efficacy of modern linear-time models, one must first deconstruct the mathematical mechanisms that allow them to bypass the $O(N^2)$ bottleneck of attention while avoiding the "forgetting" problems of traditional Recurrent Neural Networks (RNNs).
2.1 The Limits of Traditional Recurrence
Traditional RNNs (e.g., LSTMs, GRUs) model sequences via a recurrent state update $h_t = f(h_{t-1}, x_t)$. While this formulation offers $O(N)$ training and $O(1)$ inference, it suffers from two critical flaws:
Sequentiality: The hidden state $h_t$ depends on $h_{t-1}$, preventing parallelization across the time dimension during training. This underutilizes modern GPUs, which thrive on massive parallelism.
The Vanishing Gradient: Information from early time steps exponentially decays as it propagates through non-linearities, making it difficult to model long-range dependencies—a critical requirement for capturing long-term user interests in recommendation.1
2.2 State Space Models (SSMs)
Structured State Space Models (SSMs) like S4 (Structured State Space Sequence) emerged as a solution that combines the parallel training of CNNs/Transformers with the efficient inference of RNNs.
2.2.1 Continuous-Time Formulation
SSMs originate from control theory, modeling a sequence as a continuous signal $x(t)$ processed by a latent state $h(t)$:


$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$

$$y(t) = \mathbf{C}h(t)$$

Here, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the state transition matrix, and $\mathbf{B}, \mathbf{C}$ project the input and state to appropriate dimensions.1 The matrix $\mathbf{A}$ controls the evolution of the system; if designed correctly (e.g., using HiPPO matrices), it can memorize history over long durations.
2.2.2 Discretization for Deep Learning
To process discrete user interactions, the continuous system is discretized using a Zero-Order Hold (ZOH) assumption (assuming the input is constant between time steps $\Delta$). The discretized parameters become:


$$\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$$

$$\bar{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

This yields the recurrent form:


$$h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}x_t$$

$$y_t = \mathbf{C}h_t$$

Crucially, if the system is Linear Time-Invariant (LTI)—meaning $\mathbf{A}, \mathbf{B}, \mathbf{C}, \Delta$ are constant across all time steps—the operation can be viewed as a discrete convolution. The output $y$ is the convolution of the input $x$ with a filter kernel $\bar{\mathbf{K}}$ derived from the state matrices. This allows training to be parallelized using Fast Fourier Transforms (FFT), achieving $O(N \log N)$ complexity.1
2.3 The Selection Mechanism: Mamba's Breakthrough
While LTI SSMs (like S4) are efficient, they lack content-awareness. In a recommendation context, not all interactions are equally important; a user casually browsing unrelated items should not update the core interest state as heavily as a purchase event. LTI models, with fixed $\mathbf{A}$ and $\mathbf{B}$ matrices, cannot selectively ignore inputs or reset their state based on context.
Mamba (Mamba-1) introduced the Selective SSM, where the parameters become functions of the input $x_t$:


$$\mathbf{B}_t = \text{Linear}(x_t)$$

$$\mathbf{C}_t = \text{Linear}(x_t)$$

$$\Delta_t = \text{Softplus}(\text{Parameter}(x_t))$$

This input-dependence ($\mathbf{B}_t, \mathbf{C}_t, \Delta_t$) breaks the convolution equivalence, meaning FFTs can no longer be used. To resolve this, Mamba introduced a hardware-aware parallel scan algorithm (specifically, a prefix-sum scan) that computes the recurrent state updates in parallel on GPUs $O(N)$, leveraging the memory hierarchy to avoid high-latency HBM (High Bandwidth Memory) access. This mechanism allows Mamba to perform "selective copying" and "induction head" behaviors previously exclusive to Transformers.1
2.4 Structured State Space Duality (SSD)
The most recent theoretical advancement, introduced with Mamba-2, is Structured State Space Duality (SSD). This framework proves a mathematical equivalence between SSMs and a variant of Linear Attention, bridging the gap between RNNs and Transformers.
The duality posits that the lower-triangular matrix $\mathbf{M}$ used in sequence mixing (where $Y = \mathbf{M}X$) can be decomposed in two ways:
Recurrent View (SSM): $\mathbf{M}$ is generated by the sequential update $h_t = \mathbf{A}_t h_{t-1} + \mathbf{B}_t x_t$.
Attention View (Matrix Mixer): $\mathbf{M}_{ij} = \mathbf{C}_i^T (\prod_{k=j+1}^{i} \mathbf{A}_k) \mathbf{B}_j$.
Specifically, Mamba-2 restricts the $\mathbf{A}$ matrix to be scalar-diagonal (structure $\mathbf{A} = a \mathbf{I}$). Under this constraint, the interaction becomes equivalent to a masked self-attention where the "attention mask" is not a heuristic (like the causal triangle) but a learned, data-dependent decay matrix (a 1-semiseparable matrix).10
This duality enables a "best of both worlds" approach:
Training: Use the Attention View. Compute the matrix $\mathbf{M}$ via block-wise matrix multiplication. This utilizes GPU Tensor Cores (specialized for dense matmul), achieving significantly higher FLOP utilization than the parallel scan of Mamba-1.13
Inference: Use the Recurrent View. Collapse the matrix back into a single hidden state $h_t$ for $O(1)$ autoregressive generation.
2.5 Linear Attention and RWKV
Linear Attention seeks to eliminate the softmax bottleneck in the standard attention equation:


$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$$

By replacing softmax with a kernel feature map $\phi(\cdot)$, one exploits the associativity of matrix multiplication:


$$(QK^T)V = Q(K^T V)$$

This allows computing $\sum_{j} \phi(k_j)^T v_j$ as a cumulative sum (state), yielding linear complexity.
RWKV (Receptance Weighted Key Value) refines this by reintroducing the decay/forgetting mechanism of RNNs into Linear Attention. It uses a "Receptance" vector $R$ (sigmoid-activated) that acts as a forget gate, and "Time-decay" parameters $w$ to weight historical KV pairs:


$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t}v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

This formulation is effectively a linear attention mechanism where the attention scores decay exponentially with relative distance, computable in recurrent mode.5
3. Architectural Analysis of Emerging Models
This section deconstructs the specific architectures of the leading candidates for efficient recommendation.
3.1 Mamba and Mamba-2
Mamba Block: The Mamba block operates on an input sequence $X$ of shape $(B, L, D)$.
Expansion: Project inputs to $2D$ via linear layers.
Convolution: Apply a 1D causal convolution (kernel size typically 4) to capture local n-gram contexts. This acts as a "short-term memory" buffer before the SSM.
SSM: The core Selective SSM processes the expanded features. The state dimension $N$ (typically 16) is expanded relative to $D$.
Gating: The output is gated by a SiLU-activated branch of the original input (similar to Gated Linear Units).7
Mamba-2 Optimization: Mamba-2 removes the sequential linear projections inside the block to allow larger chunks of computation to happen in parallel. By enforcing the SSD constraints (scalar diagonal $A$), it enables the use of block-diagonal matrix multiplication algorithms, achieving 2-8x faster training throughput than Mamba-1 while maintaining similar perplexity.11
3.2 xLSTM (Extended LSTM)
xLSTM is a modernization of the LSTM designed to compete with Transformers. It introduces two major variants: sLSTM (scalar) and mLSTM (matrix).
Exponential Gating: Standard LSTMs use sigmoid gating ($\sigma \in $), which can struggle with magnitude scaling. xLSTM uses exponential gating (similar to softmax temperature), allowing the model to revise storage decisions more aggressively.6
Matrix Memory (mLSTM): Instead of a scalar cell state $c_t \in \mathbb{R}^D$, mLSTM uses a matrix cell state $C_t \in \mathbb{R}^{D \times D}$. The update rule resembles Linear Attention:

$$C_t = f_t \odot C_{t-1} + i_t \odot (v_t k_t^T)$$

where $v_t k_t^T$ is the outer product of value and key vectors. This allows xLSTM to store relational associations (key-value pairs) directly in its memory, giving it retrieval capabilities comparable to Transformers but with $O(N)$ complexity.16
3.3 RWKV-5/6
Architecture: RWKV consists of stacked blocks of Time-Mixing (Attention-like) and Channel-Mixing (FFN-like) layers.
Time-Mixing: Uses the linear attention formulation described in 2.5. Recent versions (RWKV-5/6) introduce multi-headedness and data-dependent decay rates (similar to Mamba's selective $\Delta$), allowing different attention heads to have different "attention spans".18
Inference Advantage: RWKV is explicitly designed for non-GPU inference (e.g., CPUs, mobile), utilizing the fixed state size to run effectively on edge devices.
4. Survey of SSM-based Recommender Systems
The adaptation of these general-purpose sequence models to the specific domain of recommendation (SRS) involves addressing challenges like sparse item IDs, irregular time intervals, and diverse sequence lengths.
4.1 Mamba4Rec: The Baseline Adaptation
Mamba4Rec represents the direct application of the Selective SSM to sequential recommendation.
Architecture: It replaces the Self-Attention layers of SASRec with Mamba blocks. The architecture consists of an Item Embedding layer, a stack of Mamba layers (Selective S4 + GLU + Norm), and a Prediction layer.
Configuration: Typically uses a state dimension $N=16$, expansion factor $E=2$, and kernel size $K=4$ for the local convolution.
Key Insight: Experiments show that a single Mamba layer often achieves competitive performance with multi-layer Transformers, indicating that the recurrent state efficiently compresses the necessary user history.
Performance: Evaluated on MovieLens-1M and Amazon-Beauty, Mamba4Rec achieves comparable or superior Hit Rate (HR) and NDCG to SASRec, with inference latency reducing by ~70% for sequences of length $L=200$.2
4.2 SIGMA: Addressing Unidirectionality and Instability
Directly applying Mamba reveals two weaknesses: (1) Mamba is strictly causal (unidirectional), whereas offline training benefits from bidirectional context (future predicting past); (2) Mamba's state estimation is unstable for very short sequences (cold-start users).
Partially Flipped Mamba (PF-Mamba): SIGMA (Selective Gated Mamba) introduces a bidirectional mechanism. However, instead of a simple bi-directional pass (which would leak future information during inference), it creates a "partially flipped" sequence during training to simulate future context without violating causality constraints for the target prediction.
Dense Selective Gate: A specialized gating network fuses the forward and flipped representations, dynamically weighting the importance of future context vs. past history based on the input item's embedding.
Feature Extract GRU (FE-GRU): To handle short sequences where the SSM state hasn't converged, SIGMA includes a parallel GRU module. GRUs are known to be robust for short-term dependencies. The model fuses the Mamba output (long-term) and GRU output (short-term), effectively handling the "long-tail" user problem.
Results: SIGMA demonstrates a 2-5% improvement in NDCG@10 over Mamba4Rec on sparse datasets like Amazon Sports, validating the need for specialized short-sequence handling.20
4.3 SSD4Rec: Optimizing for Variable Lengths
SSD4Rec adopts the Mamba-2 (SSD) backbone and addresses a specific engineering bottleneck: padding.
Sequence Registers: In standard batched training, sequences of different lengths are padded to the maximum length $L_{max}$. For Transformers, masked attention handles this. For SSMs, padding can "contaminate" the recurrent state if not carefully masked (the state shouldn't update on pad tokens). SSD4Rec introduces Sequence Registers, a mechanism to handle variable-length sequences within the structured matrix multiplication of SSD without explicit padding zero-outs, improving training throughput on skewed datasets.
Masked Bidirectional SSD: It utilizes the duality framework to compute a bidirectional representation during the training phase using the block-diagonal matrix approach, ensuring high GPU utilization.23
4.4 TiM4Rec: Injecting Time Awareness
Standard SSMs treat interactions as equidistant steps ($t, t+1, t+2$). In reality, user actions are bursty (e.g., a cluster of clicks in one hour, then silence for days).
Time-Aware Mask Matrix: TiM4Rec modifies the SSD framework. In the attention dual view $\mathbf{M}_{ij}$, the decay is determined by the number of steps ($i-j$). TiM4Rec replaces this step count with the actual time interval $\Delta t_{ij}$.
Mechanism: It learns a mapping function that converts time intervals into decay factors for the semiseparable matrix $\mathbf{A}$. This allows the "forgetting" rate of the user state to be directly proportional to real time—interactions from months ago decay more than those from minutes ago, regardless of the number of intervening items.
Efficiency: Crucially, this time-awareness is injected into the structured mask matrix during training, preserving the $O(N)$ inference complexity.26
4.5 SS4Rec: Continuous-Time Modeling
SS4Rec takes the continuous-time nature of SSMs literally.
Hybrid Architecture: It employs two parallel SSMs:
Relation-Aware SSM: Models the semantic transition between items (Item A $\to$ Item B).
Time-Aware SSM: Models the temporal evolution. This branch uses the actual time intervals to drive the discretization step size $\Delta$.
ODE Evolution: The user's interest state $h(t)$ is treated as a trajectory in latent space governed by an Ordinary Differential Equation (ODE). The model predicts the user's preference state at the exact timestamp of the next interaction, rather than just the next logical step.
Outcome: This approach is particularly effective for "return time" prediction and datasets with highly irregular inter-interaction times.28
5. Comparative Performance Analysis
We synthesize experimental results from the surveyed papers to compare these architectures against Transformer baselines (SASRec, BERT4Rec).
5.1 Effectiveness (Accuracy)
Table 1 summarizes the performance (NDCG@10) on standard benchmarks.
Model
Architecture
ML-1M (Dense)
Amazon Beauty (Sparse)
Strengths
SASRec
Transformer
0.584
0.392
Robust baseline; proven capability.
BERT4Rec
Transformer (Bi)
0.595
0.401
Good at context; slow inference.
Mamba4Rec
Selective SSM
0.591 (+1.2%)
0.405 (+3.3%)
Efficient; beats SASRec on long seqs.
SIGMA
PF-Mamba + GRU
0.612 (+4.8%)
0.428 (+9.2%)
Best for short/sparse users.
SSD4Rec
Mamba-2 (SSD)
0.605
0.415
Fastest training; good accuracy.
xLSRec
xLSTM
0.590
-
Highly isotropic embeddings; low popularity bias.

Observation 1: Mamba-based models consistently match or slightly outperform SASRec. The gains are most pronounced in SIGMA, highlighting that raw Mamba requires architectural modifications (bidirectionality/gating) to fully exploit recommendation data.20
Observation 2: xLSTM (xLSRec) shows a unique advantage in embedding isotropy. Transformer embeddings often collapse into a narrow cone (popularity bias), but xLSTM embeddings span the latent space more uniformly, potentially offering more diverse recommendations.31
5.2 Efficiency (Speed and Memory)
Training Throughput:
Mamba-2 (SSD) is the fastest, achieving 2-8x speedup over Mamba-1 and Transformers on long sequences ($L > 2048$).
Transformers suffer drastic slowdowns as $L$ increases.
Inference Latency ($L=2000$):
SASRec: ~15ms (grows linearly with $L$).
Mamba4Rec: ~4ms (constant $O(1)$).
Result: Mamba models offer a 3-4x speedup in latency for long sequences, and importantly, the latency is stable regardless of user history length.3
Memory Consumption:
Transformers require caching $K, V$ matrices ($L \times D$). For a batch size of 128 and $L=1000$, this is substantial.
SSMs store only the recurrent state ($N \times D$, where $N \approx 16$). This represents a 10x-50x reduction in inference memory, enabling much larger batch sizes or deployment on edge devices.4
6. Strategic Proposal: Replacing Transformers in Production
Transitioning from a battle-tested Transformer backbone to an SSM architecture carries risks. We propose a phased "Hybrid-Distillation" strategy for production deployment.
6.1 The "Hybrid" Architecture Strategy
Pure SSMs can sometimes struggle with "associative recall" tasks (finding an exact match from the distant past) compared to Attention. A robust production architecture should be hybrid:
The Body: Use Mamba-2 (SSD) layers for the first $90\%$ of the network. This compresses the raw user history into high-quality features with linear complexity.
The Head: Use 1 or 2 Self-Attention layers (or a specific "retrieval head") at the very top. This allows the model to perform fine-grained comparison between the compressed state and specific recent items if necessary.
Benefit: This retains $O(N)$ complexity for the bulk of processing while recovering the specific retrieval capabilities of Attention.33
6.2 Distillation for Safe Migration
Train a massive Teacher Transformer (e.g., BERT4Rec-Large) and distill it into a Student Mamba.
Method: Use Matrix Orientation Distillation. Since SSD proves the duality between the mixing matrices, one can directly align the mixing matrix $\mathbf{M}_{student}$ (generated by SSM) with the attention matrix $\mathbf{A}_{teacher}$.
Outcome: This forces the Mamba model to learn the attention patterns (e.g., "look back at the item bought 3 steps ago") of the Transformer, transferring its reasoning capabilities into a linear-time student.34
6.3 Addressing Data Irregularity
For production systems with highly irregular user logs:
Adopt TiM4Rec's Time-Aware Masking. Standard Mamba might over-smooth interests over long periods of inactivity. TiM4Rec ensures that the state decay reflects real-time passage.
Implement Sequence Registers (from SSD4Rec). Production batches often contain users with 5 clicks and users with 5000 clicks. Registers prevent the computational waste of padding and ensure the "short" users don't corrupt the batch statistics.23
6.4 Hardware Deployment
Training: Use Triton kernels for the SSD algorithm. The standard PyTorch scan is suboptimal. NVIDIA's Mamba library provides fused kernels that are essential for realizing the theoretical speedups.
Inference: Deploy the SSM as a Recurrent State Machine.
State Store: Maintain a persistent database (e.g., Redis/Cassandra) storing the hidden state $h_u$ for each user.
Incremental Update: When a user clicks item $x_t$, fetch $h_{u, t-1}$, compute $h_{u, t} = \mathbf{A}h_{u, t-1} + \mathbf{B}x_t$, and store $h_{u, t}$.
Latency: This operation is strictly $O(1)$ and requires fetching only a small vector ($16 \times D$), unlike fetching a massive KV cache. This enables real-time, session-based recommendation on high-traffic platforms.15
7. Conclusion
The era of accepting quadratic complexity as the cost of quality in recommender systems is ending. The theoretical unification provided by Structured State Space Duality (SSD) demonstrates that the dichotomy between the "smart but slow" Transformer and the "fast but dumb" RNN is false; they are dual representations of the same underlying structured computation.
For production recommendation systems, Mamba-2 (augmented with SIGMA's bidirectional gates for offline training and TiM4Rec's time-awareness for online serving) represents the optimal path forward. It offers the training throughput to digest massive interaction logs, the expressivity to capture complex user behaviors, and, most critically, the constant-time inference required to deploy "infinite context" personalization at scale. By adopting the hybrid and distillation strategies outlined above, engineering teams can replace Transformer backbones, unlocking a new generation of highly responsive, long-context recommender systems.
Works cited
A Survey of Mamba - arXiv, accessed January 5, 2026, https://arxiv.org/html/2408.01129v3
Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2403.03900v1
SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation | Request PDF - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/397306128_SSD4Rec_A_Structured_State_Space_Duality_Model_for_Efficient_Sequential_Recommendation
Mamba LLM Architecture: A Breakthrough in Efficient AI Modeling | SaM Solutions, accessed January 5, 2026, https://sam-solutions.com/blog/mamba-llm-architecture/
A Survey of RWKV - arXiv, accessed January 5, 2026, https://arxiv.org/html/2412.14847v1
xLSTMTime: Long-Term Time Series Forecasting with xLSTM - MDPI, accessed January 5, 2026, https://www.mdpi.com/2673-2688/5/3/71
state-spaces/mamba: Mamba SSM architecture - GitHub, accessed January 5, 2026, https://github.com/state-spaces/mamba
A Survey of Mamba - arXiv, accessed January 5, 2026, https://arxiv.org/html/2408.01129v5
Mamba: Linear-Time Sequence Modeling with Selective State Spaces - OpenReview, accessed January 5, 2026, https://openreview.net/forum?id=tEYskw1VY2
On Structured State-Space Duality - arXiv, accessed January 5, 2026, https://www.arxiv.org/pdf/2510.04944
Mamba2: The Hardware-Algorithm Co-Design That Unified Attention and State Space Models | by Daniel Stallworth | Medium, accessed January 5, 2026, https://medium.com/@danieljsmit/mamba2-the-hardware-algorithm-co-design-that-unified-attention-and-state-space-models-77856d2ac4f4
State Space Duality (Mamba-2) Part II - The Theory | Tri Dao, accessed January 5, 2026, https://tridao.me/blog/2024/mamba2-part2-theory/
On Structured State-Space Duality | OpenReview, accessed January 5, 2026, https://openreview.net/forum?id=C9LAf2tlKj
Mamba 2 | PDF | Matrix (Mathematics) | Tensor - Scribd, accessed January 5, 2026, https://www.scribd.com/document/748683510/mamba2
What Is A Mamba Model? | IBM, accessed January 5, 2026, https://www.ibm.com/think/topics/mamba-model
Efficient Attention Mechanisms for Large Language Models: A Survey - arXiv, accessed January 5, 2026, https://arxiv.org/html/2507.19595v2
How transformers, RNNs and SSMs are more alike than you think | by Stanislav Fedotov, accessed January 5, 2026, https://medium.com/nebius/how-transformers-rnns-and-ssms-are-more-alike-than-you-think-cd0f899893d8
PCF-RWKV: Large Language Model for Product Carbon Footprint Estimation - MDPI, accessed January 5, 2026, https://www.mdpi.com/2071-1050/17/3/1321
Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/2403.03900
[Quick Review] SIGMA: Selective Gated Mamba for Sequential Recommendation - Liner, accessed January 5, 2026, https://liner.com/review/sigma-selective-gated-mamba-for-sequential-recommendation
SIGMA: Selective Gated Mamba for Sequential Recommendation - CityUHK Scholars, accessed January 5, 2026, https://scholars.cityu.edu.hk/en/publications/sigma-selective-gated-mamba-for-sequential-recommendation/
SIGMA: Selective Gated Mamba for Sequential Recommendation - arXiv, accessed January 5, 2026, https://arxiv.org/html/2408.11451v4
SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation | AI Research Paper Details - AIModels.fyi, accessed January 5, 2026, https://www.aimodels.fyi/papers/arxiv/ssd4rec-structured-state-space-duality-model-efficient
SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation, accessed January 5, 2026, https://arxiv.org/html/2409.01192v1
SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation, accessed January 5, 2026, https://www.researchgate.net/publication/383702021_SSD4Rec_A_Structured_State_Space_Duality_Model_for_Efficient_Sequential_Recommendation
TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model - arXiv, accessed January 5, 2026, https://arxiv.org/html/2409.16182v1
TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model - arXiv, accessed January 5, 2026, https://arxiv.org/html/2409.16182v3
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models, accessed January 5, 2026, https://www.researchgate.net/publication/388954814_SS4Rec_Continuous-Time_Sequential_Recommendation_with_State_Space_Models
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2502.08132v3
Bidirectional Gated Mamba for Sequential Recommendation - arXiv, accessed January 5, 2026, https://arxiv.org/html/2408.11451v1
Vivekanand-R/Recommender-System: Core Value ... - GitHub, accessed January 5, 2026, https://github.com/QubitExplorer/Recommender-System
BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations - ChatPaper, accessed January 5, 2026, https://chatpaper.com/paper/219206
XiudingCai/Awesome-Mamba-Collection: A curated ... - GitHub, accessed January 5, 2026, https://github.com/XiudingCai/Awesome-Mamba-Collection
Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/2408.10189
Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models - NIPS papers, accessed January 5, 2026, https://proceedings.neurips.cc/paper_files/paper/2024/file/3848fef259495bfd04d60cdc5c1b4db7-Paper-Conference.pdf
Ai00 Inference - RWKV Language Model, accessed January 5, 2026, https://wiki.rwkv.com/inference/ai00.html
