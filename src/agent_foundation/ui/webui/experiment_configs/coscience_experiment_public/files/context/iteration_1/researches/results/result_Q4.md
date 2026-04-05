# Challenging Design Assumptions in Sequential Recommenders: Temporal Dynamics and User-Item Interaction Patterns

Production recommendation systems inherit two categories of design assumptions that have never been rigorously questioned. **First, temporal representation**: current Transformer-based architectures treat sequence position as a proxy for temporal dynamics, meaning a two-week vacation gap is encoded identically to a one-second browsing pause. **Second, user-item interaction design**: the way user representations and item representations communicate is often shallow and late in the pipeline, potentially limiting expressiveness. This report documents both issues, investigates alternatives from other domains, and proposes architectural directions that could unlock significant improvements.

---

## Current approaches conflate position with time

The dominant sequential recommenders—SASRec, BERT4Rec, and HSTU—encode position rather than actual elapsed time between interactions. **SASRec and BERT4Rec** use learned positional embeddings that create a learnable matrix P ∈ ℝ^(n×d) for sequence positions, with input representations computed as E_i = M_s + P (item embedding plus position embedding). From the SASRec paper: "Previous sequential recommenders discard timestamps and preserve only the order of items." This design choice has cascading consequences: session boundaries become invisible, interest decay cannot be mathematically modeled, and users returning after long absences appear indistinguishable from active users.

**HSTU** (Meta's Hierarchical Sequential Transduction Units) represents the most sophisticated temporal handling among major architectures, employing both relative time attention and relative position attention. Unlike its predecessors, HSTU can incorporate timestamp differences between target items and historical interactions, enabling context-aware recommendations where inference includes desired timestamps. However, even HSTU treats returning users as continuous sequences without explicit re-engagement detection or preference reset mechanisms.

**TiSASRec** (Time Interval Aware Self-Attention) addresses some limitations by computing pairwise time interval matrices where r^u_ij represents normalized time intervals between items i and j. It uses personalized scaling that divides all intervals by the user's minimum interval and clips long gaps to a maximum threshold k. The authors acknowledge this limitation explicitly: "We hypothesize that precise relative time intervals are not useful beyond a certain threshold"—meaning very long absences are treated as maximum-interval events regardless of their actual duration.

| Architecture | Position Encoding | Time Encoding | Interest Decay | Re-engagement |
|--------------|-------------------|---------------|----------------|---------------|
| SASRec | Learned | None | Implicit only | Not handled |
| BERT4Rec | Learned | None | Implicit only | Not handled |
| TiSASRec | Learned | Interval embeddings | Implicit | Clipped gaps |
| HSTU | Relative | Relative time attention | Implicit | Not handled |

The fundamental limitation across all architectures is the absence of explicit decay mechanisms—interest decay is learned implicitly through attention weights rather than mathematically modeled. No architecture explicitly handles users returning after long absences or detects when user context has shifted (morning versus evening, weekday versus weekend).

---

## Neural ODEs fail production requirements despite theoretical appeal

Neural Ordinary Differential Equations offer theoretically attractive continuous-time dynamics but face **insurmountable practical barriers** in production recommendation settings. The original Chen et al. (2018) paper established that Neural ODE models run **2-4× slower than equivalent ResNets**, with this gap widening during training as learned dynamics become more complex. The Number of Function Evaluations (NFE) increases unpredictably—the model effectively "learns to be deep."

The core architectural problem is that ODE integration is inherently sequential: each solver step depends on the previous step's result. This fundamentally breaks GPU parallelism. Research from MIT on GPU ODE solvers notes: "Integrating parallelism with differential equation solvers is particularly challenging due to the inherently serial nature of the time-stepping integration procedure." Even with batch parallelization, **over 95% of GPU threads remain idle** during timesteps due to sequential dependencies.

Training stability presents additional challenges. Stiff ODEs—systems with widely varying timescales—are pervasive in user behavior modeling where some preferences evolve over months while others change within minutes. A 2024 paper titled "The Vanishing Gradient Problem for Stiff Neural Differential Equations" proves this is fundamental: "This vanishing gradient phenomenon is not an artifact of any particular method, but a universal feature of all A-stable and L-stable stiff numerical integration schemes."

Production recommender systems require **sub-10ms inference latency** for real-time serving. Optimized Transformer inference achieves sub-1ms with TensorRT/ONNX optimization, while Neural ODEs offer no comparable optimization pathway—adaptive solvers vary computation based on learned dynamics, making latency SLAs impossible to guarantee. Despite active academic research (GDERec, TGODE, GNG-ODE), **no major technology company has publicly deployed Neural ODEs for production recommendations**.

---

## Exponential decay has an exact closed-form solution

The exponential decay ODE central to temporal preference modeling—**dh/dt = -h/τ + f(x)/τ**—has a complete analytical solution that eliminates any need for numerical solvers. This is a first-order linear inhomogeneous ODE solvable via the integrating factor method.

For constant f(x) = f₀ between events, the closed-form solution is:

```
h(t) = h(t₀)·e^{-(t-t₀)/τ} + f₀·(1 - e^{-(t-t₀)/τ})
```

This describes a **leaky integrator** where the state exponentially decays toward the equilibrium value f₀, with τ controlling decay speed. The solution requires no numerical approximation—it can be computed exactly for any time gap (t - t₀).

State-space models like S4 and Mamba exploit similar closed-form solutions through structured discretization. The continuous-time state equations x'(t) = Ax(t) + Bu(t) are converted to discrete form using **Zero-Order Hold (ZOH)** or **bilinear transforms**:

```
ZOH discretization:
Ā = exp(Δ·A)
B̄ = A⁻¹(exp(Δ·A) - I)·B
```

For the scalar exponential decay case (A = -1/τ), this simplifies to:
```
Ā = e^{-Δ/τ}
B̄ = (1 - e^{-Δ/τ})/τ
```

Recent work on **Closed-form Continuous-time (CfC) networks** derives approximate closed-form solutions for more complex Liquid Time-Constant dynamics, achieving **100-150× speedup** over ODE-based networks without sacrificing expressiveness. The key insight enabling practical continuous-time modeling is that **linear dynamics within layers combined with nonlinear transformations between layers** can approximate complex systems while maintaining computational tractability.

| Model Type | Closed Form? | Implementation |
|------------|--------------|----------------|
| Exponential decay ODE | ✅ Yes | Direct formula computation |
| S4/Mamba linear SSM | ✅ Yes | Discretization + convolution |
| Hawkes process (exp kernel) | ✅ Yes | Analytical intensity |
| CfC networks | ✅ Yes | Approximate sigmoidal gate |
| General Neural ODEs | ❌ No | Requires numerical solver |

---

## Semantic-aware decay through input-dependent τ(x)

The most promising approach to semantic-aware temporal modeling is **learning decay rates as functions of input semantics** rather than treating decay as a global constant. Different preference types require different temporal dynamics: stable preferences like "likes horror movies" should decay slowly (large τ), while transient intents like "shopping for birthday gift" should decay rapidly (small τ).

The most directly relevant work is **Adaptive Collaborative Filtering** (RecSys 2023), which explicitly computes personalized decay rates through a feed-forward network:

```python
α_ui = sigmoid(Linear(concat(user_embedding, item_embedding)))
```

This constrains decay rates to [0,1] and makes them dependent on both user and item semantics—a direct template for τ(x).

**Mamba's selective mechanism** provides the most sophisticated architectural pattern for input-dependent parameterization. In standard state-space models, parameters A, B, C are time-invariant. Mamba makes them input-dependent:

```python
B(x) = Linear_N(x)      # Input projection
C(x) = Linear_N(x)      # Output projection
Δ(x) = softplus(Linear(x))  # Discretization step
```

The Δ parameter is particularly relevant—it controls effective decay rate and can be viewed as directly analogous to τ(x). Making Δ input-dependent allows the model to "selectively propagate or forget information along the sequence length dimension depending on the current token."

LSTM forget gates demonstrate that input-dependent memory control is learnable through simple gating mechanisms. The forget gate f_t = σ(W_f·[h_{t-1}, x_t] + b_f) determines how much old memory to retain, with the effective decay period determined by gate values. Research on **power-law forget gates** suggests τ(x) should be flexible enough to capture both exponential and power-law decay patterns, matching empirical observations about human memory.

Several architectural patterns emerge for implementing semantic-aware decay:

- **Neural network output**: τ(x) = softplus(MLP(concat(item_embedding, context)))
- **Mamba-style selection**: Δ(item) = softplus(Linear(item_embedding))
- **Category-conditioned decay**: τ(x) = τ_base[item_category] + residual_MLP(item_embedding)
- **Dual-stream architecture**: Separate large-τ pathway for stable preferences and learnable small-τ pathway for transient intents, with attention-weighted combination

---

## State-space models enable O(1) inference with true time gaps

For production systems requiring both proper temporal gap modeling and O(1) inference per item, **state-space models (SSMs)** emerge as the optimal architecture family. The critical innovation is **variable discretization step sizes based on actual time intervals** rather than uniform positional steps.

**SS4Rec** (2025) represents the most directly relevant architecture, explicitly designed for continuous-time sequential recommendation:

```python
# Time-aware SSM with variable discretization
Δ_t = actual_time_gap(t, t-1)     # Variable Δ from timestamps
h_t = Ā(Δ_t)·h_{t-1} + B̄(Δ_t)·u_t  # Time-dependent state update
```

This breaks the uniform time interval assumption that limits standard SSMs while maintaining O(1) inference complexity through the recurrent formulation. SS4Rec outperforms SASRec, GRU4Rec, and other baselines by **5-15%** across five benchmark datasets.

**Mamba** achieves 5× higher throughput than Transformers with constant memory regardless of context length (versus growing KV cache in attention-based models). The selective mechanism enables content-based reasoning while maintaining linear complexity. Production deployments exist: AI21's Jamba is a 52B parameter hybrid SSM-Transformer model.

**xLSTM** (Extended LSTM) offers an alternative with exponential gating replacing sigmoid for more dynamic memory control. The mLSTM variant uses matrix memory that is fully parallelizable during training while maintaining O(1) inference. However, xLSTM lacks the native continuous-time formulation of SSMs and would require explicit time interval input similar to GRU-D.

| Architecture | Training | Inference/Item | Memory | Time Handling |
|--------------|----------|----------------|--------|---------------|
| Transformer | O(L²) | O(L) | O(L) KV cache | Position only |
| S4/Mamba | O(L) | **O(1)** | O(N) state | Variable Δ possible |
| SS4Rec | O(L) | **O(1)** | O(N) state | Native time gaps |
| xLSTM (mLSTM) | O(L) | **O(1)** | O(d²) | Requires Δt input |
| GRU-D | O(L) | **O(1)** | O(d) | Exponential decay |

The recommended production architecture combines:
1. **SSM backbone with time-interval-dependent discretization** (Δ = actual_time_gap)
2. **Input-dependent decay parameters** via Mamba-style selection (τ(x) learned from semantics)
3. **Hybrid attention layer** for precise retrieval where SSMs show weakness (copying tasks)
4. **Closed-form state updates** exploiting the analytical solution for exponential decay

---

## Implementation pathway for production systems

The mathematical foundation for proper temporal modeling in recommendations is well-established. The exponential decay ODE has an exact closed-form solution; state-space models provide the architectural framework to exploit it; and input-dependent parameterization enables semantic-aware decay rates. The path to production involves three key components:

**Variable discretization from timestamps**: Rather than uniform Δ, compute discretization directly from interaction timestamps. For diagonal A matrices (standard in modern SSMs), this is computationally efficient: Ā_i = exp(Δ·a_i) computed element-wise.

**Semantic-aware τ(x)**: Implement decay rate as a learned function of item/context embeddings using softplus activation to ensure positivity. Train end-to-end from recommendation performance rather than hand-tuning.

**Hybrid architecture for retrieval**: SSMs show weakness on copying/retrieval tasks due to fixed state size. A sparse attention layer operating on the SSM hidden states can provide precise retrieval capability without sacrificing the O(1) recurrent efficiency.

The result is an architecture that models true temporal gaps (not sequence position), captures interest decay and context shifts mathematically, supports semantic-aware decay rates, and maintains O(1) inference complexity suitable for real-time serving at scale—addressing the fundamental limitations of current Transformer-based sequential recommenders without the computational overhead of Neural ODE solvers.

---

## Part B: User-Item Interaction Design Assumptions

Beyond temporal representation, a second category of unquestioned assumptions deserves scrutiny: **how and when user representations interact with item representations**. Our audit of HSTU, SASRec, and related architectures reveals that this interaction happens surprisingly late and shallowly—potentially limiting the expressiveness of these models.

### Current interaction patterns are shallow and late

Examining the generative-recommenders codebase (HSTU implementation), we find that user and item representations follow largely **separate encoding paths** until the final scoring stage:

| Component | User Processing | Item Processing | Interaction |
|-----------|-----------------|-----------------|-------------|
| **Preprocessing** | UIH embeddings concatenated with actions/context | Item features concatenated via MLP | Additive combination only |
| **Main Encoder** | Full HSTU attention (12-24 layers) | 2-layer MLP transformation | None during encoding |
| **Scoring** | User embedding extracted | Item embedding extracted | Element-wise multiplication |

The critical observation: **user context and item features first interact only at the final scoring stage, via element-wise multiplication** (`user_emb * item_emb`). This is a remarkably shallow interaction for such sophisticated representations.

The preprocessing stage does combine multiple feature types (content embeddings, action encodings, contextual features), but these combinations are **purely additive**:
```
output = content_embedding + action_embedding + contextual_embedding
```

No cross-feature attention or multiplicative interactions occur during the encoding process itself.

### Other domains use richer cross-representation interaction

Surveying how other domains handle interaction between different representation types reveals significantly richer patterns:

**NLP (Cross-Attention in Encoder-Decoder Models)**:
BERT, T5, and similar models use cross-attention mechanisms where one sequence can attend to another during encoding. In translation models, the decoder attends to encoder outputs at every layer—not just at the final output.

**Multimodal Learning**:
Vision-language models like CLIP and Flamingo enable image features to influence text encoding and vice versa through:
- Cross-modal attention layers
- Gated fusion mechanisms
- Perceiver-style latent arrays that aggregate information across modalities

**Recommender Systems with Richer Interaction**:
- **Deep Interest Network (DIN)**: Uses attention between candidate items and user history within the encoding phase
- **DIEN**: Models evolving user interests through GRU with attention to current candidate
- **Feature interleaving approaches**: Some architectures interleave features from different sources, enabling attention across types

The pattern that emerges: **successful multimodal and cross-sequence systems enable interaction DURING encoding, not just at scoring**.

### What expressiveness is lost by late interaction?

Consider what relationships **cannot** be captured when user and item only interact via final-stage multiplication:

1. **User-Conditioned Item Salience**: The importance of an item feature should depend on user context. A "comedy" genre tag matters more to users who recently watched comedies. Current architectures assign fixed importance during item encoding.

2. **Item-Conditioned History Attention**: Which historical interactions are relevant depends on the candidate item. When scoring a horror movie, recent horror views should be weighted more heavily. Current architectures compute user embeddings independently of the candidate.

3. **Cross-Feature Dependencies**: A user's "morning" context combined with an item's "coffee" category creates a compound signal stronger than either alone. Additive preprocessing misses such multiplicative interactions.

4. **Dynamic Feature Importance**: The relevance of different user features should adapt to the item being scored. A user's age matters more for some products than others. Late-stage multiplication cannot capture this.

### Promising directions for richer interaction

Without prescribing specific solutions, the research suggests several promising directions:

**Direction 1: Earlier User-Item Interaction**
Rather than processing user and item sequences entirely separately, introduce interaction points earlier in the pipeline. The user representation could influence how item features are encoded, or vice versa.

**Direction 2: Cross-Attention Mechanisms**
Borrowing from encoder-decoder architectures, items could attend to user history during their encoding, and user history encoding could be conditioned on candidate items (in a computationally efficient manner).

**Direction 3: Learned Aggregation Tokens**
NLP models aggregate sequence information through special tokens that attend to the full sequence. Similar mechanisms could aggregate user context in a form that's available to item encoding—creating a bridge between the two representation spaces.

This concept naturally extends to **multiple aggregation tokens**—analogous to multi-head attention or mixture-of-experts architectures. Rather than compressing all user information into a single token, multiple tokens could capture different aspects of user preferences:
- One token for long-term stable interests
- Another for recent session-level intent
- Additional tokens for different preference facets (genre preferences, temporal patterns, social influences)

These multiple tokens could then participate in item encoding with different weights or through routing mechanisms, enabling richer and more nuanced user-item interaction during the encoding phase.

**Direction 4: Feature Interaction Before Sequence Processing**
Wukong's Factorization Machine Block captures higher-order feature interactions through learned factorization. Similar patterns could be applied in HSTU's preprocessing to create richer feature combinations before sequence transduction.

**Direction 5: Iterative Refinement**
Rather than single-pass encoding, user and item representations could be refined iteratively, with each refinement step incorporating information from the other side.

### Estimated impact and computational considerations

Based on analogous improvements in other domains, richer user-item interaction could yield:
- **5-15% improvement** in recommendation quality metrics (NDCG, HR@K)
- **Larger gains on cold-start** where shallow interaction particularly fails
- **Better personalization** for users with diverse or evolving interests

Computational overhead depends on implementation:
- Cross-attention adds O(U×I) complexity where U=user sequence length, I=item features
- Aggregation token approaches add minimal overhead (single token attending to sequence)
- Preprocessing interaction adds O(features²) but runs before the expensive sequence encoder

The key insight is that **richer interaction during encoding may be more valuable than deeper encoders**—a single cross-attention layer could outperform multiple additional self-attention layers that process user and item independently.

---

## Synthesis: Two Interrelated Opportunities

This research reveals two fundamental assumptions limiting current sequential recommenders:

| Assumption | Current State | Opportunity |
|------------|---------------|-------------|
| **Temporal** | Position = Time | Analytical continuous-time with learned decay |
| **Interaction** | User-Item interact only at scoring | Earlier, richer cross-representation interaction |

These opportunities are **interrelated**: a continuous-time formulation could naturally incorporate interaction timing (when in the sequence does user-item interaction matter most?), and richer interaction mechanisms could be time-aware (recent items interact differently than older items).

The combined architectural direction points toward systems where:
1. **Temporal dynamics** are modeled analytically with learned per-item decay rates
2. **User-item interaction** happens during encoding, not just at scoring
3. **Information flows bidirectionally** between user and item representations
4. **Feature combinations** go beyond additive to capture cross-feature dependencies

Both assumptions were inherited rather than justified. Both limit expressiveness in ways that empirical analysis and cross-domain comparison suggest are unnecessary. Challenging both simultaneously could yield compounding improvements in recommendation quality.
