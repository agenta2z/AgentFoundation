Temporal Dynamics in Sequential Recommenders: From Positional Proxies to True Continuous-Time Modeling Product
1. Introduction: The Chronological Imperative in Recommender Systems
The architecture of recommender systems has undergone a profound metamorphosis over the last decade, transitioning from static matrix completion tasks to dynamic, sequence-aware modeling. This shift reflects a fundamental recognition in the field: user preference is not a fixed point in a high-dimensional space, but a trajectory—a continuous, evolving path influenced by internal shifts in taste and external environmental triggers. Traditional Collaborative Filtering (CF) approaches viewed user-item interactions as a static bipartite graph, effectively collapsing years of behavioral history into a single, timeless snapshot. While effective for capturing broad, global preferences, these methods failed to account for the sequential dependencies and temporal evolutions that define real-world human behavior.1
The emergence of Sequential Recommendation (SR) addressed this deficiency by explicitly modeling the order of interactions. Early SR models treated user history as a Markov chain, where the next action was dependent solely on the previous one, or as a sequence of tokens processed by Recurrent Neural Networks (RNNs). These models introduced the concept of "context" as a dynamic state, updated with each new interaction. However, a critical limitation persisted across the majority of these architectures: the discretization of time. By treating a sequence of interactions as an ordered list $\{i_1, i_2, \dots, i_n\}$, these models rely on "positional proxies"—integer indices that represent the relative order of events but discard the absolute temporal information. In this paradigm, the interval between the first and second interaction is treated identically to the interval between the second and third, regardless of whether the actual elapsed time was five seconds or five months.3
This "Uniformity Assumption" is a significant deviation from reality. User behavior is inherently irregular and bursty, characterized by short periods of intense activity (e.g., a shopping session) followed by long periods of dormancy. The temporal gap between interactions carries immense signal: a short gap might indicate positive excitation or comparison shopping, while a long gap might signal a drift in preference or a shift in user needs. Ignoring these continuous time intervals limits the model's ability to distinguish between immediate intent and long-term interest evolution.5
Consequently, the frontier of sequential recommendation has moved toward Continuous-Time Modeling. This paradigm shift seeks to replace heuristic time-embeddings with rigorous mathematical frameworks rooted in stochastic processes and differential equations. We are witnessing the integration of Temporal Point Processes (TPP) to model the intensity of events, the application of Neural Ordinary Differential Equations (Neural ODEs) to represent preference as a continuous flow, and the utilization of Neural Controlled Differential Equations (NCDEs) and State Space Models (SSMs) to handle irregular sampling and long-range dependencies.7
This report provides an exhaustive analysis of this transition. We dissect the theoretical underpinnings and architectural innovations of discrete models like SASRec and BERT4Rec, hybrid approaches like TiSASRec and Time-LSTM, and the advanced continuous-time frameworks of Neural ODEs, SDEs, and SSMs. By synthesizing evidence from recent literature, we demonstrate how treating user preference as a continuous trajectory in latent space offers superior capabilities in handling data sparsity, irregular intervals, and the complex interplay between short-term excitation and long-term evolution.
2. The Positional Proxy Paradigm: Architectures and Limitations
To understand the necessity of continuous-time modeling, one must first dissect the mechanics and deficiencies of discrete sequential models. The dominant paradigm in SR for the past decade has treated user history as an ordered list of items $\mathcal{S}_u = \{i_1, i_2, \dots, i_n\}$, where the primary signal is the relative order rather than the absolute or relative time.
2.1 The Transformer Hegemony: SASRec and BERT4Rec
The application of the Transformer architecture to sequential recommendation marked a watershed moment in the field. Models like SASRec (Self-Attentive Sequential Recommendation) and BERT4Rec leveraged the self-attention mechanism to capture long-range dependencies that RNNs struggled to retain. These models have become the standard baselines against which all temporal models are compared.9
SASRec:
In SASRec, the model processes interactions unidirectionally (left-to-right). To inform the self-attention mechanism of the sequence order—since self-attention is permutation invariant—it injects a learnable positional embedding $\mathbf{P} \in \mathbb{R}^{n \times d}$ into the item embeddings. The input embedding $\mathbf{E}$ for the item at position $t$ is defined as:


$$\mathbf{\hat{E}}_t = \mathbf{M}_{i_t} + \mathbf{P}_t$$

where $\mathbf{M}_{i_t}$ is the item embedding and $\mathbf{P}_t$ represents the embedding for the integer index $t$. The model then applies stacked Multi-Head Self-Attention (MHSA) layers to compute the hidden representation.


$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$

This mechanism allows the model to assign attention weights to previous items based on their semantic relevance to the current context, weighted by their learned positional relevance.11
BERT4Rec:
BERT4Rec extends this by employing a deep bidirectional encoder trained via a Cloze objective (Masked Item Prediction). Unlike SASRec's next-item prediction, BERT4Rec masks random items in the sequence and trains the model to predict the masked item using context from both the left (past) and right (future). This bidirectional modeling allows for a richer representation of item co-occurrence patterns but fundamentally relies on the same positional encoding scheme as SASRec.9
The Uniformity Fallacy:
The critical flaw in these architectures is the "Uniformity Assumption." By relying on integer positions, SASRec and BERT4Rec implicitly assume that the influence of item $i_{t-1}$ on item $i_t$ is independent of the wall-clock time elapsed between them. This assumption fails to distinguish between two radically different user behaviors:
Session-based browsing: A user clicks on three items within 5 minutes.
Long-term periodic consumption: A user purchases three items over the course of 3 months.
To a standard Transformer using vanilla positional encodings, these two sequences appear identical if the item IDs are the same. The model cannot naturally decay the influence of older interactions based on time, nor can it recognize the "burstiness" of a session. It must learn these patterns indirectly through the sequence order, which is a noisy and lossy proxy for the true temporal dynamics.3
2.2 Limitations of Recurrent Architectures
Prior to Transformers, RNNs and Gated Recurrent Units (GRUs) were the standard. While they process data sequentially, they typically update the hidden state $h_t$ based solely on the previous hidden state $h_{t-1}$ and current input $x_t$, occurring at discrete steps.


$$h_t = \text{GRU}(h_{t-1}, x_t)$$

Standard RNNs lack an intrinsic mechanism to account for the time interval $\Delta t$. If a user returns after a year-long hiatus, a standard RNN carries over the hidden state from a year ago with the same weight as if the user had returned yesterday. This leads to the "stale state" problem, where obsolete preferences contaminate current recommendations. The gradient flow in RNNs is also restricted to discrete steps, making it difficult to model processes that evolve continuously between observations.5
2.3 The Impact of Irregularity on Preference Drift
Real-world interaction data is highly irregular. The distribution of time intervals often follows a heavy-tailed distribution (e.g., power-law), characterized by frequent short intervals (sessions) and rare long intervals (breaks).
Short Intervals: These often indicate positive or negative excitation. A user buying a camera might immediately buy a lens (positive excitation) or reject other cameras (negative excitation). Capturing this requires modeling the high-frequency dynamics of user intent.13
Long Intervals: These indicate preference drift or a shift in latent context. A gap of six months might signify a change in user life stage, financial status, or general interests. Discrete models attempt to mitigate this via heuristic data augmentation or by limiting sequence length, but these are band-aid solutions that do not address the root inability to model continuous evolution.5
The limitations of discrete positional proxies necessitated the development of models that explicitly incorporate time, leading to the "Hybrid" era of Time-Aware Recurrent and Attention models.
3. Bridging the Gap: Explicit Time Modeling in Discrete Architectures
Recognizing the deficiencies of pure positional proxies, researchers developed "hybrid" architectures. These models retain the discrete backbone (RNN or Transformer) but introduce mechanisms to explicitly encode timestamp information, transforming the input from a sequence of items to a sequence of tuples $(item, time)$.
3.1 Time-LSTM: Gating Memory with Time
Time-LSTM represents an early but significant attempt to integrate continuous time into Recurrent Neural Networks. The core innovation is the introduction of "Time Gates" that modulate the cell state based on the time interval $\Delta t$ between interactions.12
Standard LSTMs use Forget ($f_t$), Input ($i_t$), and Output ($o_t$) gates to control the flow of information. Time-LSTM adds a time-dependent modification to these gates. The research proposes three varying structures:
Time-LSTM 1: Uses a single time gate $T_m$ controlled by the interval $\Delta t$. This gate acts as a filter, allowing recent information to pass through more strongly while dampening older signals. It essentially weights the memory cell $c_{t-1}$ by a decay function derived from $\Delta t$.
Time-LSTM 2: Separates the time influence into two distinct gates:
$T_{1m}$: Modulates the impact of the current input on the memory, allowing the model to decide how much the new interaction should update the state based on how much time has passed.
$T_{2m}$: Modulates the previous memory, effectively storing time intervals for long-term dependency modeling.
Time-LSTM 3: Uses coupled input and forget gates to improve parameter efficiency while retaining temporal sensitivity.
By weighting the memory cell $c_{t-1}$ by a function of $\Delta t$ (e.g., $c_t = f_t \odot c_{t-1} \cdot g(\Delta t) + i_t \odot \tilde{c}_t$), Time-LSTM allows the model to "forget" irrelevant history during long gaps, effectively modeling preference decay. This explicit handling of $\Delta t$ showed significant improvements over standard LSTM and Phased LSTM on datasets like LastFM and CiteULike.12
3.2 TiSASRec: Time Interval Aware Self-Attention
TiSASRec (Time Interval Aware Self-Attention for Sequential Recommendation) adapts the Transformer architecture to handle irregular intervals. The authors argue that the interaction between two items $i_u$ and $i_v$ depends not just on their semantic similarity (dot product of embeddings) but also on the time distance between them.3
Mechanism:
TiSASRec modifies the positional embedding mechanism. Instead of just absolute position $k$, it computes a relative time interval $r_{ij} = |t_i - t_j|$ for every pair of items in the sequence. These intervals are bucketed (quantized) to handle the continuous range of time and mapped to learnable time embeddings $\mathbf{M}_k \in \mathbb{R}^d$.
The attention score calculation is updated to include these time embeddings:


$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}(\mathbf{K} + \mathbf{R}_K)^T}{\sqrt{d}}\right)(\mathbf{V} + \mathbf{R}_V)$$

where $\mathbf{R}_K$ and $\mathbf{R}_V$ are representations derived from the relative time intervals. This allows the model to learn complex temporal dynamics, such as:
Recency: Placing higher attention weights on recent items (short intervals).
Periodicity: Focusing on items that occurred at similar time-of-day or seasonal intervals (e.g., attending to a winter coat purchase from exactly one year ago).
Critique:
While TiSASRec outperforms SASRec, it fundamentally remains a discrete model. It treats time intervals as relations between discrete tokens rather than modeling the evolution of the user state between tokens. The quantization of time intervals (clipping max intervals) also introduces resolution loss, and the $O(N^2)$ complexity of calculating pairwise time intervals can be computationally heavy for long sequences.3
3.3 Static-Dynamic Interest Learning (SDIL)
Recent frameworks like SDIL attempt to decouple user interests into static (stable) and dynamic (evolving) components. Using Temporal Positive and Negative Excitation (TPNE), this approach explicitly models how a prior interaction excites or inhibits the next choice based on the time interval. It employs a time-decay kernel function to measure excitation strength, acknowledging that the causal link between purchase A and purchase B diminishes as $\Delta t$ increases. This framework explicitly addresses the "excitation" aspect of user behavior, where short intervals imply a causal link (buying a phone $\rightarrow$ buying a case) rather than just a sequential one.13
4. Temporal Point Processes (TPP): Probabilistic Foundations
To move beyond heuristic time embeddings, we must turn to Temporal Point Processes (TPP), a rigorous mathematical framework for modeling discrete events in continuous time. TPPs are stochastic processes composed of a time series of events $\{(t_i, y_i)\}$, where $t_i$ is the timestamp and $y_i$ is the mark (item type/ID). Unlike the discrete time-step models, TPPs model the probability of an event occurring at any continuous time point $t$.16
4.1 The Conditional Intensity Function
The core component of a TPP is the conditional intensity function $\lambda(t | \mathcal{H}_t)$, which represents the instantaneous rate of an event occurring at time $t$ given the history $\mathcal{H}_t$. Formally:
$$ \lambda(t) = \lim_{\Delta t \to 0} \frac{P(\text{event in }
4.2 The Hawkes Process
A specific type of TPP, the Hawkes Process, is particularly relevant for recommendation. It is a "self-exciting" process where past events increase the probability of future events.


$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \kappa(t - t_i)$$

Here, $\mu$ is the base intensity (user's general activity level), and $\kappa(\cdot)$ is a decay kernel (usually exponential, e.g., $e^{-\beta(t-t_i)}$) that models how the excitement from event $t_i$ fades over time. This mathematically formalizes the intuition that user actions are clustered in time (bursts). For example, purchasing a flight ticket excites the probability of booking a hotel, but this excitement decays as the travel date approaches or passes.20
4.3 Recurrent Marked Temporal Point Processes (RMTPP)
While classical TPPs (like Hawkes) rely on fixed parametric kernels, RMTPP utilizes Recurrent Neural Networks to learn the intensity function from data, freeing the model from rigid assumptions.
Architecture:
RMTPP embeds the event history using an RNN (typically LSTM or GRU). The hidden state $h_i$ at step $i$ encapsulates the history. The intensity function for the next event at time $t > t_i$ is parameterized as a non-linear function of the history and the elapsed time:


$$\lambda^*(t) = \exp\left( \mathbf{v}^T h_i + w(t - t_i) + b \right)$$

This formulation allows the intensity to vary continuously between events. The term $w(t - t_i)$ explicitly models the time decay or accumulation of pressure for the next event. RMTPP predicts both the time of the next interaction and the item (mark) involved, unifying timing prediction and item recommendation into a single loss function (negative log-likelihood of the process). This establishes a connection between the discrete event prediction of RNNs and the continuous timing of TPPs.17
4.4 Neural TPPs and Deep Learning Integration
The integration of TPPs with deep learning (Neural TPPs) allows for flexible, non-linear modeling of user dynamics. By using RNNs or Transformers to encode history and TPP layers to output continuous density functions, these models provide a "Continuous-Time" view where predictions can be made for any future time $t$, not just the next step.
Neural TPPs often employ Neural ODEs or flexible intensity functions to capture complex multimodal distributions of inter-event times, moving beyond the simple exponential decay of Hawkes processes.18
4.5 TPP-LLM: Semantic and Temporal Fusion
Recent advancements have merged Large Language Models (LLMs) with TPPs. TPP-LLM utilizes the semantic reasoning capabilities of LLMs to understand the nature of events (marks) while using TPP heads to model the timing.
Methodology:
Unlike traditional methods that rely on categorical event representations (IDs), TPP-LLM directly utilizes textual descriptions of event types. The LLM captures the rich semantic information (e.g., reasoning that "User bought a crib" implies a specific temporal trajectory towards "diapers"). The model incorporates temporal embeddings and uses parameter-efficient fine-tuning (PEFT) to learn temporal dynamics without retraining the massive LLM backbone.
This addresses a limitation of standard Neural TPPs, which treat item IDs as abstract integers. TPP-LLM can reason about causal chains and temporal intervals based on world knowledge, enriching the intensity function with semantic context.16
5. The Neural Differential Equation Revolution
The most significant leap in continuous-time modeling for recommenders comes from the application of Neural Ordinary Differential Equations (Neural ODEs). Introduced by Chen et al. (2018), Neural ODEs challenge the notion of discrete layers in deep learning, viewing the transformation of hidden states as a continuous flow parameterized by a differential equation.24
5.1 From Discrete Layers to Continuous Flows
In a standard Residual Network (ResNet), the state transition is defined as:


$$h_{t+1} = h_t + f(h_t, \theta_t)$$

Neural ODEs take the limit as the step size approaches zero, yielding a continuous differential equation:


$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

The hidden state at any future time $T$ is obtained by integration:


$$h(T) = h(0) + \int_{0}^{T} f(h(t), t, \theta) dt$$

This integral is solved using a black-box ODE solver (e.g., Runge-Kutta, Euler, Dormand-Prince).
Advantages for Recommendation:
Irregular Sampling: Neural ODEs naturally handle data arriving at arbitrary timestamps. The solver simply integrates from $t_i$ to $t_{i+1}$, regardless of the interval size. The state $h(t)$ evolves continuously between observations, allowing the model to estimate user preference at unobserved time points. This is a massive advantage over RNNs which jump from state to state.5
Parameter Efficiency: The dynamics function $f$ is shared across time, often resulting in fewer parameters than stacking many discrete layers. The model learns the law of evolution rather than a specific mapping for each step.
Adjoint Method: Gradients can be computed via the Adjoint Sensitivity Method, which solves an augmented ODE backward in time. This allows training with constant memory cost ($O(1)$ memory with respect to depth), as it avoids storing intermediate activations during the forward pass.25
5.2 Modeling Preference Evolution
In the context of RS, the latent state $h(t)$ represents the user's preference vector at time $t$. The function $f$ (a neural network) learns the "vector field" of preference evolution—how interests naturally drift or decay in the absence of new interactions.


$$\text{Drift:} \quad \frac{d\mathbf{u}(t)}{dt} = \text{Net}_{\theta}(\mathbf{u}(t), t)$$

When a new interaction occurs, the continuous trajectory is typically updated or "jumped" to fuse the evolved preference with the new observation. This models the user's "state of mind" as a continuous flow that is occasionally perturbed by discrete actions.27
5.3 Technical Challenges and Solutions
Implementing Neural ODEs in SR is not trivial.
Stiffness: The differential equations describing user behavior can be "stiff" (combining very fast changes during sessions and very slow drifts between sessions), requiring sophisticated solvers that are computationally expensive.
Static Trajectory Limitation: A standard ODE is determined entirely by its initial condition $h(0)$. Once the trajectory starts, it is deterministic. This is insufficient for sequence modeling where subsequent inputs (new purchases) must alter the trajectory. This limitation led to the development of Neural Controlled Differential Equations (NCDEs).29
6. Neural Controlled Differential Equations (NCDEs): Solving the Irregularity Problem
Neural Controlled Differential Equations (NCDEs) extend the ODE framework to process sequential data, acting as the continuous-time analogue of RNNs. They are specifically designed to address the limitation of standard ODEs in incorporating incoming data streams.
6.1 The Mechanism of Control
In an NCDE, the vector field is modulated by a "control path" $X(t)$, which represents the continuous interpolation of the input data sequence.


$$dh(t) = f(h(t)) dX(t)$$

In integral form:


$$h(T) = h(0) + \int_{0}^{T} f(h(t)) \frac{dX(t)}{dt} dt$$

Here, $X(t)$ is typically a continuous spline (e.g., cubic Hermite spline) created from the discrete interaction sequence $\{(t_i, x_i)\}$. The neural network $f$ determines how the hidden state should change in response to changes in the input path.
Unlike standard ODEs where the trajectory is fixed at $t=0$, NCDEs allow the hidden state to react to incoming data events as they happen on the continuous timeline. This makes them ideal for sequential recommendation where user intent is updated by every click.7
6.2 Log-NCDEs and Lie Brackets
A major bottleneck in NCDEs is the dimensionality and smoothness of the control path. Log-NCDEs introduce a significant optimization using the Log-ODE method from rough path theory. Instead of using the raw path $X$, they construct vector fields from iterated Lie Brackets of the neural network functions.
The equation for the Log-NCDE update over an interval $[r_i, r_{i+1}]$ is:


$$g_{\theta, X}(h_s) = \bar{f}_{\theta}(h_s) \frac{\log(S^N(X)_{[r_i, r_{i+1}]})}{r_{i+1} - r_i}$$

where $S^N(X)$ is the depth-$N$ truncated signature of the path, and the log-signature compresses the path information.
Insight: This approach compresses the path information into high-order terms (signatures) that capture the geometric properties of the sequence (e.g., the area enclosed by the trajectory, order of operations) more efficiently than raw splines. In experiments on multivariate time series classification (a proxy for sequence modeling), Log-NCDEs showed superior performance (accuracy of 64.43%) and efficiency compared to standard NCDEs and even discrete baselines like S5 or LRU.31
6.3 Causal-Aware Collaborative Neural CDE (C3DE)
Applying NCDEs to long-term prediction (e.g., crowd flow or user lifetime value) introduces the risk of error accumulation and spurious correlations. C3DE (Causal-aware Collaborative neural CDE) utilizes a dual-path NCDE structure to capture asynchronous dynamics across different timescales.
Architecture:
Dual-Path NCDE: Models the evolution of collaborative signals and individual signals separately.
Causal Effect-Based Dynamic Correction: Integrates a causal effect estimator to prune spurious correlations that might be amplified during the continuous integration process. This ensures that the model learns robust causal drivers of behavior rather than just temporal coincidences.
Experiments on datasets with notable fluctuations demonstrated C3DE's superiority in modeling long-term dynamics compared to standard discrete and continuous baselines.34
7. Stochastic Dynamics: Neural SDEs and Uncertainty
User behavior is not fully deterministic; it contains inherent aleatoric uncertainty (noise) and epistemic uncertainty (lack of knowledge). Neural Stochastic Differential Equations (Neural SDEs) extend the deterministic ODE framework by adding a diffusion term driven by Brownian motion.36
7.1 The Stochastic Formulation
A Neural SDE models the state evolution as:


$$dh(t) = \underbrace{f(h(t), t) dt}_{\text{Drift (Deterministic)}} + \underbrace{g(h(t), t) dW(t)}_{\text{Diffusion (Stochastic)}}$$
Drift $f(\cdot)$: Represents the expected evolution of user preference (e.g., the gradual shift from "Action" to "Drama").
Diffusion $g(\cdot)$: Represents the uncertainty or volatility in the user's state. $W(t)$ is a Wiener process (Brownian motion).
This formulation acknowledges that the user's state is a distribution, not a point, evolving over time.
7.2 Evidential Neural SDE (E-NSDE) for Recommendation
E-NSDE integrates Neural SDEs with Evidential Deep Learning (EDL) to explicitly quantify uncertainty in sequential recommendation.
Insight: The uncertainty of a user's preference should increase as the time interval $\Delta t$ since the last interaction increases. If a user hasn't logged in for a month, the model should be less confident than if they logged in 5 minutes ago.
Mechanism:
NSDE Module: Learns the continuous user and item representations, incorporating stochasticity via the diffusion function.
Evidential Module: Quantifies both aleatoric and epistemic uncertainties. It uses a monotonic network to ensure that uncertainty grows with time $\Delta t$.
Exploration: The model uses the predicted epistemic uncertainty to guide exploration. When uncertainty is high (e.g., after a long gap), it recommends diverse or novel items (like a new genre) to maximize information gain, rather than sticking to safe, stale predictions. This creates a feedback loop where uncertainty drives the "exploration" arm of the explore-exploit tradeoff.36
7.3 Neural Jump SDEs (NJSDE)
Some user behaviors are smooth (gradual interest shift), while others are abrupt (shocks, immediate needs). Neural Jump Stochastic Differential Equations (Neural JSDEs) combine continuous flow (ODE/SDE) with discrete jumps, creating a hybrid system.
Equation:


$$dz(t) = f(z(t), t)dt + w(z(t), t)dN(t)$$

Here, $dN(t)$ is a point process (like Hawkes) that triggers instantaneous updates to the state vector $z(t)$. The term $w(z(t), t)$ determines the magnitude and direction of the jump when an event occurs.
Adjoint Method with Jumps:
Training NJSDEs requires a modified adjoint method. At the time of a discontinuity (jump) $\tau_j$, the adjoint vector $a(t)$ describing the loss gradients also undergoes a jump. The backward pass must integrate the adjoint ODE between jumps and then applying a discrete update at each $\tau_j$ to account for the discontinuity in the forward pass.
This hybrid system perfectly models the dual nature of recommendation: the continuous evolution of "taste" and the discrete impact of "purchase events".27
8. Graph-Based Continuous Time Models
Sequential Recommendation often ignores the collaborative signal (other users). Graph Neural Networks (GNNs) capture this, and their integration with ODEs creates powerful Spatio-Temporal models that treat the user-item graph as a living, evolving organism.
8.1 GDERec: Graph ODE for Sequential Recommendation
GDERec proposes an autoregressive Graph ODE framework. It views the user-item interaction graph not as a static structure but as a dynamic system evolving continuously.
Architecture:
Edge Evolving Module: Uses an ODE-based GNN to model the continuous formation and decay of edges (relationships) between users and items. The adjacency matrix is effectively treated as a continuous variable $A(t)$.
Temporal Aggregating Module: Uses a temporal attention network to aggregate information from neighbors. Crucially, it employs a Trainable Time Encoding $\Phi(t)$ (based on Bochner's theorem) to explicitly map timestamps into the latent space during message passing.

$$\Phi(t) = \frac{1}{\sqrt{d}} [\cos(\omega_1 t), \sin(\omega_1 t), \dots]$$
Mechanism: The model alternates between evolving the graph structure via ODE (filling in the gaps between interactions) and updating node embeddings via GNN (at interaction points). This captures the "collaborative evolution"—how the community's influence on a user changes over time.39
8.2 TGODE: Time-Guided Evolution Graphs
TGODE (Time-Guided Graph ODE) addresses two specific challenges: Irregular user interests (sparsity) and Uneven item distributions (popularity shifts).
Graph Construction:
User Time Graph ($\mathcal{G}_u^s$): Captures individual user sequences. Nodes are items in the sequence, and edges are labeled with interaction times.
Item Evolution Graph ($\mathcal{G}_c^s$): Captures the global evolution of item relationships across all users, structured to model the temporal popularity trends.
Time-Guided Diffusion Generator:
To handle the temporal sparsity (gaps where user interest is unknown), TGODE employs a Time-Guided Diffusion Generator. This module uses a diffusion process (adding and removing noise) to generate "augmented" interactions in the sparse time intervals. It effectively "hallucinates" probable user states during long gaps to smooth the trajectory.


$$q(\mathbf{z}_K | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_K; \sqrt{\bar{\alpha}_K} \mathbf{z}_0, (1 - \bar{\alpha}_K)\mathbf{I})$$

This augmented graph is then fed into a generalized Graph Neural ODE to align the user's personal interest evolution with the global item trend evolution. This dual-graph approach allows TGODE to correct for "popularity bias," distinguishing whether a user bought an item because they liked it or because everyone else was buying it at that moment.5
9. State Space Models (SSMs) and Continuous Time
A recent entrant to the field is the State Space Model (SSM), popularized by the S4, S5, and Mamba architectures. SSMs mathematically link RNNs, CNNs, and continuous-time ODEs, offering linear scaling with sequence length.
9.1 The SSM Connection
A linear State Space Model is defined as a continuous system:


$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$

$$y(t) = \mathbf{C}h(t)$$

Through discretization techniques (like the Zero-Order Hold or Bilinear transform), this continuous system can be converted into a recurrent form (efficient for inference) or a convolutional form (efficient for parallel training).
9.2 SS4Rec: Breaking the Uniformity of Mamba
Standard Mamba/S4 models assume a fixed sampling rate (uniform step size $\Delta$), which makes them ill-suited for the irregular timestamps of recommender systems. SS4Rec adapts SSMs for continuous-time recommendation by handling irregular intervals explicitly.
Architecture:
SS4Rec employs a hybrid architecture with two distinct SSMs:
Time-Aware SSM: This component handles the temporal dynamics. Crucially, it uses variable step sizes $\Delta_t$ derived from the actual time intervals in the user sequence.

$$\bar{\mathbf{A}}_t = \exp(\mathbf{A} \Delta_t)$$
$$\bar{\mathbf{B}}_t = (\mathbf{A}^{-1}(\exp(\mathbf{A} \Delta_t) - \mathbf{I})) \mathbf{B}$$

This discretizes the continuous ODE differently for every step, preserving the temporal fidelity of the sequence.
Relation-Aware SSM: Models the contextual dependencies between items, focusing on the sequence order and item properties.
Hybridization:
By combining these, SS4Rec enables the efficiency of Mamba (linear scaling $O(N)$ vs Transformer's $O(N^2)$) while respecting the continuous nature of user behavior. It effectively models the "continuous dependency" from irregular intervals, providing time-specific personalized recommendations that outperform SASRec and BERT4Rec on dense datasets.4
10. Comparative Analysis and Performance Metrics
The shift from discrete to continuous models involves significant trade-offs in accuracy, complexity, and efficiency.
10.1 Comparative Overview
The following table summarizes the key characteristics of the discussed architectures:
Model Class
Time Representation
Mechanism
Pros
Cons
SASRec / BERT4Rec
Positional Embeddings
Self-Attention (Transformer)
High parallelization, captures long-term dependency.
Ignores actual time intervals; assumes uniform steps; fails on irregular bursts.
TiSASRec
Relative Time Intervals
Modified Self-Attention
Captures interval-aware dependencies.
Still discrete; quantization loss; complexity $O(N^2)$.
RMTPP
Continuous (Intensity)
RNN + Point Process
Models event timing and probability natively.
RNN bottlenecks (sequential processing); typically less expressive than Transformers.
Neural ODE (NODE)
Continuous (Integral)
Differential Eq. Solver
Handles irregular data perfectly; constant memory during training.
Slow training (multiple solver steps); static trajectory limitation (Initial Value Problem).
Neural CDE (NCDE)
Continuous (Control Path)
Controlled Diff. Eq.
Reacts to incoming data continuously; SOTA for irregular series.
High computational cost; complex to implement; heavy math overhead.
Log-NCDE
Continuous (Log-Signature)
Lie Brackets + Log-ODE
More efficient path representation; superior classification accuracy.
Mathematical complexity (Rough Path Theory); requires pre-computation of signatures.
SS4Rec (SSM)
Discretized Continuous
State Space / Mamba
Linear complexity $O(N)$; models continuous dynamics via variable $\Delta$.
Requires specialized discretization for irregular times; newer, less stable training than Transformers.
TGODE
Continuous Graph
Graph ODE + Diffusion
Models collaborative signal + time evolution; handles sparsity via diffusion.
Highly complex architecture; expensive training due to diffusion + ODE combination.

10.2 Performance Drivers: Why Continuous Models Win
Data Sparsity: In sparse datasets (e.g., Amazon Reviews, where users might have only 5 interactions over 2 years), time intervals are the only reliable signal linking disjointed events. Continuous models leverage this "time-between-events" signal, whereas discrete models see only a short, disconnected sequence. This explains why models like TGODE show improvements of 10-46% over baselines on sparse datasets.1
Prediction Accuracy: Experiments consistently show that models like GDERec and TGODE outperform SASRec/TiSASRec by margins of 5-15% on metrics like NDCG@10 and HR@10. The performance gap is widest on datasets with high irregularity (e.g., sporadic e-commerce vs. regular music streaming).5
Future Prediction: Continuous models can predict when the user will return, not just what they will buy. This enables "Just-in-Time" recommendation strategies (e.g., sending a push notification exactly when the user's intensity function peaks) which is impossible with standard Transformers.46
11. Future Directions and Open Challenges
Despite the theoretical elegance of continuous-time recommendation, several challenges impede widespread industrial adoption.
11.1 Scalability and Latency
ODE solvers (e.g., Dormand-Prince) are computationally expensive. They require multiple function evaluations per forward pass. While Adjoint methods save memory, they do not save time. For real-time latency-sensitive applications (serving <100ms), pure Neural ODEs are often too slow. SSMs (SS4Rec) and Log-NCDEs offer a middle ground, providing continuous modeling properties with faster inference speeds. Future work must focus on "stiffness-aware" solvers and distillation techniques to compress ODE models into lighter inference engines.8
11.2 Integration with Large Language Models
The rise of LLM-powered recommendation suggests a convergence. TPP-LLM is a pioneer here, but deep integration is lacking. Future models might use LLMs to generate the differential equation itself (symbolic regression) or use the LLM's world model to initialize the drift functions of a Neural SDE, combining semantic reasoning with continuous temporal dynamics. The ability of LLMs to understand temporal concepts ("Christmas is in December," "Baby grows in 9 months") could significantly enhance the "Drift" functions in Neural SDEs.16
11.3 Causal Inference in Continuous Time
Most current models are correlational. They learn that "Event A follows Event B." They do not robustly model intervention. If a platform sends a push notification (intervention) at time $t$, how does that alter the continuous trajectory? Causal NCDEs (C3DE) are a step in this direction, using counterfactual reasoning to disentangle true user preference from system-induced bias. Expanding this to full recommendation pipelines is a critical frontier.34
11.4 Handling Cold-Start and Long-Tail
Continuous models theoretically handle cold-start better (by modeling the global drift $\lambda_0$), but training stable ODEs on extremely short sequences (1-2 items) is unstable. Diffusion-based augmentation (as seen in TGODE) helps, but more robust "few-shot ODE" meta-learning frameworks are needed to stabilize the initial trajectories for new users.1
12. Conclusion
The field of Sequential Recommendation is undergoing a fundamental transformation. The era of "Positional Proxies"—where complex temporal dynamics were flattened into integer indices—is giving way to True Continuous-Time Modeling.
This shift is not merely academic pedantry; it is a necessary response to the complex, irregular, and non-stationary nature of human behavior.
Positional models (SASRec) excel at structure but fail at dynamics.
Hybrid models (TiSASRec) offer a bridge but remain constrained by discretization.
Neural Differential Equations (NODEs, NCDEs, SDEs) and Continuous SSMs provide the ultimate fidelity, treating preference as a fluid trajectory that flows, jumps, and diffuses over time.
While computational hurdles remain, architectures like SS4Rec and Log-NCDEs demonstrate that it is possible to combine the mathematical rigor of differential equations with the scalability of deep learning. As recommender systems move from predicting the "next token" to anticipating the "next moment," continuous-time modeling will become the backbone of the next generation of personalized intelligence. The future of recommendation is not a sequence of steps; it is a continuous flow.
Works cited
A Survey on Sequential Recommendation - arXiv, accessed January 5, 2026, https://arxiv.org/html/2412.12770v2
A Systematic Survey on Federated Sequential Recommendation - Preprints.org, accessed January 5, 2026, https://www.preprints.org/manuscript/202503.1015/v1
Time Interval Aware Self-Attention for Sequential Recommendation - University of California San Diego, accessed January 5, 2026, https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2502.08132v1
Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs - arXiv, accessed January 5, 2026, https://arxiv.org/html/2511.18347v1
Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/371923207_Uniform_Sequence_Better_Time_Interval_Aware_Data_Augmentation_for_Sequential_Recommendation
Comprehensive Review of Neural Differential Equations for Time Series Analysis - IJCAI, accessed January 5, 2026, https://www.ijcai.org/proceedings/2025/1179.pdf
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2502.08132v3
BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/1904.06690
BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer | alphaXiv, accessed January 5, 2026, https://www.alphaxiv.org/overview/1904.06690v1
BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer | Request PDF - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/337015455_BERT4Rec_Sequential_Recommendation_with_Bidirectional_Encoder_Representations_from_Transformer
[Quick Review] What to Do Next: Modeling User Behaviors by Time-LSTM - Liner, accessed January 5, 2026, https://liner.com/review/what-to-do-next-modeling-user-behaviors-by-timelstm
Modeling Temporal Positive and Negative Excitation for Sequential Recommendation, accessed January 5, 2026, https://arxiv.org/html/2410.22013v1
Understanding Architecture of LSTM - Analytics Vidhya, accessed January 5, 2026, https://www.analyticsvidhya.com/blog/2021/01/understanding-architecture-of-lstm/
Time Interval Aware Self-Attention for Sequential Recommendation - Semantic Scholar, accessed January 5, 2026, https://www.semanticscholar.org/paper/Time-Interval-Aware-Self-Attention-for-Sequential-Li-Wang/9c7a455b1e48d01a99e58e884a6c1acb75074ad0
TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models | OpenReview, accessed January 5, 2026, https://openreview.net/forum?id=RofgmKmk5n
Recurrent Marked Temporal Point Processes: Embedding ... - SIGKDD, accessed January 5, 2026, https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
Advances in Temporal Point Processes: Bayesian, Neural, and LLM Approaches - arXiv, accessed January 5, 2026, https://arxiv.org/html/2501.14291v2
Neural Temporal Point Processes: A Review - Technical University of Munich, accessed January 5, 2026, https://portal.fis.tum.de/en/publications/neural-temporal-point-processes-a-review/
[PDF] Recurrent Marked Temporal Point Processes: Embedding Event History to Vector, accessed January 5, 2026, https://www.semanticscholar.org/paper/Recurrent-Marked-Temporal-Point-Processes%3A-Event-to-Du-Dai/439d216e9cf62fe2fe6fa95646543140bb074b43
Neural Temporal Point Processes: A Review | Request PDF - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/350749926_Neural_Temporal_Point_Processes_A_Review
Marked Temporal Dynamics Modeling Based on Recurrent Neural Network - Shenghua Liu, accessed January 5, 2026, https://shenghua-liu.github.io/assets/pdf/papers/pakdd2017-markedtemp.pdf
TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models - arXiv, accessed January 5, 2026, https://arxiv.org/html/2410.02062v1
Neural Ordinary Differential Equations (Neural ODEs): Rethinking Architecture - Medium, accessed January 5, 2026, https://medium.com/@justygwen/neural-ordinary-differential-equations-neural-odes-rethinking-architecture-272a72100ebc
Neural Ordinary Differential Equations - NeurIPS, accessed January 5, 2026, http://papers.neurips.cc/paper/7892-neural-ordinary-differential-equations.pdf
Understanding Neural ODE's - Jonty Sinai, accessed January 5, 2026, https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html
High Performance Neural Jump Stochastic Differential Equations - GitHub, accessed January 5, 2026, https://raw.githubusercontent.com/mitmath/18337projects/main/spring2023/project_reports/18_337final_Shijie_Zhang.pdf
Neural Jump Stochastic Differential Equations - NeurIPS, accessed January 5, 2026, http://papers.neurips.cc/paper/9177-neural-jump-stochastic-differential-equations.pdf
An ODE based neural network approach for PM2.5 forecasting - PMC - PubMed Central, accessed January 5, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12246124/
Neural Controlled Differential Equations for Irregular Time Series - NeurIPS, accessed January 5, 2026, https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf
Log Neural Controlled Differential Equations: The Lie Brackets Make A Difference, accessed January 5, 2026, https://proceedings.mlr.press/v235/walker24a.html
Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference - Liner, accessed January 5, 2026, https://liner.com/review/log-neural-controlled-differential-equations-lie-brackets-make-difference
Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference - arXiv, accessed January 5, 2026, https://arxiv.org/html/2402.18512v1
[2509.12289] C3DE: Causal-Aware Collaborative Neural Controlled Differential Equation for Long-Term Urban Crowd Flow Prediction - arXiv, accessed January 5, 2026, https://arxiv.org/abs/2509.12289
Causal-Aware Collaborative Neural Controlled Differential Equation for Long-Term Urban Crowd Flow Prediction - arXiv, accessed January 5, 2026, https://arxiv.org/html/2509.12289v1
Evidential Stochastic Differential Equations for Time-Aware Sequential Recommendation - NeurIPS, accessed January 5, 2026, https://proceedings.neurips.cc/paper_files/paper/2024/file/7cdbd53dfbcf9a5263227555aac5b9cd-Paper-Conference.pdf
stable neural stochastic differential equa - arXiv, accessed January 5, 2026, https://arxiv.org/pdf/2402.14989
Learning Neural Jump Stochastic Differential Equations with Latent Graph for Multivariate Temporal Point Processes - IJCAI, accessed January 5, 2026, https://www.ijcai.org/proceedings/2025/0383.pdf
Learning Graph ODE for Continuous-Time Sequential Recommendation - arXiv, accessed January 5, 2026, https://arxiv.org/html/2304.07042v2
Graph ODEs and Beyond: A Comprehensive Survey on Integrating Differential Equations with Graph Neural Networks - PMC - NIH, accessed January 5, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12363673/
Learning Graph ODE for Continuous-Time Sequential Recommendation - IEEE Xplore, accessed January 5, 2026, https://ieeexplore.ieee.org/iel7/69/10549876/10379836.pdf
Learning Graph ODE for Continuous-Time Sequential Recommendation - ResearchGate, accessed January 5, 2026, https://www.researchgate.net/publication/377134645_Learning_Graph_ODE_for_Continuous-Time_Sequential_Recommendation
Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs - CSE, CUHK, accessed January 5, 2026, https://www.cse.cuhk.edu.hk/~cslui/PUBLICATION/KDD2025_TGODE_FinalVersion.pdf
[Literature Review] SS4Rec: Continuous-Time Sequential Recommendation with State Space Models - Moonlight, accessed January 5, 2026, https://www.themoonlight.io/en/review/ss4rec-continuous-time-sequential-recommendation-with-state-space-models
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models, accessed January 5, 2026, https://www.researchgate.net/publication/388954814_SS4Rec_Continuous-Time_Sequential_Recommendation_with_State_Space_Models
Time Interval Aware Self-Attention for Sequential Recommendation | Request PDF, accessed January 5, 2026, https://www.researchgate.net/publication/338757732_Time_Interval_Aware_Self-Attention_for_Sequential_Recommendation
A Comprehensive Review on Harnessing Large Language Models to Overcome Recommender System Challenges - arXiv, accessed January 5, 2026, https://arxiv.org/html/2507.21117v1
