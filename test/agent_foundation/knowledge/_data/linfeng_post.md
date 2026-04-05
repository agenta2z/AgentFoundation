RankEvolve-CMSL: Breaking the Heuristic Ceiling via Human-Guided Algorithm Evolution
Author: Linfeng Liu, Junjie Yang, Tao Jia, Hong Li, Hong Yan on behalf of RankEvolve-CMSL v-team.
TLDR: As part of MRS RankEvolve, we piloted a human-in-the-loop algorithm evolution approach on CMSL. Modern model development increasingly benefits from human-guided AI evolution: research scientists define the conceptual goals and constraints, while LLMs iteratively generate, implement, and refine candidate architectures. CMSL-Evolve is an example—humans framed the core modeling dilemma (“early-iteration length vs. later-interaction width”), and LLMs explored and synthesized architectural variants that navigate this tradeoff.
Our pilot exploration achieved an additional 0.15% - 0.2% NE gain on top of the original CMSL (which had ~0.43% NE gain on the IFR main MTMTL model), demonstrating the promise of RankEvolve - AI as an expert co-designer for algorithm evolution.
1. The Human Insight: Identifying the "Length-versus-Width" Paradox
Modeling rich, high-order interactions in complex feature spaces poses a fundamental trade-off.
Representation fidelity requires preserving raw input information through early interaction layers, favoring long sequence lengths and fine-grained attention.
Interaction capacity requires wide, dense transformations in later modules to capture high order feature interactions and semantic correlations.
Experienced ML engineers identified this core tension—preserving length vs. expanding width—and specified the goal: design an architecture that maintains long-context fidelity while still enabling wide, high-capacity reasoning under strict resource budgets. Attempting to satisfy both directly is computationally infeasible—dense layers applied over long sequences cause parameter and activation blow-up memory. To this end, we experimented with a new approach: RankEvolve.
2. The RankEvolve Loop
Our ranking stack has grown highly complicated over decades of developments. Manual architecture design is slow and constrained by engineer intuitions. In RankEvolve-CMSL, we shifted our ML engineer’s role to Constraint Architects. We established a "Human-in-the-Loop" evolutionary workflow leveraging Large Language Models (LLMs) for Automated Algorithm Design in the pilot work. Engineers focus on what to optimize and under what constraints; LLMs explore how to realize the objective in code through the human-in-the-loop evolutionary workflow.
Phase 1: The Prompt (Human Direction)
We provided the LLM (acting as the "Navigator") with our dilemma and the constraints:
"Design a PyTorch architecture that decouples sequence resolution from reasoning capacity. It must maximize information flow between a high-fidelity stream and a high-capacity stream without exceeding our FLOPs budget. Current manual baselines achieve +0.43% NE; the goal is to discover non-intuitive interaction pathways that exceed this."
Phase 2: The Code Generation (LLM Implementation)
The LLM (acting as the "Generator") did not just tune hyperparameters; it wrote novel Python code. Over iterations, it experimented with various topologies. Through a Reflexion Loop—where we fed back error logs and performance metrics—the LLM self-corrected.
Phase 3: The evaluation
With Human-in-the-loop evaluating the NE changes, the architecture evolves to a structurally distinct direction and achieves good early gain.
3. The Evolved Solution
The RankEvolve-CMSL loop converged to a Dual-stream architecture with Latent-Guided Compression. The final design focuses on the length-versus-width paradox: it keeps a long-context, high-fidelity stream while iteratively refining dense reasoning in a small, high-capacity latent space.
May be an image of text that says 'Feature Encoders Float- Sparse- Embed Seq → HSTU CAT The Dual- Stream CMSL Interaction Layer Context Stream (Co) Context Cι Latent Update Latent Lị Latent Stream LatentStream( (Lo) Latent Reasoning Q=Li,K,V= Q=Li,K.v=[4114 [CIILi] Task Heads Li+1 Context Update (Lfiat) Like, share, comment, vpvd comment,vpvd... ... Ci+1: Latent-Guided'
3.1 The Dual-Stream Topology
The most significant structural change proposed by the LLM is the a dual-stream architecture:
Stream 1 - Context Stream: (C_i: [B, K_i, D]): The high-fidelity pathway. It is initialized with the full input sequence (K_0=L) to avoid early information loss. Across layers, K_i follows a learned reduction schedule to progressively filter redundancy.
Stream 2- Latent Stream: (Li: [B, M, D]): The high-capacity pathway. A fixed-size workspace that summarizes the information most relevant for the ranking objectives. It persists across layers, serving as the dedicated locus for dense computation.
This separation lets the model keep long-context fidelity in the Context Stream while concentrating expensive computation on a compact set of objective-focused latents in the Latent Stream.
3.2 Interaction via Subspace Expansion
Instead of simple averaging, the LLM implemented a Subspace Expansion mechanism to couple the streams. It wrote a Concat operation that projects the streams into a widened manifold before processing. The expanded representation then feeds a cross-attention block, with latents as queries and the joint stream as keys/values, yielding updated latents. This maximizes the bandwidth for correlation detection before the integration phase, a nuance often missed in manual designs aimed purely at parameter reduction.
The core reasoning engine operates exclusively within the Latent Stream - flattening the expanded latent state L'_i to a unified vector L_flat: [B, MD] and process it through a Holistic Integration Block: L_next = Reshape(Φ(L_flat)). Here, Φ(*) denotes a deep and wide fully-connected network. This block entangles information across all latent positions and interaction heads. Because M is small, we can afford a very large hidden dimension for Φ, giving the model the structural capacity to infer complex global properties without the memory cost of processing the full sequence C_i.
3.3 The Core Innovation: Latent-Guided Compression
The most significant "mutation" the LLM introduced was in how the Context Stream is compressed. It wrote a controller module—the CMSL Block—where the Latent Stream dictates the compression of the Context Stream.
The Context Stream is compressed using a policy derived from the Latent Stream. The updated latent state L_next conditions a lightweight hypernetwork to generate low-rank projection operators U and V: U, V = HyperNet(L_next), where U: [B, K_i+1, R] and V: [B, R, K_i]. The compression is applied as: C_i+1= U @ (V @ Ci). This Latent-Guided Pooling ensures that the Context Stream is compressed based on semantic relevance determined by the high-capacity Latent Stream, rather than by static pooling operations.
The evolve iteration includes:
Latent pulls information from context from advanced feature interaction algorithms.
Latent drives compression and refinement of the context stream.
Latent reason globally through a deep and wide MLP.
The model effectively thinks in latents and reads/writes in context. Expensive computation is distributed in a compact latent space, decoupling long-context fidelity from high-capacity reasoning to mitigate the length-versus-width paradox.
4. Experiments on IFR
We prototyped RankEvolve-CMSL on the IFR main MTML ranking model. The baseline is the CMSL MC9 proposal (~+0.43% NE over the MC9 baseline). We evaluate two capacity settings: RankEvolve-air (light) and RankEvolve-pro (heavy). The RankEvolve-air uses smaller internal dimensions and is sized to run on H100, while the RankEvolve-pro targets B200. We use half batch size during explorations.
On IFR critical tasks, the RankEvolve-air shows a stable 0.1% NE improvement, while the RankEvolve-pro achieves +0.15% NE, approaching +0.2%. Notably, these gains are from a minimal instantiation (e.g. shrinked interaction-module compared to our MC9 proposal, a 1-layer latent reasoning MLP), indicating additional headroom for further improvements as we enrich the latent reasoning and the training recipe.
May be an image of text
May be an image of text
May be an image of text that says '0 my yexperiment:variable_step_ expeiniteiti/ie/ -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 w/hrrm/rwmnd mmmwhw "pTM 0 10B 20 3OB 40 50B mvai_base mvai_ base mvai_cmsl-mc9 mvai_RankEvolve-air mvai_RankEvolve-pro 60B examples 70B 8OB'

5. Summary and looking forward
By moving humans out of the implementation loop and into the direction loop, we allowed RankEvolve-CMSL to emerge.
Speed: The architecture was discovered in days, not months.
Performance: The dynamic, latent-guided compression provided gains significantly on top of the 0.43% NE baseline, mitigating the "Length-versus-Width" paradox by structurally separating fidelity from capacity.
Looking Ahead:
Our pilot study demonstrates the promise of RankEvolve AI as an expert co-designer for algorithm evolution. As the next step, we plan to
Explore curriculum-based strategies to guide and accelerate intelligent RankEvolve - algorithmic evolution.
Move toward the north star of RankEvolve - an intelligent self-improving RecSys Algorithms.
More updates to come—stay tuned.
V-Team
MRS: Linfeng Liu, Junjie Yang, Zefeng Zhang, Tao Jia, Hong Li, Haoyue Tang, Jijie Wei, Li Sheng, Tai Guo, Yujia Hao, Yujunrong Ma, Zikun Cui, Yan Li, Renzhi Wu, Haicheng Wang, Tony Chen, Yu Zheng, Xiong Zhang, Chenglin Wei, Yanzun Huang, Yuting Zhang, Matt Ma, Hao Wang, Wei Zhao, Yifan Shao
CFR: Zheng Wu, Xinyue Shen, Yizhou Qian, Ji Qi
FM: Honghao Wei, Hang Wang, Pu Zhang, Xinzhe Chai, Jeff Wang, Mingda Li, Jianwu, Harry Huang, Li Yu
IFR: Wanli Ma, Wenshun Liu, Yue Weng
FBR: Johann Dong, Baokui Yang, Srivatsan Ramanujam, Zhen Hang Jiang, Jugal Marfatia, Yihuan Huang, Kuen Ching
AIDI TorchRec: Huanyu He, Shuangping Liu
AI & Compute Foundation: Haoyu Zhang, Jeremy Hadidjojo
DS: Lingxiao Zhai, Michael Li, Shan Huang, Ke Gong
PMs: Neeraj Bhatia, Vijayant Bhatnagar, Deepak Vijaywargi
Acknowledgement
Thanks Hong Yan for guiding the team toward an inspiring technical path. Thanks Hong Yan and Lars Backstrom, for fostering a fast-moving culture that empowers us to boldly pursue new ideas. Thanks Sri Reddy, Xinyao Hu, Sophia (Xueyao) Liang, Neeraj Bhatia, Nipun Mathur for the strong leadership and EM support.
