# RankEvolve Architecture Design Document

## Executive Summary

**RankEvolve** is an AI-powered co-science framework that collaborates with researchers and scientists to scalably and consistently improve machine learning systems through iterative research cycles. The system demonstrates autonomous research on generative recommendation, achieving significant improvements on public benchmarks through human-in-the-loop evolution cycles.

**Key Results:**
- SYNAPSE v2 achieved **+6.3% NDCG improvement** over HSTU + linear-decay sampling baseline
- **2.3× throughput improvement** while maintaining quality
- Two iterative evolution cycles demonstrating the system's ability to identify bottlenecks, propose solutions, and validate improvements

---

## 1. System Overview

### 1.1 Vision

RankEvolve enables **AI-assisted autonomous research** by structuring the research process into iterative evolution cycles. The system:

- **Understands codebases comprehensively** - Parses and documents ML codebases to identify architectural patterns and extension points
- **Identifies high-potential research directions** - Uses structured reasoning patterns to generate targeted research queries
- **Conducts deep research** - Leverages multiple AI models for external knowledge and internal documentation search
- **Generates implementable proposals** - Synthesizes research findings into unified architecture proposals
- **Executes experiments and analyzes results** - Runs benchmarks and collects metrics automatically
- **Iteratively improves based on findings** - Identifies bottlenecks and triggers new evolution cycles when targets aren't met

### 1.2 Core Philosophy: The Evolve Methodology

The "Evolve" methodology breaks down complex ML research into manageable, iterative cycles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLVE METHODOLOGY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │
│   │ Codebase  │─▶│ Research  │─▶│   Deep    │─▶│ Proposal  │─▶│  Merge   │ │
│   │ Understand│  │ Planning  │  │ Research  │  │Generation │  │Synthestic│ │
│   └───────────┘  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │
│                                                                      │      │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐                       │      │
│   │ Analysis  │◀─│Experiment │◀─│Implement- │◀──────────────────────┘      │
│   │           │  │           │  │  ation    │                               │
│   └─────┬─────┘  └───────────┘  └───────────┘                               │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    EVOLVE TRIGGER DECISION                           │   │
│   │  If targets NOT met: Start new iteration with updated context        │   │
│   │  If targets met: Complete with final summary                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
1. **Iterative Refinement** - Each cycle builds on insights from previous iterations
2. **Human-in-the-Loop** - Critical decision points allow human researchers to guide the process
3. **Structured Research** - Systematic approach using reasoning patterns rather than ad-hoc exploration
4. **Measurable Progress** - Clear metrics and targets determine when to iterate vs. complete

---

## 2. System Architecture

### 2.1 Layered Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RANKEVOLVE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         USER INTERFACE LAYER                            ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │  Chatbot     │  │Human-in-Loop │  │  Progress    │                  ││
│  │  │  Interface   │  │  Controls    │  │  Dashboard   │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ORCHESTRATION LAYER                             ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │   Flow       │  │   Session    │  │   Context    │                  ││
│  │  │   Engine     │  │   Manager    │  │   Manager    │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         RESEARCH ENGINE LAYER                           ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ ││
│  │  │  Codebase    │  │   Query      │  │    Deep      │  │  Proposal   │ ││
│  │  │  Analyzer    │  │  Generator   │  │  Research    │  │  Generator  │ ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │Architecture  │  │    Code      │  │  Experiment  │                  ││
│  │  │ Synthesizer  │  │ Implementer  │  │   Executor   │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         AI/LLM INFERENCE LAYER                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │   Claude     │  │   ChatGPT    │  │   MetaMate   │                  ││
│  │  │   Sonnet     │  │     o1       │  │   (Internal) │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         EXECUTION LAYER                                 ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │  DevServer   │  │   GPU        │  │   Results    │                  ││
│  │  │  Execution   │  │   Cluster    │  │  Collector   │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 User Interface Layer

The User Interface Layer provides an interactive chatbot experience with real-time progress visualization.

**Chatbot Interface:**
- Conversational interaction for configuring and controlling research workflows
- Real-time message streaming via WebSocket connections
- Support for viewing generated documentation and code artifacts

**Human-in-the-Loop Controls:**
- Approve, modify, or add research directions at critical decision points
- Inject domain expertise to guide the AI's research focus
- Override automated decisions when human judgment is preferred

**Progress Dashboard:**
- Step-by-step visualization of the current evolution cycle
- Collapsible progress sections showing parallel research streams
- Iteration tracking with metrics comparison across cycles

**Research Mode Options:**
| Mode | Description | Use Case |
|------|-------------|----------|
| 🚀 Paradigm-Shifting | Aggressive exploration of novel approaches | Breakthrough research |
| 🔬 Systematic | Methodical improvement within existing paradigms | Incremental optimization |
| 🎯 Targeted | Focused problem-solving for specific issues | Bug fixes, known bottlenecks |
| 🔍 Exploratory | Open-ended investigation | Early-stage research |

### 2.3 Orchestration Layer

The Orchestration Layer manages workflow execution, session state, and context across evolution cycles.

**Flow Engine:**
- Declarative workflow definitions supporting multi-step, multi-iteration research
- Configurable step sequencing with dependency management
- Support for parallel execution of independent research streams
- Automatic state persistence and recovery

**Session Manager:**
- User session tracking across long-running research workflows
- Variable storage for user inputs and intermediate results
- Input field management for collecting configuration and feedback

**Context Manager:**
- Maintains iteration context across evolution cycles
- Carries forward key insights, bottleneck analyses, and successful approaches
- Manages file-based artifacts (documentation, proposals, code, results)
- Enables informed decision-making by preserving historical context

### 2.4 Research Engine Layer

The Research Engine Layer contains the core AI-powered components that drive the research process.

**Codebase Analyzer:**
- Parses and understands ML codebases comprehensively
- Generates structured documentation (RST format for Sphinx)
- Identifies architectural patterns, extension points, and modification candidates
- Extracts key abstractions and interfaces for targeted research

**Query Generator:**
- Uses 5 structured reasoning patterns to generate research directions:
  1. **Efficiency & Scalability** - Computational optimizations, memory reduction
  2. **Cold-Start & Transfer Learning** - New user/item handling, cross-domain transfer
  3. **Temporal Modeling** - Time-aware representations, decay mechanisms
  4. **User-Item Interaction** - Cross-sequence patterns, behavioral modeling
  5. **Training & Optimization** - Loss functions, regularization, convergence

**Deep Research Module:**
- Orchestrates parallel research across multiple AI models
- External research via ChatGPT, Claude, and DeepSeek for academic literature and open-source knowledge
- Internal research via MetaMate for proprietary documentation and internal best practices
- Aggregates and synthesizes findings from multiple sources

**Proposal Generator:**
- Generates detailed proposals for each research direction
- Creates implementable design documents with clear specifications
- Identifies potential risks and mitigation strategies

**Architecture Synthesizer:**
- Merges multiple proposals into a coherent unified architecture
- Resolves conflicts between competing approaches
- Generates named research frameworks with clear branding (e.g., "SYNAPSE")
- Ensures architectural consistency and integration feasibility

**Code Implementer:**
- Generates implementation plans with clear milestones
- Produces runnable code that integrates with the target codebase
- Maintains consistency with existing coding patterns and conventions

**Experiment Executor:**
- Automates experiment setup and execution on GPU clusters
- Handles training job submission, monitoring, and result collection
- Formats metrics for analysis and comparison

### 2.5 AI/LLM Inference Layer

The AI/LLM Inference Layer provides multi-model orchestration for different research tasks.

**Model Allocation Strategy:**
| Model | Primary Use | Strengths |
|-------|-------------|-----------|
| Claude Sonnet | Primary reasoning, synthesis | Strong analysis, code generation |
| ChatGPT o1 | Deep research, literature review | Academic knowledge, reasoning chains |
| MetaMate | Internal knowledge search | Proprietary documentation, internal practices |
| DeepSeek | Alternative reasoning | Cost-effective, diverse perspectives |

**Orchestration Patterns:**
- **Parallel Research**: Multiple models research simultaneously, results synthesized
- **Sequential Refinement**: One model's output feeds into another for deeper analysis
- **Consensus Building**: Multiple models validate critical decisions
- **Fallback Chains**: Automatic failover to alternative models on errors

### 2.6 Execution Layer

The Execution Layer handles the computational infrastructure for running experiments.

**DevServer Execution:**
- Environment setup and dependency management
- Code deployment and execution orchestration
- Log collection and error handling

**GPU Cluster Integration:**
- Resource allocation and scheduling
- Multi-GPU training job management
- Checkpoint management and recovery

**Results Collector:**
- Metrics collection (NDCG, HR, throughput, latency)
- Automatic formatting for analysis step
- Historical result storage for cross-iteration comparison

---

## 3. Research Workflow

### 3.1 Evolution Cycle Overview

Each evolution cycle follows a structured 8-step process:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVOLUTION CYCLE STEPS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Step 1  │───▶│ Step 2  │───▶│ Step 3  │───▶│ Step 4  │                  │
│  │Codebase │    │Research │    │  Deep   │    │Proposal │                  │
│  │ Analyze │    │Planning │    │Research │    │Generate │                  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                  │
│                                                      │                       │
│                                                      ▼                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Step 8  │◀───│ Step 7  │◀───│ Step 6  │◀───│ Step 5  │                  │
│  │Analysis │    │Experim- │    │Implement│    │Architect│                  │
│  │& Decide │    │  ents   │    │  Code   │    │Synthestic│                  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Iteration 1: Full Research Cycle

| Step | Name | Description | Output |
|------|------|-------------|--------|
| 0 | Welcome & Plan Selection | User configures codebase and research mode | Configuration |
| 1 | Codebase Investigation | Analyze target codebase (e.g., HSTU) | Documentation |
| 2 | Research Planning | Generate targeted queries using 5 reasoning patterns | Query list |
| 3 | Deep Research Execution | Parallel external + internal research | Research results |
| 4 | Proposal Generation | Per-direction proposals | Individual proposals |
| 5 | Unified Architecture Synthesis | Merge into unified proposal | Design document |
| 6 | Implementation | Code generation | Python modules |
| 7 | Experiments | Run benchmarks on GPU cluster | Metrics |
| 8 | Analysis | Identify bottlenecks, decide next steps | Iteration decision |

### 3.3 Iteration 2+: Targeted Refinement

Subsequent iterations focus on addressing specific bottlenecks identified in previous cycles:

**Example: Iteration 2 Focus Areas (from SYNAPSE demo)**
- **Multi-Timescale FLUID**: Replacing fixed τ=24h with learnable decay timescales
- **Enhanced Multi-Token v2**: Adding GQA + sparse attention for efficiency

The cycle structure remains the same, but:
- Research planning is informed by previous iteration's analysis
- Queries are more targeted to specific bottlenecks
- Proposals build on proven approaches from earlier iterations
- Experiments include ablation studies comparing to previous versions

### 3.4 Human-in-the-Loop Intervention Points

Critical intervention points where human researchers can guide the process:

| Step | Intervention Type | Purpose |
|------|-------------------|---------|
| Research Planning | Add/Modify/Prioritize | Inject domain expertise, focus on promising directions |
| Deep Research | Request Additional | Fill gaps in AI-generated research |
| Proposal Review | Approve/Modify | Ensure proposals align with practical constraints |
| Experiment Config | Adjust Parameters | Fine-tune hyperparameters, add baselines |
| Analysis | Override Decision | Force iteration or completion based on external factors |

**Example Intervention:**
> "We found our deep research results are missing latest developments in quasi-sequence interaction. We therefore ask it to take a look again."
>
> Through additional research, RankEvolve obtained updates that proved helpful in Iteration 2.

---

## 4. Data Flow & Integration Patterns

### 4.1 Workflow Orchestration

The Flow Engine uses a declarative configuration approach:

**Step Configuration Elements:**
- **Pre-messages**: Status indicators shown before processing
- **Progress sections**: Parallel streams of work with animated progress
- **Post-messages**: Results and summaries after completion
- **Suggested actions**: User options for next steps (continue, modify, add)
- **Input fields**: User input collection (queries, parameters, feedback)

**Step Transitions:**
- **Automatic**: Proceed when step completes without user action
- **Wait for user**: Pause for human review and approval
- **Conditional**: Branch based on analysis results or user input

### 4.2 Context Management

Context flows across iterations through a layered approach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT HIERARCHY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SESSION CONTEXT                                 ││
│  │  - User configuration (codebase, mode, custom instructions)             ││
│  │  - Session variables and input field values                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ITERATION CONTEXT                               ││
│  │  - Research queries and results                                         ││
│  │  - Proposals and synthesized architecture                               ││
│  │  - Implementation code and experiment results                           ││
│  │  - Analysis insights and bottleneck identification                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         EVOLUTION CONTEXT                               ││
│  │  - Cross-iteration trajectory and progress                              ││
│  │  - Cumulative insights and proven approaches                            ││
│  │  - Final metrics comparison and summary                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Multi-Model AI Coordination

**Research Phase Coordination:**
1. Query Generator creates research directions
2. Deep Research module dispatches queries to multiple models in parallel
3. Results are aggregated and deduplicated
4. Synthesis model (Claude) creates unified research summary

**Proposal Phase Coordination:**
1. Each research direction generates a proposal independently
2. Architecture Synthesizer identifies common patterns and conflicts
3. Unified proposal merges best approaches with conflict resolution
4. Human review point allows adjustments before implementation

---

## 5. Research Output: SYNAPSE Architecture

### 5.1 Architecture Overview

SYNAPSE (Sequential Yield Network via Analytical Processing and Scalable Embeddings) emerged from RankEvolve's research cycles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYNAPSE ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │     PRISM      │    │   SSD-FLUID    │    │  Multi-Token   │            │
│  │   Perception   │───▶│     Memory     │───▶│   Aggregation  │            │
│  │     Layer      │    │    Backbone    │    │     Layer      │            │
│  └────────────────┘    └────────────────┘    └────────────────┘            │
│                                                                              │
│  Components:                                                                 │
│  - SSD-FLUID: O(N) training, O(1) inference via State Space Duality        │
│  - PRISM: User-conditioned polysemous item embeddings                       │
│  - FLUID: Continuous-time temporal decay with learnable τ                   │
│  - Multi-Token: Enhanced cross-sequence interaction                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Key Results

| Model | NDCG@10 | Δ vs HSTU | Δ vs Sampling | Throughput |
|-------|---------|-----------|---------------|------------|
| **HSTU (full attention)** | 0.1823 | — | — | 1× |
| **HSTU + linear-decay sampling** | 0.1626 | -10.8% | — | 1.6× |
| **SYNAPSE v1** | 0.1654 | -9.3% | **+1.5%** | 2.0× |
| **SYNAPSE v2** | 0.1729 | -5.2% | **+6.3%** | 2.3× |

### 5.3 Evolution Trajectory

**Iteration 1 → Iteration 2 Improvements:**

| Aspect | Iteration 1 | Iteration 2 | Improvement |
|--------|-------------|-------------|-------------|
| Temporal Modeling | Fixed τ=24h | Learnable 3-tier τ (fast/medium/slow) | Adaptive to item types |
| Multi-Token Attention | Dense (K=8) | GQA + sparse (K=4) | -50% compute, same quality |
| Temporal Item Performance | -2.5% vs baseline | +0.5% vs baseline | Recovered from negative |
| Latency Overhead | +35% | +16% | Reduced by half |

---

## 6. Automation Capabilities & Roadmap

### 6.1 Current Capabilities

| Capability | Status | Description |
|------------|--------|-------------|
| Codebase Understanding | ✅ Automated | Parses and documents ML codebases |
| Research Planning | ✅ Automated | Generates structured research queries |
| Deep Research Execution | ✅ Automated | Multi-model parallel research |
| Proposal Generation | ✅ Automated | Per-direction and unified proposals |
| Architecture Synthesis | ✅ Automated | Merges proposals into coherent design |
| Code Implementation | ✅ Automated | Generates runnable Python code |
| Experiment Execution | ✅ Automated | DevServer training and evaluation |
| Analysis & Iteration | ✅ Automated | Bottleneck identification and decision |

### 6.2 Future Enhancements

**Phase 1: Enhanced Experiment Automation**
- Automatic hyperparameter search
- Multi-dataset parallel evaluation
- Ablation study automation
- Statistical significance testing

**Phase 2: Extended Research Capabilities**
- Patent and paper search integration
- Broader external research sources
- Code repository mining
- Automated literature review

**Phase 3: Production Integration**
- A/B test design automation
- Production metric monitoring
- Rollout recommendation
- Regression detection

---

## 7. Appendix

### 7.1 Demo Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEMO FLOW: 2 ITERATIONS, 17 STEPS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ITERATION 1 (Steps 0-8):                                                   │
│  ───────────────────────────────────────────────────────────────────────── │
│  0. Welcome & Plan Selection (paradigm-shifting mode)                       │
│  1. Codebase Investigation → Documentation generated                        │
│  2. Research Planning → 4 research directions identified                    │
│  3. Deep Research → External + internal research completed                  │
│  4. Proposal Generation → Per-direction proposals                           │
│  5. Architecture Synthesis → SYNAPSE v1 unified proposal                   │
│  6. Implementation → Python code generated                                  │
│  7. Experiments → Results: +1.5% vs sampling, 2× throughput                │
│  8. Analysis → Fixed τ=24h identified as bottleneck                        │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════   │
│                              EVOLVE TRIGGER                                  │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ITERATION 2 (Steps 9-17):                                                  │
│  ───────────────────────────────────────────────────────────────────────── │
│  9. Evolve Transition → Context from Iteration 1                           │
│  10. Research Planning → Focus on temporal + efficiency                    │
│  11. Deep Research → Multi-timescale + efficient attention                 │
│  12. Proposal Generation → SYNAPSE v2 enhancements                         │
│  13. Architecture Synthesis → Learnable τ + GQA Multi-Token                │
│  14. Implementation → Enhanced modules                                      │
│  15. Experiments → Results: +6.3% vs sampling, 2.3× throughput             │
│  16. Analysis → All targets met!                                            │
│  17. Final Summary → Complete                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Key Metrics Summary

| Metric | HSTU+Sampling | SYNAPSE v1 | SYNAPSE v2 | v2 vs Baseline |
|--------|---------------|------------|------------|----------------|
| NDCG@10 | 0.1626 | 0.1654 (+1.5%) | **0.1729** (+6.3%) | **+6.3%** ✅ |
| Throughput | 1.6× | 2.0× (+25%) | **2.3×** (+44%) | **+44%** ✅ |
| Temporal Items | baseline | -2.5% | **+0.5%** | **Recovered** ✅ |
| Latency Overhead | — | +35% | **+16%** | **Improved** ✅ |

### 7.3 References

- **HSTU**: "Actions Speak Louder than Words" (ICML 2024)
- **Mamba-2**: "State Space Duality" (ICML 2024)
- **generative-recommenders**: GitHub open-source codebase
- **wukong-recommendation**: Scaling laws for recommendation

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Based on: Public Demo (MovieLens-20M, Amazon Books)*
