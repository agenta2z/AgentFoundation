"""First-class workflow framework — two-layer architecture.

Layer 1: StateGraph (static blueprint) — defines phases, dependencies, gotos, gates.
Layer 2: WorkGraph (runtime executor) — SOPWorkGraphNode bridges the layers.
"""
