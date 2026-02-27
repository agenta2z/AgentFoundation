"""
Domain taxonomy for knowledge classification.

This module defines the domain hierarchy and associated tags used for
classifying knowledge pieces. The taxonomy enables:
1. Retrieval filtering: Find pieces by domain and/or tags
2. LLM-guided classification: Structured prompts for automatic categorization
3. Validation: Ensure pieces use valid domains and tags

Domain Categories:
- Model-Related: model_optimization, model_architecture, feature_engineering
- Training/Inference: training_efficiency, inference_efficiency
- Data and Evaluation: data_engineering, model_evaluation
- Infrastructure: infrastructure
- Engineering: debugging, testing, workflow
- General: general (fallback category)

Each domain has:
- description: Human-readable description for LLM prompts
- tags: List of specific topics within the domain
"""

from typing import Any, Dict, List


DOMAIN_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # ── Model-Related ──
    "model_optimization": {
        "description": "Performance optimization for model training and inference",
        "tags": [
            "flash-attention",
            "kernel-fusion",
            "memory-optimization",
            "graph-breaks",
            "dynamic-shapes",
            "cuda-optimization",
            "h100",
            "gpu-utilization",
            "profiling",
            "benchmarking",
            "numerical-stability",
            "cfl-ramp",
            "convergence",
        ],
    },
    "model_architecture": {
        "description": "Model structure, layers, and architectural choices",
        "tags": [
            "attention",
            "transformer",
            "embedding",
            "mlp",
            "encoder",
            "decoder",
            "residual",
            "normalization",
            "positional-encoding",
            "activation",
        ],
    },
    "feature_engineering": {
        "description": "Feature processing, embeddings, and data transformation",
        "tags": [
            "sparse-features",
            "dense-features",
            "embedding-tables",
            "feature-interaction",
            "bucketing",
            "normalization",
            "sequence-features",
            "id-features",
        ],
    },
    # ── Training/Inference ──
    "training_efficiency": {
        "description": "Training speed, memory, and resource utilization",
        "tags": [
            "distributed-training",
            "gradient-accumulation",
            "mixed-precision",
            "checkpoint",
            "data-loading",
            "batch-size",
            "learning-rate",
            "warmup",
            "scheduler",
        ],
    },
    "inference_efficiency": {
        "description": "Inference latency, throughput, and deployment",
        "tags": [
            "quantization",
            "pruning",
            "distillation",
            "batching",
            "caching",
            "serving",
            "latency",
            "throughput",
        ],
    },
    # ── Data and Evaluation ──
    "data_engineering": {
        "description": "Data pipelines, preprocessing, and dataset management",
        "tags": [
            "data-pipeline",
            "preprocessing",
            "data-loading",
            "dataset",
            "sampling",
            "augmentation",
            "feature-store",
        ],
    },
    "model_evaluation": {
        "description": "Metrics, evaluation strategies, and testing",
        "tags": [
            "metrics",
            "offline-eval",
            "online-eval",
            "a-b-testing",
            "regression-testing",
            "benchmark",
            "baseline-comparison",
        ],
    },
    # ── Infrastructure ──
    "infrastructure": {
        "description": "Cluster management, job scheduling, and resource allocation",
        "tags": [
            "cluster",
            "gpu-allocation",
            "job-scheduling",
            "slurm",
            "distributed-systems",
            "networking",
            "storage",
        ],
    },
    # ── Engineering ──
    "debugging": {
        "description": "Troubleshooting, error analysis, and diagnostics",
        "tags": [
            "error-analysis",
            "gradient-issues",
            "nan-detection",
            "memory-leak",
            "performance-regression",
            "logging",
            "tracing",
        ],
    },
    "testing": {
        "description": "Testing strategies and validation approaches",
        "tags": [
            "unit-test",
            "integration-test",
            "regression-test",
            "benchmark",
            "validation",
            "numerical-verification",
        ],
    },
    "workflow": {
        "description": "End-to-end procedures and multi-step processes",
        "tags": [
            "setup",
            "migration",
            "upgrade",
            "deployment",
            "experiment-tracking",
            "reproducibility",
            "ci-cd",
        ],
    },
    # ── General (fallback) ──
    "general": {
        "description": "General knowledge that doesn't fit other categories",
        "tags": [
            "background",
            "reference",
            "definition",
            "concept",
            "best-practice",
            "guideline",
        ],
    },
}


def get_all_domains() -> List[str]:
    """Return list of all valid domain names."""
    return list(DOMAIN_TAXONOMY.keys())


def get_domain_tags(domain: str) -> List[str]:
    """Return tags for a given domain.

    Args:
        domain: The domain name to get tags for.

    Returns:
        List of tag strings for the domain.

    Raises:
        ValueError: If domain is not in the taxonomy.
    """
    if domain not in DOMAIN_TAXONOMY:
        raise ValueError(
            f"Unknown domain: '{domain}'. Valid domains: {get_all_domains()}"
        )
    return DOMAIN_TAXONOMY[domain]["tags"]


def validate_domain(domain: str) -> bool:
    """Check if a domain is valid.

    Args:
        domain: The domain name to validate.

    Returns:
        True if domain is in the taxonomy, False otherwise.
    """
    return domain in DOMAIN_TAXONOMY


def validate_tags(domain: str, tags: List[str]) -> bool:
    """Validate that all tags are valid for the given domain.

    Args:
        domain: The domain to validate tags against.
        tags: List of tags to validate.

    Returns:
        True if all tags are valid for the domain, False otherwise.

    Raises:
        ValueError: If domain is not in the taxonomy.
    """
    if domain not in DOMAIN_TAXONOMY:
        raise ValueError(
            f"Unknown domain: '{domain}'. Valid domains: {get_all_domains()}"
        )
    allowed_tags = set(DOMAIN_TAXONOMY[domain]["tags"])
    return all(tag in allowed_tags for tag in tags)


def format_taxonomy_for_prompt() -> str:
    """Format the taxonomy as a string for inclusion in LLM prompts.

    Returns:
        Formatted string containing every domain name and its tags.
    """
    lines = []
    for domain, info in DOMAIN_TAXONOMY.items():
        lines.append(f"- **{domain}**: {info['description']}")
        tags_str = ", ".join(info["tags"])
        lines.append(f"  Tags: {tags_str}")
    return "\n".join(lines)
