Synthetic Data Generation
=========================

This page documents the synthetic data generation capabilities for creating
billion-scale datasets for large-scale experimentation.

Overview
--------

The codebase provides tools to generate synthetic large-scale datasets using
the **Fractal Expansion** algorithm from arXiv:1901.08910. This allows scaling
MovieLens-20M to billion-scale datasets while preserving statistical properties.

Fractal Expansion Algorithm
---------------------------

The algorithm works by:

1. Analyzing the user-item interaction graph structure
2. Creating fractal copies that preserve local patterns
3. Expanding both user and item spaces proportionally
4. Maintaining degree distributions and clustering coefficients

**Reference**::

    @article{shin2019fractal,
      title={Synthesizing Labeled Graph Data with Fractal Graph Expansion},
      author={Shin, Kijung and others},
      journal={arXiv:1901.08910},
      year={2019}
    }

Running Synthetic Data Generation
---------------------------------

Use ``run_fractal_expansion.py`` to generate datasets::

    # Generate ML-3B (3 billion ratings)
    python run_fractal_expansion.py --target-scale 3B

    # Generate ML-13B (13 billion ratings)
    python run_fractal_expansion.py --target-scale 13B

    # Generate ML-18B (18 billion ratings)
    python run_fractal_expansion.py --target-scale 18B

Available Scales
----------------

+----------+----------------+----------------+-------------------+
| Scale    | Approx. Users  | Approx. Items  | Approx. Ratings   |
+==========+================+================+===================+
| ML-3B    | ~150M          | ~5M            | ~3 billion        |
+----------+----------------+----------------+-------------------+
| ML-13B   | ~650M          | ~20M           | ~13 billion       |
+----------+----------------+----------------+-------------------+
| ML-18B   | ~900M          | ~30M           | ~18 billion       |
+----------+----------------+----------------+-------------------+

Prerequisites
-------------

1. **Base Dataset**: MovieLens-20M must be preprocessed first::

       python preprocess_public_data.py

2. **Disk Space**: Ensure sufficient storage:

   - ML-3B: ~50GB
   - ML-13B: ~200GB
   - ML-18B: ~300GB

3. **Memory**: Expansion is memory-intensive:

   - ML-3B: ~64GB RAM
   - ML-13B: ~256GB RAM
   - ML-18B: ~512GB RAM

Output Structure
----------------

Generated data follows the same format as preprocessed public datasets::

    data/
    └── ml-3b/
        ├── ml-3b.txt
        ├── ml-3b_train.txt
        ├── ml-3b_valid.txt
        └── ml-3b_test.txt

Use Cases
---------

**Scaling Law Studies**

The synthetic datasets enable studying how model performance scales with:

- Dataset size (users, items, interactions)
- Model capacity (layers, dimensions)
- Compute budget (training time, batch size)

**Production Simulation**

Billion-scale datasets simulate production workloads for:

- Memory efficiency testing
- Distributed training validation
- Inference latency benchmarking

**Research Reproducibility**

Since the fractal expansion is deterministic given the same seed, results
can be reproduced across different research groups.

Limitations
-----------

1. **Statistical Approximation**: Synthetic data preserves structure but not
   semantic meaning of interactions.

2. **Cold Start**: Fractal copies may not accurately represent cold-start
   scenarios present in real data.

3. **Temporal Patterns**: Time-based patterns from original data may be
   distorted in expanded versions.

4. **Computation Cost**: Generation itself can take several hours for
   billion-scale datasets.

Custom Scale Generation
-----------------------

For custom scales, modify the expansion parameters::

    from fractal_expansion import FractalExpander

    expander = FractalExpander(
        base_dataset="ml-20m",
        target_users=500_000_000,
        target_items=10_000_000,
        seed=42
    )
    expander.expand(output_path="data/ml-custom/")
