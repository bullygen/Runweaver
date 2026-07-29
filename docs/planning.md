# Experiment planning

`ParameterSpace` supports float/integer (linear or log), categorical, boolean,
ordinal, fixed, derived and conditional parameters. Every sampled parameter has
unit-interval transforms, optional quantization, description/unit, activation
condition and hash participation.

Built-in planners:

- grid and full factorial;
- seeded random;
- SciPy Latin hypercube, Sobol and Halton;
- imported/exported design matrices;
- optional Optuna TPE, CMA-ES, GP and NSGA-II.

Constraints use a small data-only expression language: no function calls or
arbitrary Python execution. Replicates receive derived deterministic seeds.
Optuna trial identity is stored as plan metadata; the public domain object
remains immutable.
