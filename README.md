# MultiGridBarrier

# Author: Sébastien Loisel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/MultiGridBarrier.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/MultiGridBarrier.jl/dev/)
[![Build Status](https://github.com/sloisel/MultiGridBarrier.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/MultiGridBarrier.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sloisel/MultiGridBarrier.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/MultiGridBarrier.jl)

This package solves convex variational problems, e.g. nonlinear PDEs and BVPs, using the MultiGrid Barrier method (with either finite elements or spectral elements), which is theoretically optimal for some problem classes.

## Citation

If you use this package in your research, please cite:

> S. Loisel, "The spectral barrier method to solve analytic convex optimization problems in function spaces," *Numerische Mathematik*, 2025. DOI: [10.1007/s00211-025-01508-0](https://doi.org/10.1007/s00211-025-01508-0)

BibTeX:
```bibtex
@article{Loisel2025,
  author = {Loisel, Sébastien},
  title = {The spectral barrier method to solve analytic convex optimization problems in function spaces},
  journal = {Numerische Mathematik},
  year = {2025},
  doi = {10.1007/s00211-025-01508-0}
}
```
