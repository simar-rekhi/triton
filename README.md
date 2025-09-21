# Triton: Machine Learning Compilation with GPU Kernel Programming

A Python DSL-based project leveraging Triton for hyper-fast GPU kernel programming and machine learning compilation. This repository showcases starter packs for deploying Triton kernels, performing parallel efficiency checks, and a personal setup tool bridging Google Colab with GitHub version control.

## Overview

This project explores compilation and deployment of machine learning workloads on GPUs using Triton, a CUDA-inspired Python DSL designed for writing highly efficient GPU kernels. It demonstrates practical usage of CUDA kernel programming abstractions with a focus on performance optimization for parallel computations. In addition, it features an innovative `bridge_setup.ipynb` tool facilitating seamless Git operations within Google Colab environments.

## My Contribution

This repository encapsulates my individual efforts to:

- Understand and apply advanced GPU kernel programming using Triton's Python DSL
- Create starter packs that benchmark and validate parallel computation efficiency on GPU hardware
- Develop a custom Jupyter Notebook bridge (`bridge_setup.ipynb`) to enable Git version control directly within Google Colab notebooks, improving research and development workflows
- Continually expand the project with ongoing GPU kernel optimizations and compiler-level enhancements

All work is independently developed to demonstrate:

- GPU programming fundamentals grounded in CUDA
- Parallel algorithm design and efficiency measurement
- Integration between cloud-based development platforms and source control systems
- Hands-on experience with emerging machine learning compilation tools

## Lab Affiliation

This work is being done under Professor Dr. Wei Yang's Deep Learning Lab (2025) and the Machine Learning Compilation Sub-group principally investigated by Tingxi Li.

## Getting Started

### Prerequisites

- Python 3.8+
- Triton library installed (`pip install triton`)
- Triton Language module installed (`pip install triton.language`)
- Torch library installed (`pip install torch`)
- NVIDIA GPU with CUDA Toolkit installed
- Google Colab environment (for the bridge setup notebook)

### Usage
#### To run the starter pack GPU kernels and tests:
```bash
python starter_pack.ipynb
```

#### To use the Google Colab GitHub bridge:

1. Open `bridge_setup.ipynb`.
2. Follow instructions to configure a seamless Git environment inside Colab.
3. Use Git commands in your Colab notebooks as usual, with standard version control workflows.

## Project Status

This project is currently in active development. GPU kernel programming components and Triton DSL mastery are being refined continuously. The bridge setup notebook has been tested and is ready for use.

## License

This project is licensed under the MIT License.

## Citation

```bibtex
@misc{srekhi2025triton,
author = {Simar Rekhi},
title = {Triton: Machine Learning Compilation with GPU Kernel Programming},
year = {2025},
note = {Standalone contribution to Machine Learning Compilation Lab},
howpublished = {\url{https://github.com/simar-rekhi/triton}}
}
```

