# Graph Autoencoder Optimization with Optuna

This repository contains implementations of Graph Autoencoder models with automated hyperparameter tuning using Optuna. The project includes synthetic data generation and progressively deeper architectures (2 to 5 layers).

## Key Features

1. **Synthetic Data Generation**
   - `data_generation.ipynb`: Creates artificial graph datasets for model testing

2. **Optimized Graph Autoencoders** 
   - Progressive architectures from 2 to 5 layers
   - Automated hyperparameter tuning via Optuna
   - Comparative analysis across model depths

## How to cite

If you use this repository or its data/code in your research, please cite the following paper:
> **Ximena Pocco**, Waqar Hassan, Karelia Salinas, Vladimir Molchanov, Luis G. Nonato, *Exploring Urban Factors with Autoencoders: Relationship Between Static and Dynamic Features*, SIBGRAPI Conference on Graphics, Patterns and Images, 2025, DOI: [https://arxiv.org/abs/2509.06167](https://arxiv.org/abs/2509.06167)

### BibTeX
```bibtex
@article{pocco2025exploring,
  title={Exploring Urban Factors with Autoencoders: Relationship Between Static and Dynamic Features},
  author={Pocco, Ximena and Hassan, Waqar and Salinas, Karelia and Molchanov, Vladimir and Nonato, Luis G},
  journal={arXiv preprint arXiv:2509.06167},
  year={2025}
}
```
