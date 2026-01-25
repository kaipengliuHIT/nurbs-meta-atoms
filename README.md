# NURBS Meta-Atoms Transformer Model

A configurable Transformer-based surrogate model for NURBS (Non-Uniform Rational B-Splines) meta-atom optimization. Replaces conventional FDTD numerical solvers with parallel accelerated deep learning inference.

## Features

- **Flexible Architecture**: Configurable attention heads, encoder/decoder layers, model dimensions
- **Binary Grid Encoding**: Efficient discretization of NURBS control points
- **Dual Output**: Predicts optical response (phase/amplitude) and parametric gradients
- **End-to-End Differentiable**: Enables automatic differentiation for structural optimization
- **MEEP Integration**: Compatible with FDTD simulation data generation

## Default Configuration

| Parameter | Default | Configurable |
|-----------|---------|--------------|
| **Architecture** | Encoder-Decoder Transformer | ✓ |
| **Attention Heads** | 12 | ✓ |
| **Encoder/Decoder Layers** | 8 | ✓ |
| **Model Dimension** | 384 | ✓ |
| **Optimizer** | Adam | ✓ |
| **Learning Rate** | 5×10⁻⁵ | ✓ |
| **Control Point Encoding** | Binary grid indices | ✓ |
| **Spectrum Range** | 400-700nm | ✓ |

## Project Structure

```
nurbs-meta-atoms/
├── meta_transformer.py              # Main Encoder-Decoder Transformer model
├── train_model.py                   # Training script
├── nurbs_atoms_data.py              # NURBS meta-atom FDTD simulation with MEEP
├── generate_training_data_parallel.py # Parallel data generation
├── inference_transformer_model.py   # Inference and evaluation
├── metalens_optimization.py         # Metalens optimization
├── example_usage.py                 # Example usage scripts
├── visualize_field.py               # Field visualization
└── training_data/                   # Training data directory
```

## Requirements

```bash
pip install numpy torch matplotlib scikit-learn scipy tqdm meep
```

## Data Format

Training data consists of:
- **Input**: Control points `(n_samples, 8, 2)` - 8 control points with (x, y) coordinates normalized to [0, 1]
- **Wavelengths**: `(n_samples,)` - wavelength values in nm (400-700nm range)
- **Phase**: `(n_samples,)` - phase response in radians [-π, π]
- **Transmittance**: `(n_samples,)` - amplitude/transmittance [0, 1]

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **R²** | Coefficient of determination |

## FDTD Data Generation

Training data is generated using MEEP FDTD simulations:
```bash
python generate_training_data_parallel.py --n_samples 500000 --n_workers 8
```

## License

MIT License
