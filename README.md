# NURBS Meta-Atoms for Metalens Design

This project implements a neural network-based approach for designing NURBS (Non-Uniform Rational B-Splines) meta-atoms for metalens applications, inspired by the DFLAT (Deep Focus via Learned Aperture Tuning) methodology. The system uses transformer-based surrogate models to optimize the geometric parameters of meta-atoms to achieve desired optical responses (phase and transmittance).

## Overview

The project consists of several components:

1. **NURBS Meta-Atom Simulation**: Physics-based simulation using Meep to compute optical properties
2. **Transformer Surrogate Model**: Neural network model to predict optical response from geometric parameters
3. **Metalens Optimization**: Design optimization using the surrogate model
4. **Data Generation**: Tools for creating training datasets

## Project Structure

```
nurbs-meta-atoms/
├── nurbs_atoms_data.py        # NURBS meta-atom simulation with Meep
├── transformer_nurbs_model.py # Transformer-based surrogate model
├── train_transformer_model.py # Training script for the surrogate model
├── inference_transformer_model.py # Inference and evaluation script
├── metalens_optimization.py   # Metalens optimization using surrogate model
├── example_usage.py          # Example usage scripts
├── dflatdata.ipynb          # Jupyter notebook with example data
└── README.md               # This file
```

## Features

### NURBS Meta-Atom Simulation
- Implementation of NURBS curves for meta-atom geometry
- Electromagnetic simulation using Meep FDTD solver
- Calculation of phase and transmittance for given geometric parameters
- Support for various NURBS shapes through control points

### Transformer Surrogate Model
- Transformer-based architecture for predicting optical response
- Input: Control points coordinates of NURBS meta-atoms
- Output: Phase and transmittance values
- Handles sequence-to-value mapping efficiently
- Attention mechanism captures complex geometric relationships

### Metalens Optimization
- Optimization of metalens focusing properties
- Uses surrogate model for fast evaluation
- Implements phase profile matching for focusing applications
- Calculates focusing efficiency metrics

## Requirements

```bash
pip install numpy torch matplotlib scikit-learn scipy tqdm meep
```

## Usage

### 1. Training the Surrogate Model

```bash
# Generate training data and train the transformer model
python train_transformer_model.py --train
```

### 2. Model Inference

```bash
# Test the trained model
python inference_transformer_model.py
```

### 3. Metalens Optimization

```bash
# Optimize metalens design using the surrogate model
python metalens_optimization.py
```

### 4. Example Usage

```bash
# Run example workflows
python example_usage.py
```

## Model Architecture

The transformer-based surrogate model consists of:

- **Input Projection**: Maps 2D control point coordinates to model dimension
- **Positional Encoding**: Adds positional information to sequence
- **Transformer Encoder**: Multiple layers of multi-head attention
- **Output Projection**: Maps to 2D output (phase, transmittance)

### Key Parameters:
- Input: 8 control points × 2 coordinates each
- Model dimension: 128
- Attention heads: 8
- Transformer layers: 4
- Output: 2 values (phase, transmittance)

## DFLAT-Style Metalens Design

The optimization follows the DFLAT methodology:

1. **Phase Profile Calculation**: Compute ideal phase distribution for focusing
2. **Segment Optimization**: Optimize each annular segment independently
3. **Surrogate Model Usage**: Fast evaluation using trained neural network
4. **Efficiency Calculation**: Quantify focusing performance

### Ideal Phase Profile
For a metalens with focal length `f` and radial position `r`:
```
φ(r) = -k * (sqrt(r² + f²) - f)
```
where `k = 2π/λ` is the wave number.

## Data Format

Training data consists of:
- **Input**: Control points coordinates as (N, 2) array where N is number of control points
- **Output**: [phase, transmittance] as (2,) array
- **Phase**: Ranges from -π to π radians
- **Transmittance**: Ranges from 0 to 1

## Performance Metrics

The system evaluates performance using:
- Mean Squared Error (MSE) for phase and transmittance
- Mean Absolute Error (MAE) for phase and transmittance
- R² coefficient of determination
- Focusing efficiency for metalens applications

## Applications

This system can be used for:
- Metalens design optimization
- Achromatic metalens development
- Multi-wavelength optical devices
- Beam shaping applications
- Holographic elements

## Future Improvements

Potential enhancements include:
- Multi-wavelength surrogate modeling
- Polarization-dependent response
- Manufacturing constraint integration
- Real-time optimization algorithms
- Integration with fabrication processes

## References

This work is inspired by the DFLAT methodology and related research in computational metasurfaces:

- DFLAT: Deep Focus via Learned Aperture Tuning
- NURBS-based meta-atom design
- Transformer architectures for physical systems

## License

This project is open-source and available under the MIT License.# nurbs-meta-atoms
