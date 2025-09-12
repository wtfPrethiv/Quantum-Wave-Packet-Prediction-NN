# Quantum Wavepacket Spreading Prediction with Deep Learning

A machine learning approach to modeling and predicting the time evolution of quantum Gaussian wavepackets using PyTorch neural networks.

##  Project Overview : 

This project bridges quantum physics with machine learning by developing a supervised learning pipeline that predicts the time-dependent spreading factor σ(t) of quantum wavepackets. Instead of explicitly solving the time-dependent Schrödinger equation, we leverage deep learning to learn quantum dynamical properties from data and reconstruct the wavefunction numerically.

## Problem Statement :

In quantum mechanics, a wavepacket evolves in time such that its spatial spread increases (quantum dispersion). Traditional computation of ψ(x,t) involves solving partial differential equations, which can be computationally expensive.

**My Approach:** Reframe as a regression task

- **Input:** Time t (and physical parameters: x₀, k₀, m)
- **Output:** Spreading parameter σ(t) for reconstructing |ψ(x,t)|²

## Project Structure

```
quantum-wavepacket-spreading/
├── data/
│   └── wave_packet_spread.csv    # Generated dataset
├── models/                       # Saved model checkpoints
├── notebooks/
│   └── model.ipynb              # Jupyter notebook implementation
├── results/                     # Training outputs and visualizations
├── utils/                       # Utility functions
├── main.py                      # Main training script
├── requirements.txt                      
└── README.md

```

##  Quick Start : 

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

1. **Clone the repository:**
    
    ```bash
    git clone https://github.com/wtfPrethiv/Quantum-Wave-Packet-Prediction-NN.git
    cd Quantum-Wave-Packet-Prediction-NN
    ```
    
2. **Run the main training script:**
    
    ```bash
    python main.py
    ```
    
3. **Explore with Jupyter:**
    
    ```bash
    jupyter notebook notebooks/model.ipynb
    ```
    
### Loading Pre-trained Model

The trained model weights are saved as `model.pth` and can be loaded for inference or further training.

## Model Architecture

The model is implemented in PyTorch with the following architecture:

**QWaveModel Sequential Network:**

- **Input Layer:** 8 features → 128 neurons
- **Hidden Layer 1:** 128 → 64 neurons
- **Hidden Layer 2:** 64 → 64 neurons
- **Output Layer:** 64 → 1 neuron

**Layer Components:**

- Linear transformations (nn.Linear)
- Batch Normalization (nn.BatchNorm1d) for training stability
- ReLU activation functions (nn.ReLU)
- Single output neuron for σ(t) regression

**Training Configuration:**

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with optional weight decay
- Device: GPU acceleration when available
- Regularization: Batch normalization and optional dropout

## Results : 

Our model achieves excellent performance on quantum wavepacket prediction:

|Metric|Value|
|---|---|
|**Test MSE**|0.0018|
|**RMSE**|0.0428 (4.3% error)|
|**MAE**|0.0248|
|**R² Score**|0.9982|

## Dataset

The dataset contains synthetic data generated using analytical physics formulas for σ(t), spanning various physical parameters:

- Initial packet width (σ₀)
- Particle mass (m)
- Planck's constant (ℏ)
- Time values (t)
- Initial position and momentum (x₀, k₀)

**Data Splits:**

- Training: ~70%
- Validation: ~15%
- Test: ~15%

##  Visualizations : 

The project includes several visualization capabilities:

- **Training Curves:** Loss vs. epoch monitoring
- **Performance Metrics:** Accuracy and generalization analysis
- **3D Wavefunction Plots:** |ψ(x,t)|² reconstruction using predicted σ(t)
- **Physics Validation:** Comparison with analytical solutions

##  Dependencies : 

- **Core ML:** PyTorch
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Utilities:** tqdm

##  Scientific Significance : 

- **Novel Approach:** Demonstrates ML approximation of quantum evolution without solving PDEs
- **Computational Efficiency:** Faster than traditional numerical methods
- **Extensibility:** Foundation for quantum machine learning and Physics-Informed Neural Networks (PINNs)
- **Hybrid Methods:** Bridges data-driven and physics-based modeling

##  License : 

This project is licensed under the MIT License - see the [LICENSE](https://github.com/wtfPrethiv/Quantum-Wave-Packet-Prediction-NNblob/main/LICENSE) file for details.

---
