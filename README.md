```markdown
# Sfiralium-Core: Topological Computing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Sfiralium-Core** is an open-source framework for topological and fractal computing, based on the physical and mathematical model of the **Sfiral** (Сфираль). 

The Sfiral is a 3D topological structure consisting of two mirror-antisymmetric coils connected by an S-shaped transition loop. This architecture provides phase stability, noise immunity, and a new paradigm for Artificial Intelligence (Sfiral Artificial Neurons - SAN/FSAN).

## 🌌 Core Concepts

### 1. The Sfiral Principle (Сфираль)
Unlike classical spirals (Archimedean, Fermat) or Möbius strips, the Sfiral introduces **mirror antisymmetry** and an **S-loop** (S-петля) as a phase inversion node.
- **V⁻ and V⁺ Coils:** Two complementary branches processing data with opposite characteristics.
- **S-Loop:** The critical node of phase transition, where parameters are inverted and integrated.
- **Fractal Embedding:** Self-similar scaling across multiple levels of reality (from photons to AI architectures).

### 2. Mathematical Core (H₂ & H₃ Algebras)
The framework implements computations in hypercomplex spaces:
- **H₂ (Double Numbers):** $z = x + jy$, where $j^2 = 1$. Used for modeling S-loop phase transitions and hyperbolic geometry.
- **H₃ (Triple Numbers):** $z = x_1 + x_2 + x_3$. Introduces cubic metrics and new geometric invariants:
  - **Bingle (Бингл):** A generalized angle between two vectors in a cubic metric.
  - **Tringle (Трингл):** A three-vector phase invariant, defining the topology of the Sfiral's transition.

### 3. Sfiral Artificial Neurons (SAN / FSAN)
A new class of neural networks designed for:
- High noise immunity (due to antisymmetric cancellation).
- Energy efficiency (distributed fractal load).
- Spatiotemporal data processing (time as an internal phase parameter $s$).

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/sfiraliumlab-ux/Sfiralium-Core.git
cd Sfiralium-Core
pip install -r requirements.txt
```

### Basic Usage (PyTorch)
```python
import torch
from src.sfiral.neurons import SINLayer, FSINLayer

# Initialize Sfiral Neuron with mirror antisymmetry
sin_layer = SINLayer(in_features=10, out_features=5)

# Forward pass through the S-loop
x = torch.randn(32, 10)
output = sin_layer(x)

# Fractal Sfiral Neuron (FSIN) for multi-scale processing
fsin_layer = FSINLayer(in_features=10, out_features=5, fractal_depth=3)
output_fractal = fsin_layer(x)
```

---

## 📂 Project Structure

```text
Sfiralium-Core/
├── src/
│   └── sfiral/
│       ├── geometry.py      # H2/H3 algebras, Bingles, Tringles, S-loop math
│       ├── neurons.py       # SAN and FSAN PyTorch implementations
│       └── physics.py       # Sfiral models of Photon, Electron, DNA
├── examples/
│   └── fsin_anomaly_demo.py # Anomaly detection with FSIN
├── docs/                    # GitHub Pages interface
├── tests/
├── TERMS.md                 # License and philosophical context
└── README.md
```

---

## 🧬 Physics & Biology Applications

The Sfiral model is universal and applies to fundamental structures:
- **Photon:** Antisymmetric electric and magnetic components + S-loop (spin/polarization).
- **Electron:** Phase nodes and charge generation via antisymmetry.
- **DNA:** Complementary chains (V⁻/V⁺) connected by regulatory S-loops (promoters/enhancers).

---

## 🌐 Philosophy & Cultural Code

The Sfiral is not just a mathematical abstraction; it is the volumetric (3D) realization of the ancient Eastern symbol of the **Taiji (Great Ultimate / Инь-Ян)**. 
While Taiji represents the 2D projection of phase balance, the Sfiral provides the physical and topological mechanism for this balance in 3D spacetime. 

This framework is developed under the philosophy of the **Great Unity (Великое Единение)**, promoting open scientific collaboration and ethical AI development.

---

## ⚖️ License & Terms of Use

This project is released under the **MIT License** for scientific, educational, and non-commercial research purposes. 

**Important:** 
Commercial integration of the Sfiral Artificial Neurons (SAN/FSAN) or the underlying topological algorithms requires a separate written agreement with the author to preserve the philosophical and cultural context of the Sfiral. 

Please read [TERMS.md](TERMS.md) for full details.

---

## 🔗 Links & Resources

- **Repository:** [github.com/sfiraliumlab-ux/Sfiralium-Core](https://github.com/sfiraliumlab-ux/Sfiralium-Core)
- **Interactive Demo:** [sfiraliumlab-ux.github.io/Sfiralium-Core](https://sfiraliumlab-ux.github.io/Sfiralium-Core/)
- **Author's Publications (Zenodo):** [O.S. Basargin](https://zenodo.org/communities/sfiralium)

---

*«The Sfiral is the topology of differences and acts of awareness. It is the geometry of time itself.»*
```

---
