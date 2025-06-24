# XOR Neural Network Library

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A *header-only* C++ library for building and training simple feedforward neural networks.
This example project demonstrates a 2-layer network solving the XOR problem and exporting results for visualization.

---

## 📂 Repository Structure

```plaintext
net/
├── include/            # Header-only library files
│   ├── Activation.hpp  # Activation functions (tanh, sigmoid)
│   ├── Loss.hpp        # Loss functions & gradients (binary cross-entropy)
│   ├── Initialisers.hpp# Weight initialisers (Xavier, reproducible)
│   ├── Layer.hpp       # Abstract Layer interface
│   ├── DenseLayer.hpp  # Fully-connected layer implementation
│   └── Network.hpp     # Model orchestration (forward/backward)
└── app/                # Example application
    └── main.cpp        # Train XOR network & generate CSV

Makefile               # Build rules
README.md              # This document
```

---

## 🔧 Prerequisites

* **Compiler**: Any C++17-compatible (e.g., `g++`, `clang++`)
* **Build tool**: GNU Make (optional)

---

## ⚙️ Building & Running

### Using Make (recommended)

```bash
# Build executable
make

# Train & run
make run
```

### Manual build

```bash
g++ -std=c++17 -O2 -Iinclude app/main.cpp -o xor_net
./xor_net
```

On success, you’ll see training logs:

```
Epoch 1  loss 0.6931
Epoch 1000  loss 0.6930
…
Testing XOR:
(0,0) -> 0.01
(0,1) -> 0.98
…
```

and a file `xor_grid.csv` with `x,y,prob` for contour plots.

---

## 🚀 Usage in Your Project

Add `include/` to your compiler’s include path:

```cpp
#include "Activation.hpp"
#include "Loss.hpp"
#include "Initialisers.hpp"
#include "DenseLayer.hpp"
#include "Network.hpp"
```

Then construct and train:

```cpp
Network net;
// Two-layer network: 2→3→1
auto &l1 = net.add<DenseLayer>(2, 3, act::tanh, act::tanh_prime);
auto &l2 = net.add<DenseLayer>(3, 1, act::sigmoid, act::sigmoid_prime);

// Reproducible init\ nxavier(l1.W, l1.b, 2, 3, 1234);
xavier(l2.W, l2.b, 3, 1, 1234);

// Training loop
for (…)
  net.forward(input);
  net.backward(input, loss::bce_grad(output, target), lr);
```

---

## 🛠️ Extending with New Layers

1. Create a header (e.g. `Conv2DLayer.hpp`) in `include/`.
2. Derive from `Layer` and implement:

   * `forward`
   * `backward`
   * `backward_output`
   * `weights()` accessor
3. Register your layer:

```cpp
net.add<Conv2DLayer>(in_ch, out_ch, kernel_size, stride, padding);
```

No changes to core library needed—`Network` handles any `Layer`.

---

## 📊 Visualization

Use the generated `xor_grid.csv` with Python/Matplotlib or your favorite tool:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('xor_grid.csv')
plt.tricontourf(df.x, df.y, df.prob, levels=50)
plt.show()
```

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
