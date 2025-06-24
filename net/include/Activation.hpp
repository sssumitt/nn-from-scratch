#pragma once
#include <cmath>

namespace act {
inline double tanh(double x)            { return std::tanh(x); }
inline double tanh_prime(double x)      { auto t = std::tanh(x); return 1.0 - t*t; }

inline double sigmoid(double x)         { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_prime(double x)   { auto s = sigmoid(x); return s * (1.0 - s); }
} // namespace act
