#pragma once
#include <cmath>

namespace act {
inline double tanh(double x)                            { return std::tanh(x); }
inline double tanh_prime(double x)                      { auto t = std::tanh(x); return 1.0 - t*t; }

inline double sigmoid(double x)                         { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_prime(double x)                   { auto s = sigmoid(x); return s * (1.0 - s); }

inline double relu(double x)                            { return x > 0 ? x : 0; }
inline double relu_prime(double x)                      { return x > 0 ? 1.0 : 0.0; }

inline double leaky_relu(double x, double alpha = 0.01) { return x > 0 ? x : alpha * x; }
inline double leaky_relu_prime(double x, double alpha = 0.01) { return x > 0 ? 1.0 : alpha; }

inline double softmax(double x, double sum_exp) {
    return std::exp(x) / sum_exp;
}
inline double softmax_prime(double x, double sum_exp) {
    double exp_x = std::exp(x);
    return (exp_x * (sum_exp - exp_x)) / (sum_exp * sum_exp);
}
inline double softplus(double x) {
    return std::log(1.0 + std::exp(x));

}
inline double softplus_prime(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
inline double elu(double x, double alpha = 1.0) {
    return x >= 0 ? x : alpha * (std::exp(x) - 1);
}
inline double elu_prime(double x, double alpha = 1.0) {
    return x >= 0 ? 1.0 : alpha * std::exp(x);
}
inline double swish(double x) {
    return x / (1.0 + std::exp(-x));
}
inline double swish_prime(double x) {
    double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
    return sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x);
}
inline double mish(double x) {
    double tanh_x = std::tanh(x);
    return x * tanh_x;
}
inline double mish_prime(double x) {
    double tanh_x = std::tanh(x);
    double exp_neg_x = std::exp(-x);
    double sigmoid_x = 1.0 / (1.0 + exp_neg_x);
    return tanh_x + x * (1.0 - tanh_x * tanh_x) * sigmoid_x;
}
inline double gelu(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}
inline double gelu_prime(double x) {
    double tanh_part = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x));
    double exp_part = std::exp(-0.5 * (x + 0.044715 * x * x * x) * (x + 0.044715 * x * x * x));
    return 0.5 * (1.0 + tanh_part + 0.044715 * 3.0 * x * x * exp_part);
}
inline double softsign(double x) {
    return x / (1.0 + std::abs(x));
}
inline double softsign_prime(double x) {
    double denom = 1.0 + std::abs(x);
    return 1.0 / (denom * denom);
}

} // namespace act
