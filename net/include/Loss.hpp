#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

using Vec = std::vector<double>;

namespace loss {
inline double binary_cross_entropy(const Vec& y_hat, const Vec& y) {
    const double eps = 1e-15;
    double sum = 0.0;
    for (size_t i = 0; i < y_hat.size(); ++i) {
        double p = std::clamp(y_hat[i], eps, 1 - eps);
        sum += -(y[i]*std::log(p) + (1 - y[i])*std::log(1 - p));
    }
    return sum;
}

inline Vec bce_grad(const Vec& y_hat, const Vec& y) {
    const double eps = 1e-15;
    Vec g(y.size());
    for (size_t i = 0; i < y.size(); ++i)
        g[i] = std::clamp(y_hat[i], eps, 1 - eps) - y[i];
    return g;
}
} // namespace loss
