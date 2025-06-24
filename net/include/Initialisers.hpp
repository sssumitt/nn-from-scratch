#pragma once
#include <random>
#include <vector>
#include <cmath>

// ──────────────────────────────────────────────────────────────
// Pass a seed once; the helper creates its own mt19937 so the
// API stays simple.  If you’d rather share an RNG, keep the old
// overload that takes std::mt19937&.
// ──────────────────────────────────────────────────────────────
inline void xavier(std::vector<std::vector<double>>& W,
                   std::vector<double>&              b,
                   int                               fan_in,
                   int                               fan_out,
                   unsigned int                      seed = 42)   // ← reproducibility
{
    std::mt19937 gen(seed);                       // same stream every run
    double lim = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<> dist(-lim, lim);

    for (auto& row : W)
        for (auto& w : row) w = dist(gen);

    std::fill(b.begin(), b.end(), 0.0);
}
