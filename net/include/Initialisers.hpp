#pragma once
#include <random>
#include <vector>
#include <cmath>
#include "Types.hpp"


inline void xavier(Mat& W,
                   Vec& b,
                   int fan_in,
                   int fan_out,
                   unsigned int seed = 42)   // ← reproducibility
{
    std::mt19937 gen(seed);                       // same stream every run
    double lim = std::sqrt(6.0 / (fan_in + fan_out));
    std::uniform_real_distribution<> dist(-lim, lim);

    for (auto& row : W)
        for (auto& w : row) w = dist(gen);

    std::fill(b.begin(), b.end(), 0.0);
}

inline void he_normal(Mat& W,
                      Vec& b,
                      int fan_in,
                      int fan_out,
                      unsigned seed = 42)
{
    std::mt19937 gen(seed);
    double stddev = std::sqrt(2.0 / fan_in);
    std::normal_distribution<> dist(0.0, stddev);

    for (auto& row : W)
        for (auto& w : row)
            w = dist(gen);

    std::fill(b.begin(), b.end(), 0.0);
}

