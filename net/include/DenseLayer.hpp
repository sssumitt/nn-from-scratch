#pragma once
#include "Layer.hpp"
#include <functional>

struct DenseLayer : Layer {
    Mat  W;
    Vec  b, Z, A;
    std::function<double(double)> f, fp;     // activation & its derivative

    DenseLayer(int in_dim, int out_dim,
               std::function<double(double)> a,
               std::function<double(double)> ap)
      : W(out_dim, Vec(in_dim)),
        b(out_dim), Z(out_dim), A(out_dim),
        f(a), fp(ap) {}

    const Vec& forward(const Vec& x) override {
        size_t m = W.size();
        for (size_t i = 0; i < m; ++i) {
            double s = b[i];
            for (size_t j = 0; j < W[i].size(); ++j)
                s += W[i][j] * x[j];
            Z[i] = s;  A[i] = f(s);
        }
        return A;
    }

    Vec backward(const Vec& prev_act, const Mat& Wn,
                 const Vec& dZn, double lr) override {
        size_t m = b.size();
        Vec dZ(m);
        for (size_t i = 0; i < m; ++i) {
            double g = 0.0;
            for (size_t k = 0; k < Wn.size(); ++k)
                g += Wn[k][i] * dZn[k];
            dZ[i] = g * fp(Z[i]);

            b[i] -= lr * dZ[i];
            for (size_t j = 0; j < prev_act.size(); ++j)
                W[i][j] -= lr * dZ[i] * prev_act[j];
        }
        return dZ;
    }

    void backward_output(const Vec& prev_act,
                         const Vec& dZout, double lr) override {
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] -= lr * dZout[i];
            for (size_t j = 0; j < prev_act.size(); ++j)
                W[i][j] -= lr * dZout[i] * prev_act[j];
        }
    }

    const Mat& weights() const override { return W; }
};
