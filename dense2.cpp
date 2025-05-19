#include <vector>
#include <iostream>
#include <functional>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <fstream>

// Type aliases
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// Activation functions and their derivatives
inline double tanh_act(double x) {
    return std::tanh(x);
}
inline double tanh_prime(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
inline double sigmoid_prime(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Dense Layer structure
typedef std::function<double(double)> ActFn;
struct denseLayer {
    Matrix W;   // [out_dim x in_dim]
    Vector b;   // [out_dim]
    Vector Z;   // [out_dim]
    Vector A;   // [out_dim]
    ActFn act, act_prime;

    denseLayer(int in_dim, int out_dim, ActFn f, ActFn f_prime)
        : W(out_dim, Vector(in_dim)), b(out_dim), Z(out_dim), A(out_dim), act(f), act_prime(f_prime) {}

    // Forward
    const Vector& forward(const Vector &input) {
        size_t out_dim = W.size();
        for (size_t i = 0; i < out_dim; ++i) {
            double sum = b[i];
            for (size_t j = 0; j < W[i].size(); ++j)
                sum += W[i][j] * input[j];
            Z[i] = sum;
            A[i] = act(sum);
        }
        return A;
    }

    // Backprop for hidden layers
    Vector backward(const Vector &prev_act, const Matrix &W_next, const Vector &dZ_next, double lr) {
        size_t out_dim = b.size();
        Vector dZ(out_dim);
        // compute dZ and update params
        for (size_t i = 0; i < out_dim; ++i) {
            double grad = 0.0;
            for (size_t k = 0; k < W_next.size(); ++k)
                grad += W_next[k][i] * dZ_next[k];
            dZ[i] = grad * act_prime(Z[i]);
            b[i] -= lr * dZ[i];
            for (size_t j = 0; j < prev_act.size(); ++j)
                W[i][j] -= lr * dZ[i] * prev_act[j];
        }
        return dZ;
    }

    // Backprop for output layer
    void backward_output(const Vector &prev_act, const Vector &dZ_output, double lr) {
        for (size_t i = 0; i < b.size(); ++i) {
            for (size_t j = 0; j < prev_act.size(); ++j)
                W[i][j] -= lr * dZ_output[i] * prev_act[j];
            b[i] -= lr * dZ_output[i];
        }
    }
};

// Forward through network
Vector forward_pass(const Vector &input, std::vector<denseLayer> &model) {
    const Vector* current = &input;
    for (auto &layer : model)
        current = &layer.forward(*current);
    return *current;
}

// Xavier initialization
void xavier_init(denseLayer &layer, int in_dim, int out_dim, std::mt19937 &gen) {
    double limit = std::sqrt(6.0 / (in_dim + out_dim));
    std::uniform_real_distribution<> dis(-limit, limit);
    for (auto &row : layer.W)
        for (auto &w : row)
            w = dis(gen);
    std::fill(layer.b.begin(), layer.b.end(), 0.0);
}

// Loss and derivative
inline double binary_cross_entropy(const Vector &out, const Vector &t) {
    double eps = 1e-15, sum = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        double y = std::clamp(out[i], eps, 1 - eps);
        sum += -(t[i] * std::log(y) + (1 - t[i]) * std::log(1 - y));
    }
    return sum;
}
inline Vector binary_cross_entropy_derivative(const Vector &out, const Vector &t) {
    Vector d(out.size());
    for (size_t i = 0; i < out.size(); ++i)
        d[i] = std::clamp(out[i], 1e-15, 1 - 1e-15) - t[i];
    return d;
}





using Point = std::vector<double>;

std::vector<Point> generate_grid_00_11(int num = 200) {
    std::vector<Point> grid;
    grid.reserve(num * num);

    double step = 1.0 / (num - 1);
    for (int i = 0; i < num; ++i) {
        double x = i * step;
        for (int j = 0; j < num; ++j) {
            double y = j * step;
            grid.push_back({ x, y });
        }
    }

    return grid;
}



int main() {
    std::vector<Vector> inputs = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<Vector> targets = {{0},{1},{1},{0}};

    std::vector<denseLayer> layers;
    layers.emplace_back(2, 3, tanh_act, tanh_prime);
    layers.emplace_back(3, 1, sigmoid, sigmoid_prime);

    std::random_device rd;
    std::mt19937 gen(rd());
    xavier_init(layers[0], 2, 3, gen);
    xavier_init(layers[1], 3, 1, gen);

    double lr = 0.01;
    int epochs = 1000000;

    for (int e = 1; e <= epochs; ++e) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Vector out = forward_pass(inputs[i], layers);
            total_loss += binary_cross_entropy(out, targets[i]);

            Vector dE = binary_cross_entropy_derivative(out, targets[i]);
            // output layer
            const Vector &a_prev = layers.size() > 1 ? layers[layers.size()-2].A : inputs[i];
            layers.back().backward_output(a_prev, dE, lr);
            // hidden
            const Vector *prev_act;
            Vector dZ = dE;
            for (int l = layers.size()-2; l >= 0; --l) {
                prev_act = (l == 0) ? &inputs[i] : &layers[l-1].A;
                dZ = layers[l].backward(*prev_act, layers[l+1].W, dZ, lr);
            }
        }
        if (e <= 10 || e % 1000 == 0)
            std::cout << "Epoch " << e << " Loss: " << total_loss / inputs.size() << '\n';
    }

    std::cout << "\nTesting XOR:\n";

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector out = forward_pass(inputs[i], layers);
        std::cout << "(" << inputs[i][0] << ',' << inputs[i][1] << ") -> "
                  << out[0] << " (expected " << targets[i][0] << ")\n";
    }



    // use the grid points to generate a csv file  that has point and predicted output from the model 
    auto grid = generate_grid_00_11(200);
    std::ofstream out("xor_grid.csv");
    out << "x,y,prob\n";
    for (auto &pt : grid) {
        Vector pred = forward_pass(pt, layers);
        out << pt[0] << ',' << pt[1] << ',' << pred[0] << '\n';
    }
    out.close();
    std::cout << "Grid data saved to xor_grid.csv\n";


    return 0;
}
