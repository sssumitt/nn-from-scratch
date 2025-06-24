#pragma once
#include "Layer.hpp"
#include <memory>
#include <random>

class Network {
    std::vector<std::unique_ptr<Layer>> layers_;
public:
    template<typename LayerT, typename... Args>
    LayerT& add(Args&&... args) {
        layers_.emplace_back(std::make_unique<LayerT>(std::forward<Args>(args)...));
        return *static_cast<LayerT*>(layers_.back().get());
    }

    const Vec& forward(const Vec& x) {
        const Vec* cur = &x;
        for (auto& l : layers_) cur = &l->forward(*cur);
        return *cur;
    }

    // one-step SGD
    void backward(const Vec& input, const Vec& dE, double lr) {
        Vec dZ = dE;
        for (int i = int(layers_.size()) - 1; i >= 0; --i) {
            const Vec& prev = (i == 0) ? input : layers_[i-1]->forward(input); // cached A already
            if (i == int(layers_.size()) - 1)
                layers_[i]->backward_output(prev, dZ, lr);
            else
                dZ = layers_[i]->backward(prev, layers_[i+1]->weights(), dZ, lr);
        }
    }

    std::vector<std::unique_ptr<Layer>>& layers() { return layers_; }
};
