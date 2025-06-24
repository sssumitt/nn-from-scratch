#pragma once
#include <vector>

using Vec   = std::vector<double>;
using Mat   = std::vector<Vec>;

struct Layer {
    virtual const Vec& forward (const Vec& x)                           = 0;
    virtual Vec        backward(const Vec& prev_act,
                                const Mat& W_next,
                                const Vec& dZ_next,
                                double lr)                               = 0;
    virtual void       backward_output(const Vec& prev_act,
                                       const Vec& dZ,
                                       double lr)                        = 0;
    virtual const Mat& weights()  const = 0;
    virtual ~Layer() = default;
};
