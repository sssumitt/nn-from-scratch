#pragma once
#include <vector>
#include "Types.hpp"

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
    virtual const Vec& activation() const = 0;
    virtual ~Layer() = default;
};
