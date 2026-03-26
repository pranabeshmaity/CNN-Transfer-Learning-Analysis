#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    // Constructors
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);

    // Utils
    int size() const;
    std::string to_string() const;
    static Tensor from_uint8(const std::string& raw_bytes, const std::vector<int>& shape);

    // Basic Ops
    Tensor add(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor relu() const;
    Tensor relu_backward(const Tensor& grad_output) const;

    // Optimized Backprop Ops (The Lightning Functions)
    Tensor matmul_transpose_left(const Tensor& other) const;
    Tensor sum_axis0(int batch_size, int num_classes) const;
    void sgd_update(const Tensor& grad, float lr);

    // CNN Ops
    Tensor conv2d(const Tensor& kernel, int stride, int padding) const;
    Tensor maxpool2d(int kernel_size, int stride) const;
    Tensor conv2d_grad_input(const Tensor& kernel, const Tensor& grad_output, int stride, int padding) const;
    Tensor conv2d_grad_weight(const Tensor& input, const Tensor& grad_output, int kernel_size, int stride, int padding) const;
    Tensor maxpool2d_backward(const Tensor& input, const Tensor& grad_output, int kernel_size, int stride) const;
};

#endif