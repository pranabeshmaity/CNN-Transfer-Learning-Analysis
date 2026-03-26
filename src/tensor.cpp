#include "tensor.h"
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>

// CONSTRUCTORS
Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    long long total = 1;
    for (int s : shape) total *= s;
    data.assign(total, 0.0f);
}

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape) 
    : data(data), shape(shape) {}

// UTILS
int Tensor::size() const {
    int total = 1;
    for (int s : shape) total *= s;
    return total;
}

std::string Tensor::to_string() const {
    std::stringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i)
        ss << shape[i] << (i < shape.size() - 1 ? ", " : "");
    ss << "])";
    return ss.str();
}

Tensor Tensor::from_uint8(const std::string& raw_bytes, const std::vector<int>& shape) {
    Tensor t(shape);
    for (size_t i = 0; i < raw_bytes.size(); ++i) {
        t.data[i] = static_cast<unsigned char>(raw_bytes[i]) / 255.0f;
    }
    return t;
}

// BASIC MATH
Tensor Tensor::add(const Tensor& other) const {
    std::vector<float> res(size());
    int s2 = other.size();
    for (int i = 0; i < size(); ++i) res[i] = data[i] + other.data[i % s2];
    return Tensor(res, shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    int M = shape[0], K = shape[1], N = other.shape[1];
    std::vector<float> res(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = data[i * K + k];
            if (a_val == 0) continue;
            for (int j = 0; j < N; ++j) {
                res[i * N + j] += a_val * other.data[k * N + j];
            }
        }
    }
    return Tensor(res, {M, N});
}

Tensor Tensor::transpose() const {
    int R = shape[0], C = shape[1];
    std::vector<float> res(data.size());
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) res[c * R + r] = data[r * C + c];
    }
    return Tensor(res, {C, R});
}

// LIGHTNING UPDATES
Tensor Tensor::matmul_transpose_left(const Tensor& other) const {
    // Computes (this^T) @ other
    int K = shape[0]; 
    int M = shape[1]; 
    int N = other.shape[1];
    Tensor res({M, N});
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M; ++i) {
            float a_val = data[k * M + i];
            if (a_val == 0) continue;
            for (int j = 0; j < N; ++j) {
                res.data[i * N + j] += a_val * other.data[k * N + j];
            }
        }
    }
    return res;
}

Tensor Tensor::sum_axis0(int batch_size, int num_classes) const {
    Tensor res({1, num_classes});
    for (int i = 0; i < batch_size; ++i) {
        for (int c = 0; c < num_classes; ++c) {
            res.data[c] += data[i * num_classes + c];
        }
    }
    return res;
}

void Tensor::sgd_update(const Tensor& grad, float lr) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= lr * grad.data[i];
    }
}

// ACTIVATIONS
Tensor Tensor::relu() const {
    std::vector<float> res = data;
    for (float& v : res) if (v < 0) v = 0;
    return Tensor(res, shape);
}

Tensor Tensor::relu_backward(const Tensor& grad_output) const {
    std::vector<float> res(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        res[i] = (data[i] > 0) ? grad_output.data[i] : 0;
    return Tensor(res, shape);
}

// CNN OPERATIONS
Tensor Tensor::conv2d(const Tensor& kernel, int stride, int padding) const {
    int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    int KH = kernel.shape[0], KW = kernel.shape[1], F = kernel.shape[3];
    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;
    Tensor res({N, H_out, W_out, F});
    
    for (int n = 0; n < N; ++n) {
        for (int h_o = 0; h_o < H_out; ++h_o) {
            int h_s = h_o * stride - padding;
            for (int w_o = 0; w_o < W_out; ++w_o) {
                int w_s = w_o * stride - padding;
                for (int f = 0; f < F; ++f) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        int h_idx = h_s + kh;
                        if (h_idx < 0 || h_idx >= H) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            int w_idx = w_s + kw; // FIXED: Use w_idx consistent with h_idx
                            if (w_idx < 0 || w_idx >= W) continue;
                            
                            for (int c = 0; c < C; ++c) {
                                sum += data[((n * H + h_idx) * W + w_idx) * C + c] * kernel.data[((kh * KW + kw) * C + c) * F + f];
                            }
                        }
                    }
                    res.data[((n * H_out + h_o) * W_out + w_o) * F + f] = sum;
                }
            }
        }
    }
    return res;
}

Tensor Tensor::maxpool2d(int kernel_size, int stride) const {
    int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    int H_out = (H - kernel_size) / stride + 1;
    int W_out = (W - kernel_size) / stride + 1;
    Tensor res({N, H_out, W_out, C});
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float max_v = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            float val = data[((n * H + (h * stride + kh)) * W + (w * stride + kw)) * C + c];
                            if (val > max_v) max_v = val;
                        }
                    }
                    res.data[((n * H_out + h) * W_out + w) * C + c] = max_v;
                }
            }
        }
    }
    return res;
}

// GRADIENT OPERATIONS
Tensor Tensor::conv2d_grad_input(const Tensor& kernel, const Tensor& grad_output, int stride, int padding) const {
    int N = shape[0], H = shape[1], W = shape[2], C = shape[3];
    int KH = kernel.shape[0], KW = kernel.shape[1], F = kernel.shape[3];
    int H_out = grad_output.shape[1], W_out = grad_output.shape[2];
    std::vector<float> g_in(size(), 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int h_o = 0; h_o < H_out; ++h_o) {
            int h_s = h_o * stride - padding;
            for (int w_o = 0; w_o < W_out; ++w_o) {
                int w_s = w_o * stride - padding;
                for (int f = 0; f < F; ++f) {
                    float go = grad_output.data[((n * H_out + h_o) * W_out + w_o) * F + f];
                    if (go == 0) continue;
                    for (int kh = 0; kh < KH; ++kh) {
                        int h_i = h_s + kh;
                        if (h_i >= 0 && h_i < H) {
                            for (int kw = 0; kw < KW; ++kw) {
                                int w_i = w_s + kw;
                                if (w_i >= 0 && w_i < W) {
                                    for (int c = 0; c < C; ++c)
                                        g_in[((n * H + h_i) * W + w_i) * C + c] += go * kernel.data[((kh * KW + kw) * C + c) * F + f];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return Tensor(g_in, shape);
}

Tensor Tensor::conv2d_grad_weight(const Tensor& input, const Tensor& grad_output, int kernel_size, int stride, int padding) const {
    int N = input.shape[0], H = input.shape[1], W = input.shape[2], C = input.shape[3];
    int F = grad_output.shape[3], H_out = grad_output.shape[1], W_out = grad_output.shape[2], K = kernel_size;
    std::vector<float> g_w(K * K * C * F, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int h_o = 0; h_o < H_out; ++h_o) {
            int h_s = h_o * stride - padding;
            for (int w_o = 0; w_o < W_out; ++w_o) {
                int w_s = w_o * stride - padding;
                for (int f = 0; f < F; ++f) {
                    float go = grad_output.data[((n * H_out + h_o) * W_out + w_o) * F + f];
                    if (go == 0) continue;
                    for (int kh = 0; kh < K; ++kh) {
                        int h_i = h_s + kh;
                        if (h_i < 0 || h_i >= H) continue;
                        for (int kw = 0; kw < K; ++kw) {
                            int w_i = w_s + kw;
                            if (w_i < 0 || w_i >= W) continue;
                            for (int c = 0; c < C; ++c)
                                g_w[((kh * K + kw) * C + c) * F + f] += go * input.data[((n * H + h_i) * W + w_i) * C + c];
                        }
                    }
                }
            }
        }
    }
    return Tensor(g_w, {K, K, C, F});
}

Tensor Tensor::maxpool2d_backward(const Tensor& input, const Tensor& grad_output, int kernel_size, int stride) const {
    int N = input.shape[0], H = input.shape[1], W = input.shape[2], C = input.shape[3];
    int H_out = grad_output.shape[1], W_out = grad_output.shape[2];
    std::vector<float> g_in(input.size(), 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float max_v = -std::numeric_limits<float>::infinity();
                    int max_h = -1, max_w = -1;
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_i = h * stride + kh, w_i = w * stride + kw;
                            float val = input.data[((n * H + h_i) * W + w_i) * C + c];
                            if (val > max_v) { max_v = val; max_h = h_i; max_w = w_i; }
                        }
                    }
                    if (max_h != -1)
                        g_in[((n * H + max_h) * W + max_w) * C + c] += grad_output.data[((n * H_out + h) * W_out + w) * C + c];
                }
            }
        }
    }
    return Tensor(g_in, input.shape);
}
