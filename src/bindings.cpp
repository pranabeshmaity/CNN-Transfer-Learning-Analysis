#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(my_framework_cpp, m) {
    m.doc() = "C++ backend for custom deep learning framework";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&>())
        .def(py::init<const std::vector<float>&, const std::vector<int>&>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("to_string", &Tensor::to_string)
        .def_static("from_uint8", &Tensor::from_uint8)
        .def("add", &Tensor::add)
        .def("matmul", &Tensor::matmul)
        .def("transpose", &Tensor::transpose)
        .def("relu", &Tensor::relu)
        .def("relu_backward", &Tensor::relu_backward)
        .def("conv2d", &Tensor::conv2d)
        .def("maxpool2d", &Tensor::maxpool2d)
        .def("conv2d_grad_input", &Tensor::conv2d_grad_input)
        .def("conv2d_grad_weight", &Tensor::conv2d_grad_weight)
        .def("maxpool2d_backward", &Tensor::maxpool2d_backward)
        // Optimized Lightning Methods
        .def("matmul_transpose_left", &Tensor::matmul_transpose_left)
        .def("sum_axis0", &Tensor::sum_axis0)
        .def("sgd_update", &Tensor::sgd_update);
}