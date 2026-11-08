#include <torch/extension.h>
void hgemv_f16(at::Tensor a, at::Tensor b, at::Tensor c, int64_t ksplit);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hgemv_f16", &hgemv_f16, "hgemv_f16", py::arg("a"), py::arg("b"), py::arg("c"), py::arg("ksplit") = 0);
}
