#include <torch/extension.h>
at::Tensor pixel_contributions(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("pixel_contributions", &pixel_contributions); }
