// SPDX-License-Identifier: Apache-2.0
// Non-owning typed view of an already-owned A36 HIP IPC allocation.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstdint>

torch::Tensor merged_view(int64_t pointer) {
  int device = -1;
  TORCH_CHECK(hipGetDevice(&device) == hipSuccess, "current HIP device");
  hipPointerAttribute_t attributes{};
  TORCH_CHECK(
      hipPointerGetAttributes(&attributes, reinterpret_cast<void*>(pointer)) == hipSuccess,
      "invalid owned IPC pointer");
  TORCH_CHECK(attributes.device == device, "IPC pointer belongs to another device");
  // The Workspace owner, not this view, frees the allocation after all work
  // and peer imports are drained. Only the fixed local merged region is exposed.
  return torch::from_blob(
      reinterpret_cast<void*>(pointer + 16384), {96, 6144},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("merged_view", &merged_view);
}
