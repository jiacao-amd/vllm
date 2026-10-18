// SPDX-License-Identifier: Apache-2.0
// Experimental HIP IPC ownership only; numerical kernels are emitted by FlyDSL.
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
void check(hipError_t result, const char* operation) {
  TORCH_CHECK(result == hipSuccess, operation, ": ", hipGetErrorString(result));
}
}

std::tuple<int64_t, std::vector<uint8_t>> allocate(int64_t bytes) {
  TORCH_CHECK(bytes >= 2 * 1024 * 1024 && bytes % 4096 == 0,
              "Use page-aligned coarse-grained IPC allocation");
  void* pointer = nullptr;
  check(hipExtMallocWithFlags(&pointer, bytes, hipDeviceMallocUncached), "allocate");
  check(hipMemset(pointer, 0, bytes), "initialize");
  hipIpcMemHandle_t handle;
  check(hipIpcGetMemHandle(&handle, pointer), "export");
  std::vector<uint8_t> result(sizeof(handle));
  std::memcpy(result.data(), &handle, sizeof(handle));
  return {reinterpret_cast<int64_t>(pointer), result};
}

int64_t import_peer(const std::vector<uint8_t>& bytes) {
  TORCH_CHECK(bytes.size() == sizeof(hipIpcMemHandle_t), "Invalid IPC handle size");
  hipIpcMemHandle_t handle;
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  void* pointer = nullptr;
  check(hipIpcOpenMemHandle(&pointer, handle, hipIpcMemLazyEnablePeerAccess), "import");
  return reinterpret_cast<int64_t>(pointer);
}

torch::Tensor snapshot(int64_t pointer, int64_t bytes) {
  TORCH_CHECK(bytes > 0 && bytes <= 16384, "Metadata snapshot only");
  auto result = torch::empty(
      {bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  check(hipMemcpy(result.data_ptr(), reinterpret_cast<void*>(pointer), bytes,
                  hipMemcpyDeviceToHost), "snapshot");
  return result;
}

void close_peer(int64_t pointer) {
  check(hipIpcCloseMemHandle(reinterpret_cast<void*>(pointer)), "close import");
}

void release(int64_t pointer) {
  check(hipFree(reinterpret_cast<void*>(pointer)), "release");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("allocate", &allocate);
  module.def("import_peer", &import_peer);
  module.def("snapshot", &snapshot);
  module.def("close_peer", &close_peer);
  module.def("release", &release);
}
