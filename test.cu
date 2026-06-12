#include <cuda_runtime.h>

#include <iostream>

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vectorAdd(const int* a, const int* b, int* c, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        c[index] = a[index] + b[index];
    }
}

}  // namespace

int main() {
    constexpr int count = 8;
    constexpr size_t bytes = count * sizeof(int);
    const int hostA[count] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int hostB[count] = {8, 7, 6, 5, 4, 3, 2, 1};
    int hostC[count] = {};

    cudaDeviceProp device{};
    checkCuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    std::cout << "GPU: " << device.name << '\n';

    int* deviceA = nullptr;
    int* deviceB = nullptr;
    int* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(a)");
    checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(b)");
    checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(c)");
    checkCuda(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice), "copy a");
    checkCuda(cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice), "copy b");

    vectorAdd<<<1, count>>>(deviceA, deviceB, deviceC, count);
    checkCuda(cudaGetLastError(), "vectorAdd launch");
    checkCuda(cudaDeviceSynchronize(), "vectorAdd execution");
    checkCuda(cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost), "copy result");

    bool correct = true;
    std::cout << "Result:";
    for (int value : hostC) {
        std::cout << ' ' << value;
        correct = correct && value == 9;
    }
    std::cout << '\n';

    checkCuda(cudaFree(deviceA), "cudaFree(a)");
    checkCuda(cudaFree(deviceB), "cudaFree(b)");
    checkCuda(cudaFree(deviceC), "cudaFree(c)");

    std::cout << (correct ? "CUDA test PASSED" : "CUDA test FAILED") << '\n';
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
