#include "PCFG.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int MAX_GUESS_LENGTH = 128;
constexpr int THREADS_PER_BLOCK = 256;

void CheckCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error in " << operation << ": "
                  << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

segment *FindSegment(model &m, const segment &seg)
{
    if (seg.type == 1)
    {
        return &m.letters[m.FindLetter(seg)];
    }
    if (seg.type == 2)
    {
        return &m.digits[m.FindDigit(seg)];
    }
    return &m.symbols[m.FindSymbol(seg)];
}

std::string BuildBaseGuess(model &m, const PT &pt)
{
    std::string base;
    for (size_t i = 0; i + 1 < pt.content.size(); ++i)
    {
        segment *seg = FindSegment(m, pt.content[i]);
        base += seg->ordered_values[pt.curr_indices[i]];
    }
    return base;
}

__global__ void GenerateGuessesKernel(
    const char *base,
    int base_length,
    const char *values,
    const int *offsets,
    const int *lengths,
    char *results,
    int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
    {
        return;
    }

    char *output = results + static_cast<size_t>(idx) * MAX_GUESS_LENGTH;
    int position = 0;
    for (int i = 0; i < base_length; ++i)
    {
        output[position++] = base[i];
    }
    for (int i = 0; i < lengths[idx]; ++i)
    {
        output[position++] = values[offsets[idx] + i];
    }
    output[position] = '\0';
}

}  // namespace

void PriorityQueue::GenerateGPU(PT pt)
{
    CalProb(pt);

    const std::string base = BuildBaseGuess(m, pt);
    segment *last_segment = FindSegment(m, pt.content.back());
    const int count = pt.max_indices.back();

    size_t packed_size = 0;
    size_t max_suffix_length = 0;
    for (int i = 0; i < count; ++i)
    {
        packed_size += last_segment->ordered_values[i].size();
        max_suffix_length =
            std::max(max_suffix_length, last_segment->ordered_values[i].size());
    }

    if (base.size() + max_suffix_length >= MAX_GUESS_LENGTH)
    {
        Generate(pt);
        return;
    }

    std::vector<char> packed_values;
    packed_values.reserve(packed_size);
    std::vector<int> offsets(count);
    std::vector<int> lengths(count);
    for (int i = 0; i < count; ++i)
    {
        const std::string &value = last_segment->ordered_values[i];
        offsets[i] = static_cast<int>(packed_values.size());
        lengths[i] = static_cast<int>(value.size());
        packed_values.insert(packed_values.end(), value.begin(), value.end());
    }

    const size_t results_size =
        static_cast<size_t>(count) * MAX_GUESS_LENGTH;
    std::vector<char> results(results_size);

    char *device_base = nullptr;
    char *device_values = nullptr;
    int *device_offsets = nullptr;
    int *device_lengths = nullptr;
    char *device_results = nullptr;

    if (!base.empty())
    {
        CheckCuda(cudaMalloc(
                      reinterpret_cast<void **>(&device_base), base.size()),
                  "cudaMalloc(base)");
    }
    CheckCuda(cudaMalloc(
                  reinterpret_cast<void **>(&device_values),
                  packed_values.size()),
              "cudaMalloc(values)");
    CheckCuda(cudaMalloc(
                  reinterpret_cast<void **>(&device_offsets),
                  offsets.size() * sizeof(int)),
              "cudaMalloc(offsets)");
    CheckCuda(cudaMalloc(
                  reinterpret_cast<void **>(&device_lengths),
                  lengths.size() * sizeof(int)),
              "cudaMalloc(lengths)");
    CheckCuda(cudaMalloc(
                  reinterpret_cast<void **>(&device_results), results_size),
              "cudaMalloc(results)");

    if (!base.empty())
    {
        CheckCuda(cudaMemcpy(
                      device_base,
                      base.data(),
                      base.size(),
                      cudaMemcpyHostToDevice),
                  "cudaMemcpy(base)");
    }
    CheckCuda(cudaMemcpy(
                  device_values,
                  packed_values.data(),
                  packed_values.size(),
                  cudaMemcpyHostToDevice),
              "cudaMemcpy(values)");
    CheckCuda(cudaMemcpy(
                  device_offsets,
                  offsets.data(),
                  offsets.size() * sizeof(int),
                  cudaMemcpyHostToDevice),
              "cudaMemcpy(offsets)");
    CheckCuda(cudaMemcpy(
                  device_lengths,
                  lengths.data(),
                  lengths.size() * sizeof(int),
                  cudaMemcpyHostToDevice),
              "cudaMemcpy(lengths)");

    const int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GenerateGuessesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_base,
        static_cast<int>(base.size()),
        device_values,
        device_offsets,
        device_lengths,
        device_results,
        count);
    CheckCuda(cudaGetLastError(), "GenerateGuessesKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "GenerateGuessesKernel execution");
    CheckCuda(cudaMemcpy(
                  results.data(),
                  device_results,
                  results_size,
                  cudaMemcpyDeviceToHost),
              "cudaMemcpy(results)");

    CheckCuda(cudaFree(device_base), "cudaFree(base)");
    CheckCuda(cudaFree(device_values), "cudaFree(values)");
    CheckCuda(cudaFree(device_offsets), "cudaFree(offsets)");
    CheckCuda(cudaFree(device_lengths), "cudaFree(lengths)");
    CheckCuda(cudaFree(device_results), "cudaFree(results)");

    guesses.reserve(guesses.size() + count);
    for (int i = 0; i < count; ++i)
    {
        guesses.emplace_back(
            results.data() + static_cast<size_t>(i) * MAX_GUESS_LENGTH);
    }
    total_guesses += count;
}
