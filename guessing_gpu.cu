#include "PCFG.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MAX_GUESS_LENGTH = 128;
constexpr int THREADS_PER_BLOCK = 256;

double SecondsBetween(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

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
    const auto total_start = Clock::now();
    CalProb(pt);

    const auto pack_start = Clock::now();
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
    const auto pack_end = Clock::now();

    char *device_base = nullptr;
    char *device_values = nullptr;
    int *device_offsets = nullptr;
    int *device_lengths = nullptr;
    char *device_results = nullptr;

    const auto alloc_start = Clock::now();
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
    const auto alloc_end = Clock::now();

    const auto h2d_start = Clock::now();
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
    const auto h2d_end = Clock::now();

    const int blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_end;
    CheckCuda(cudaEventCreate(&kernel_start), "cudaEventCreate(kernel_start)");
    CheckCuda(cudaEventCreate(&kernel_end), "cudaEventCreate(kernel_end)");
    CheckCuda(cudaEventRecord(kernel_start), "cudaEventRecord(kernel_start)");
    GenerateGuessesKernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_base,
        static_cast<int>(base.size()),
        device_values,
        device_offsets,
        device_lengths,
        device_results,
        count);
    CheckCuda(cudaGetLastError(), "GenerateGuessesKernel launch");
    CheckCuda(cudaEventRecord(kernel_end), "cudaEventRecord(kernel_end)");
    CheckCuda(cudaEventSynchronize(kernel_end), "GenerateGuessesKernel execution");
    float kernel_milliseconds = 0.0f;
    CheckCuda(
        cudaEventElapsedTime(
            &kernel_milliseconds, kernel_start, kernel_end),
        "cudaEventElapsedTime(kernel)");
    CheckCuda(cudaEventDestroy(kernel_start), "cudaEventDestroy(kernel_start)");
    CheckCuda(cudaEventDestroy(kernel_end), "cudaEventDestroy(kernel_end)");

    const auto d2h_start = Clock::now();
    CheckCuda(cudaMemcpy(
                  results.data(),
                  device_results,
                  results_size,
                  cudaMemcpyDeviceToHost),
              "cudaMemcpy(results)");
    const auto d2h_end = Clock::now();

    const auto free_start = Clock::now();
    CheckCuda(cudaFree(device_base), "cudaFree(base)");
    CheckCuda(cudaFree(device_values), "cudaFree(values)");
    CheckCuda(cudaFree(device_offsets), "cudaFree(offsets)");
    CheckCuda(cudaFree(device_lengths), "cudaFree(lengths)");
    CheckCuda(cudaFree(device_results), "cudaFree(results)");
    const auto free_end = Clock::now();

    const auto rebuild_start = Clock::now();
    guesses.reserve(guesses.size() + count);
    for (int i = 0; i < count; ++i)
    {
        guesses.emplace_back(
            results.data() + static_cast<size_t>(i) * MAX_GUESS_LENGTH);
    }
    const auto rebuild_end = Clock::now();
    total_guesses += count;

    ++gpu_calls;
    gpu_generated_guesses += count;
    gpu_pack_time += SecondsBetween(pack_start, pack_end);
    gpu_alloc_time += SecondsBetween(alloc_start, alloc_end);
    gpu_h2d_time += SecondsBetween(h2d_start, h2d_end);
    gpu_kernel_time += kernel_milliseconds / 1000.0;
    gpu_d2h_time += SecondsBetween(d2h_start, d2h_end);
    gpu_free_time += SecondsBetween(free_start, free_end);
    gpu_rebuild_time += SecondsBetween(rebuild_start, rebuild_end);
    gpu_total_time += SecondsBetween(total_start, Clock::now());
}

void PriorityQueue::PrintGPUTimingSummary() const
{
    const double average_guesses =
        gpu_calls == 0
            ? 0.0
            : static_cast<double>(gpu_generated_guesses) / gpu_calls;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "GPU timing summary:" << std::endl;
    std::cout << "GPU calls: " << gpu_calls << std::endl;
    std::cout << "GPU generated guesses: " << gpu_generated_guesses << std::endl;
    std::cout << "Average guesses per GPU call: " << average_guesses << std::endl;
    std::cout << "GPU pack time: " << gpu_pack_time << " seconds" << std::endl;
    std::cout << "GPU allocation time: " << gpu_alloc_time << " seconds" << std::endl;
    std::cout << "GPU H2D time: " << gpu_h2d_time << " seconds" << std::endl;
    std::cout << "GPU kernel time: " << gpu_kernel_time << " seconds" << std::endl;
    std::cout << "GPU D2H time: " << gpu_d2h_time << " seconds" << std::endl;
    std::cout << "GPU rebuild time: " << gpu_rebuild_time << " seconds" << std::endl;
    std::cout << "GPU free time: " << gpu_free_time << " seconds" << std::endl;
    std::cout << "GPU total GenerateGPU time: " << gpu_total_time << " seconds"
              << std::endl;
}

void PriorityQueue::PrintDynamicSchedulingSummary() const
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Dynamic scheduling summary:" << std::endl;
    std::cout << "GPU_THRESHOLD: " << GPU_THRESHOLD << std::endl;
    std::cout << "CPU Generate calls: " << cpu_calls << std::endl;
    std::cout << "GPU Generate calls: " << gpu_calls << std::endl;
    std::cout << "CPU generated guesses: " << cpu_generated_guesses << std::endl;
    std::cout << "GPU generated guesses: " << gpu_generated_guesses << std::endl;
    std::cout << "CPU Generate time: " << cpu_generate_time << " seconds"
              << std::endl;
    std::cout << "GPU Generate time: " << gpu_total_time << " seconds"
              << std::endl;
}
