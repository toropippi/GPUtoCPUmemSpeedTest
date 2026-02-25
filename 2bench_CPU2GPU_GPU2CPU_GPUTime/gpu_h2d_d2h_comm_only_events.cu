// gpu_h2d_d2h_comm_only_events.cu
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err__), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

using Int32 = std::int32_t;
static_assert(sizeof(Int32) == 4, "Int32 must be 4 bytes");

// Lightweight kernel: read one int from input and write one int to output.
__global__ void passthrough_kernel(const Int32* input, Int32* output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = input[0];
    }
}

int main()
{
    constexpr int    iterations = 1000;
    constexpr size_t elemBytes  = sizeof(Int32);

    Int32* d_input  = nullptr;
    Int32* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, elemBytes));
    CHECK_CUDA(cudaMalloc(&d_output, elemBytes));

    Int32* h_send = nullptr;
    Int32* h_recv = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_send, iterations * elemBytes));
    CHECK_CUDA(cudaMallocHost(&h_recv, iterations * elemBytes));

    for (int i = 0; i < iterations; ++i) {
        h_send[i] = static_cast<Int32>(i);
        h_recv[i] = 0;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warm-up for first-launch effects.
    CHECK_CUDA(cudaMemcpyAsync(&d_input[0], &h_send[0],
                               elemBytes, cudaMemcpyHostToDevice, stream));
    passthrough_kernel<<<1, 1, 0, stream>>>(d_input, d_output);
    CHECK_CUDA(cudaMemcpyAsync(&h_recv[0], &d_output[0],
                               elemBytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> h2dStart(iterations), h2dStop(iterations);
    std::vector<cudaEvent_t> kernelDone(iterations);
    std::vector<cudaEvent_t> d2hStart(iterations), d2hStop(iterations);

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventCreate(&h2dStart[i]));
        CHECK_CUDA(cudaEventCreate(&h2dStop[i]));
        CHECK_CUDA(cudaEventCreate(&kernelDone[i]));
        CHECK_CUDA(cudaEventCreate(&d2hStart[i]));
        CHECK_CUDA(cudaEventCreate(&d2hStop[i]));
    }

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(h2dStart[i], stream));
        CHECK_CUDA(cudaMemcpyAsync(&d_input[0], &h_send[i],
                                   elemBytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaEventRecord(h2dStop[i], stream));

        passthrough_kernel<<<1, 1, 0, stream>>>(d_input, d_output);
        CHECK_CUDA(cudaEventRecord(kernelDone[i], stream));
        // Explicitly wait for kernel completion before D2H timing.
        CHECK_CUDA(cudaEventSynchronize(kernelDone[i]));

        CHECK_CUDA(cudaEventRecord(d2hStart[i], stream));
        CHECK_CUDA(cudaMemcpyAsync(&h_recv[i], &d_output[0],
                                   elemBytes, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaEventRecord(d2hStop[i], stream));
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    double h2dTotalMs = 0.0;
    double d2hTotalMs = 0.0;
    for (int i = 0; i < iterations; ++i) {
        float h2dMs = 0.0f;
        float d2hMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&h2dMs, h2dStart[i], h2dStop[i]));
        CHECK_CUDA(cudaEventElapsedTime(&d2hMs, d2hStart[i], d2hStop[i]));
        h2dTotalMs += static_cast<double>(h2dMs);
        d2hTotalMs += static_cast<double>(d2hMs);
    }

    long long checksum = 0;
    long long mismatch = 0;
    for (int i = 0; i < iterations; ++i) {
        checksum += static_cast<long long>(h_recv[i]);
        if (h_recv[i] != static_cast<Int32>(i)) {
            ++mismatch;
        }
    }

    const double h2dPerIter = h2dTotalMs / iterations;
    const double d2hPerIter = d2hTotalMs / iterations;
    const double commTotal  = h2dTotalMs + d2hTotalMs;
    const double commPerIter = commTotal / iterations;

    std::printf("GPU comm time (H2D only, %d iters): total %.3f ms, per-iter %.6f ms\n",
                iterations, h2dTotalMs, h2dPerIter);
    std::printf("GPU comm time (D2H only, %d iters): total %.3f ms, per-iter %.6f ms\n",
                iterations, d2hTotalMs, d2hPerIter);
    std::printf("GPU comm time (H2D + D2H, %d iters): total %.3f ms, per-iter %.6f ms\n",
                iterations, commTotal, commPerIter);
    std::printf("D2H / H2D ratio: %.3fx\n", d2hPerIter / h2dPerIter);
    std::printf("Checksum (ignore): %lld\n", checksum);
    std::printf("Mismatch count: %lld\n", mismatch);

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventDestroy(h2dStart[i]));
        CHECK_CUDA(cudaEventDestroy(h2dStop[i]));
        CHECK_CUDA(cudaEventDestroy(kernelDone[i]));
        CHECK_CUDA(cudaEventDestroy(d2hStart[i]));
        CHECK_CUDA(cudaEventDestroy(d2hStop[i]));
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_send));
    CHECK_CUDA(cudaFreeHost(h_recv));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
