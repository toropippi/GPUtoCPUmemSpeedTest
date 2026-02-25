// gpu_d2h_overhead_events.cu
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err__), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// 4-byte 型
using Int32 = std::int32_t;
static_assert(sizeof(Int32) == 4, "Int32 must be 4 bytes");

// 単純に bufferA[0] に値を書くだけのカーネル
__global__ void write_kernel(Int32* bufferA, Int32 value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        bufferA[0] = value;
    }
}

int main()
{
    constexpr int    iterations = 100;
    constexpr size_t elemBytes  = sizeof(Int32);

    // ---- デバイスバッファ (bufferA) ----
    Int32* d_bufferA = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bufferA, elemBytes));

    // =========================================================
    // 1. ブロッキング: cudaMemcpy (Device -> Host)
    //    GPU時間をイベントで計測
    // =========================================================
    Int32 h_blocking = 0;

    // ウォームアップ（コンテキスト初期化などを除外）
    write_kernel<<<1, 1>>>(d_bufferA, 0);
    CHECK_CUDA(cudaMemcpy(&h_blocking, d_bufferA,
                          elemBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    // イベント作成
    cudaEvent_t start_block_evt, stop_block_evt;
    CHECK_CUDA(cudaEventCreate(&start_block_evt));
    CHECK_CUDA(cudaEventCreate(&stop_block_evt));

    // GPUタイミング計測開始
    CHECK_CUDA(cudaEventRecord(start_block_evt, 0)); // default stream

    for (int i = 0; i < iterations; ++i) {
        // カーネル起動
        write_kernel<<<1, 1>>>(d_bufferA, static_cast<Int32>(i));

        // ブロッキング D2H 転送
        CHECK_CUDA(cudaMemcpy(&h_blocking, d_bufferA,
                              elemBytes, cudaMemcpyDeviceToHost));
        // cudaMemcpy(DeviceToHost) は戻り時に完了が保証される (ホスト視点で同期)。
        // default stream 上では、その前の kernel も完了していることになる。
    }

    // 計測終了イベントを記録
    CHECK_CUDA(cudaEventRecord(stop_block_evt, 0));
    // 計測終了まで待つ
    CHECK_CUDA(cudaEventSynchronize(stop_block_evt));

    float ms_block_gpu = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_block_gpu,
                                    start_block_evt, stop_block_evt));

    std::printf(
        "GPU time (Blocking: kernel + cudaMemcpy, %d iters): "
        "total %.3f ms, per-iter %.6f ms\n",
        iterations, ms_block_gpu, ms_block_gpu / iterations);

    // イベント破棄（blocking 用）
    CHECK_CUDA(cudaEventDestroy(start_block_evt));
    CHECK_CUDA(cudaEventDestroy(stop_block_evt));

    // =========================================================
    // 2. ノンブロッキング:
    //    cudaMemcpyAsync + pinned host + stream
    //    GPU時間をイベントで計測
    // =========================================================

    // pinned host メモリ（非同期転送の前提）
    Int32* h_async = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_async, iterations * elemBytes));

    // 非同期実行用 stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // ウォームアップ
    write_kernel<<<1, 1, 0, stream>>>(d_bufferA, 0);
    CHECK_CUDA(cudaMemcpyAsync(&h_async[0], d_bufferA,
                               elemBytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // イベント作成
    cudaEvent_t start_async_evt, stop_async_evt;
    CHECK_CUDA(cudaEventCreate(&start_async_evt));
    CHECK_CUDA(cudaEventCreate(&stop_async_evt));

    // 計測開始 (stream 上にイベントを置く)
    CHECK_CUDA(cudaEventRecord(start_async_evt, stream));

    for (int i = 0; i < iterations; ++i) {
        // 同一 stream 上で kernel -> memcpyAsync を順に積む
        write_kernel<<<1, 1, 0, stream>>>(d_bufferA, static_cast<Int32>(i));

        CHECK_CUDA(cudaMemcpyAsync(&h_async[i], d_bufferA,
                                   elemBytes, cudaMemcpyDeviceToHost, stream));
        // stream 内では in-order なので、
        // kernel(i) -> memcpyAsync(i) と直列に走る。
    }

    // ループで積んだ最後のコピーのあとにイベントを挿入
    CHECK_CUDA(cudaEventRecord(stop_async_evt, stream));
    // stream の全操作完了を待つ
    CHECK_CUDA(cudaEventSynchronize(stop_async_evt));

    float ms_async_gpu = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_async_gpu,
                                    start_async_evt, stop_async_evt));

    std::printf(
        "GPU time (Non-blocking: kernel + cudaMemcpyAsync, %d iters): "
        "total %.3f ms, per-iter %.6f ms\n",
        iterations, ms_async_gpu, ms_async_gpu / iterations);

    // 結果が消されないよう、簡単なチェックサム
    long long checksum = 0;
    for (int i = 0; i < iterations; ++i) {
        checksum += static_cast<long long>(h_async[i]);
    }
    std::printf("Checksum (ignore): %lld\n", checksum);

    // 後始末
    CHECK_CUDA(cudaEventDestroy(start_async_evt));
    CHECK_CUDA(cudaEventDestroy(stop_async_evt));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_async));
    CHECK_CUDA(cudaFree(d_bufferA));

    return 0;
}
