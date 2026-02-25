// gpu_d2h_overhead.cu
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
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

// 4-byte 型を明示
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
    using clock = std::chrono::high_resolution_clock;

    constexpr int   iterations = 100;
    constexpr size_t elemBytes = sizeof(Int32);

    // ---- デバイスバッファ (bufferA) ----
    Int32* d_bufferA = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bufferA, elemBytes));

    // =========================================================
    // 1. ブロッキング転送: cudaMemcpy (Device -> Host)
    // =========================================================

    Int32 h_blocking = 0;  // 常に同じ 4 バイトに書く

    // ウォームアップ (コンテキスト初期化などを計測から外す)
    write_kernel<<<1, 1>>>(d_bufferA, 0);
    CHECK_CUDA(cudaMemcpy(&h_blocking, d_bufferA,
                          elemBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto start_block = clock::now();

    for (int i = 0; i < iterations; ++i) {
        // カーネル起動
        write_kernel<<<1, 1>>>(d_bufferA, static_cast<Int32>(i));

        // ブロッキングで 4 バイトだけ D2H コピー
        CHECK_CUDA(cudaMemcpy(&h_blocking, d_bufferA,
                              elemBytes, cudaMemcpyDeviceToHost));
        // cudaMemcpy(DeviceToHost) は関数戻り時にコピー完了が保証される
        // (ホスト側から見ると同期)。
    }

    // 念のため（通常は不要だが明示）
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end_block = clock::now();
    double ms_block =
        std::chrono::duration<double, std::milli>(end_block - start_block).count();

    std::printf("Blocking D2H (cudaMemcpy): total %.3f ms, per-iter %.6f ms\n",
                ms_block, ms_block / iterations);

    // =========================================================
    // 2. ノンブロッキング転送: cudaMemcpyAsync + pinned host + stream
    // =========================================================

    // pinned (page-locked) host メモリ: 非同期転送が真に非同期になる前提
    // 要素数 = iterations 分確保して、書き込み先を毎回ずらす
    Int32* h_async = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_async, iterations * elemBytes));

    // 非同期実行用ストリーム
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // ウォームアップ (非同期パス)
    write_kernel<<<1, 1, 0, stream>>>(d_bufferA, 0);
    CHECK_CUDA(cudaMemcpyAsync(&h_async[0], d_bufferA,
                               elemBytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto start_async = clock::now();

    for (int i = 0; i < iterations; ++i) {
        // 同じ stream 上でカーネル起動
        write_kernel<<<1, 1, 0, stream>>>(d_bufferA, static_cast<Int32>(i));

        // 非同期コピー: CPU 側の書き込み先を i ごとに 4バイトずらす
        CHECK_CUDA(cudaMemcpyAsync(&h_async[i], d_bufferA,
                                   elemBytes, cudaMemcpyDeviceToHost, stream));
        // ここでは待たない。カーネルと D2H コピーは stream 内で逐次実行される。
    }

    // ループで投げたカーネル＋コピーが全部終わるまで待つ
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_async = clock::now();
    double ms_async =
        std::chrono::duration<double, std::milli>(end_async - start_async).count();

    std::printf("Non-blocking D2H (cudaMemcpyAsync + pinned + stream): "
                "total %.3f ms, per-iter %.6f ms\n",
                ms_async, ms_async / iterations);

    // 結果が最適化で消されないように簡単なチェックサム
    long long checksum = 0;
    for (int i = 0; i < iterations; ++i) {
        checksum += static_cast<long long>(h_async[i]);
    }
    std::printf("Checksum (ignore): %lld\n", checksum);

    // 後始末
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_async));
    CHECK_CUDA(cudaFree(d_bufferA));

    return 0;
}
