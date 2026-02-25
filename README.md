# GPU -> CPU 転送オーバーヘッドベンチマーク

このリポジトリは、`cudaMemcpy` / `cudaMemcpyAsync` を使った GPU -> CPU 転送の「小サイズ転送時のオーバーヘッド」を確認するためのベンチマークです。

## 目的

- 大きなデータ転送の帯域測定ではなく、`4 byte` の最小転送を繰り返したときの API + 同期待ちのコストを比較する
- 次の 2 パターンを比較する
  - Blocking: `cudaMemcpy` (DeviceToHost)
  - Non-blocking: `cudaMemcpyAsync` + pinned host memory + stream
- CPU 側計測と GPU 側計測の両方を用意して、観測視点の違いを確認する

## ディレクトリ構成

- `0bench_GPU2CPU_CPUTime/`
  - `gpu_d2h_overhead.cu`
  - `std::chrono` で CPU 壁時計時間を計測
  - 既存の実行ログ: `結果.txt`
- `1bench_GPU2CPU_GPUTime/`
  - `gpu_d2h_overhead_events.cu`
  - `cudaEventRecord` / `cudaEventElapsedTime` で GPU 時間を計測
- `2bench_CPU2GPU_GPU2CPU_GPUTime/`
  - `gpu_h2d_d2h_comm_only_events.cu`
  - `H2D -> 軽量kernel -> D2H` の3段を実行し、通信(H2D/D2H)のみを GPU event で分離計測
  - 実行ログ: `結果.txt`

## ベンチマーク内容

反復回数:

- `0bench` / `1bench`: `iterations = 100`
- `2bench`: `iterations = 1000`

共通条件:

- 転送サイズは `4 byte` (`int32` 1要素)
- 毎回カーネルで値を更新してから転送
- API固定オーバーヘッドを観測する目的で、最小サイズ転送を繰り返す

測定パターン:

1. Blocking
- 各反復で `kernel -> cudaMemcpy(DeviceToHost)` を実行
- `cudaMemcpy` が同期的に完了待ちする構成

2. Non-blocking
- pinned host メモリを `cudaMallocHost` で確保
- 単一 stream 上で `kernel -> cudaMemcpyAsync` を enqueue
- ループ後に `cudaStreamSynchronize`

注意:

- この実装は単一 stream で in-order 実行のため、基本的に `kernel(i)` と `memcpyAsync(i)` は直列です
- したがって「オーバーラップで速くなるか」を測るコードではなく、「API/同期方式のオーバーヘッド差」を見るコードです

## ビルドと実行

前提:

- NVIDIA GPU + Driver
- CUDA Toolkit (`nvcc`) が PATH に通っていること
- Windows PowerShell を想定

CPU 時間計測版:

```powershell
cd 0bench_GPU2CPU_CPUTime
nvcc -O2 -std=c++17 gpu_d2h_overhead.cu -o gpu_d2h_overhead.exe
.\gpu_d2h_overhead.exe
```

GPU 時間計測版:

```powershell
cd 1bench_GPU2CPU_GPUTime
nvcc -O2 -std=c++17 gpu_d2h_overhead_events.cu -o gpu_d2h_overhead_events.exe
.\gpu_d2h_overhead_events.exe
```

3段フロー通信計測版 (`H2D -> kernel -> D2H`, 通信のみ計測):

```powershell
cd 2bench_CPU2GPU_GPU2CPU_GPUTime
nvcc -O2 -std=c++17 gpu_h2d_d2h_comm_only_events.cu -o gpu_h2d_d2h_comm_only_events.exe
.\gpu_h2d_d2h_comm_only_events.exe
```

複数回実行して平均を取りたい場合:

```powershell
.\run_benchmarks.ps1 -Runs 5
```

## 実行環境 (2026-02-25)

- GPU: `NVIDIA GeForce RTX 5090`
- Driver: `576.88`
- NVIDIA-SMI 上の CUDA Version: `12.9`
- nvcc: `Cuda compilation tools, release 12.9, V12.9.41`

## 実行結果まとめ (2026-02-25)

### 既存ログ (CPU 計測版)

`0bench_GPU2CPU_CPUTime/結果.txt`:

- Blocking per-iter: `0.051204 ms`
- Non-blocking per-iter: `0.079183 ms`

### 最新単発実行

CPU 計測版 (`gpu_d2h_overhead.exe`):

- Blocking per-iter: `0.056503 ms`
- Non-blocking per-iter: `0.083247 ms`

GPU 計測版 (`gpu_d2h_overhead_events.exe`):

- Blocking per-iter: `0.056662 ms`
- Non-blocking per-iter: `0.089582 ms`

### 5 回実行の集計

| Measurement | Blocking per-iter | Non-blocking per-iter | Async / Blocking |
|---|---:|---:|---:|
| CPU wall time (`std::chrono`) | avg `0.060162 ms` (min `0.052793`, max `0.076594`) | avg `0.096655 ms` (min `0.085037`, max `0.104992`) | `1.61x` |
| GPU time (`cudaEvent`) | avg `0.060000 ms` (min `0.049311`, max `0.071500`) | avg `0.091897 ms` (min `0.086722`, max `0.099159`) | `1.53x` |

### 2bench: `H2D -> 軽量kernel -> D2H` の通信のみ計測 (GPU event)

単発実行:

- H2D per-iter: `0.002956 ms`
- D2H per-iter: `0.064179 ms`
- H2D + D2H per-iter: `0.067135 ms`
- D2H / H2D: `21.710x`

5回実行集計:

- H2D per-iter: avg `0.001832 ms` (min `0.001798`, max `0.001877`)
- D2H per-iter: avg `0.060616 ms` (min `0.056429`, max `0.067091`)
- D2H / H2D: avg `33.073x` (min `30.926`, max `35.742`)

1bench との比較 (同日5回平均):

- 1bench Blocking D2H (`0.052388 ms`) に対して 2bench D2H (`0.060616 ms`) は `1.16x`
- 1bench Non-blocking D2H (`0.092584 ms`) に対して 2bench D2H (`0.060616 ms`) は `0.65x`

### 2bench (1000反復, kernel完了待ちあり) の再測定

測定意図:

- 100反復だと 4 byte 転送の揺らぎが目立つため、`iterations=1000` で平均化する
- `kernelDone` イベントを使ってカーネル完了を明示待機し、D2H計測にカーネル時間が混ざりにくい状態にする
- 見たいものはあくまで GPU<->CPU 通信の固定コスト

単発実行:

- H2D per-iter: `0.001993 ms`
- D2H per-iter: `0.063996 ms`
- H2D + D2H per-iter: `0.065989 ms`
- D2H / H2D: `32.105x`

5回実行集計:

- H2D per-iter: avg `0.001953 ms` (min `0.001828`, max `0.002127`)
- D2H per-iter: avg `0.064211 ms` (min `0.063553`, max `0.064865`)
- H2D + D2H per-iter: avg `0.066164 ms` (min `0.065381`, max `0.066865`)
- D2H / H2D: avg `32.964x` (min `30.237`, max `34.833`)

## 読み取り方

- この条件では Non-blocking 構成のほうが遅い
  - 理由: 転送サイズが 4 byte と極小で、転送時間より API 呼び出し/同期の固定コストが支配的
  - また単一 stream 直列実行のため、オーバーラップによる改善がほぼ出ない
- 2bench では H2D と D2H が対称にはならず、D2H が有意に大きい
  - この実測では、4 byte の超小サイズ転送において D2H 側の固定オーバーヘッドが支配的
- `Checksum (ignore)` は最適化で結果が消えないことを確認するための値
  - `iterations=100` のとき `4950`、`iterations=1000` のとき `499500`

## 今後、目的に応じて拡張するなら

- 純粋な転送帯域を見たい場合
  - 転送サイズを増やす (例: 4KB, 64KB, 1MB, 16MB)
- オーバーラップ効果を見たい場合
  - 複数 stream + ダブルバッファで `kernel(n+1)` と `copy(n)` を重ねる
