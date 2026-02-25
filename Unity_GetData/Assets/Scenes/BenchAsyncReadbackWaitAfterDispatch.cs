using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

public class BenchAsyncReadbackWaitAfterDispatch : MonoBehaviour
{
    [Header("Compute shader with kernel 'CSMain'")]
    public ComputeShader cs;

    [Header("Buffer settings")]
    public int elementCount = 1024 * 1024; // 1M uint = 4MB
    public int threadsPerGroup = 256;      // must match [numthreads] in shader

    [Header("Benchmark settings")]
    public int iterations = 200;

    [Header("Schedule")]
    public int warmupStartFrame = 0;
    public int benchmarkFrame = 50;

    [Header("Frame cap (best-effort)")]
    public bool capTo60Fps = true;

    private ComputeBuffer buffer;
    private int kernel;
    private int updateCount;

    void Awake()
    {
        if (capTo60Fps)
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 60;
        }
    }

    void OnEnable()
    {
        if (cs == null)
        {
            Debug.LogError("ComputeShader not assigned.");
            enabled = false;
            return;
        }

        kernel = cs.FindKernel("CSMain");
        buffer = new ComputeBuffer(elementCount, sizeof(uint));

        cs.SetBuffer(kernel, "_Out", buffer);
        cs.SetInt("_Count", elementCount);

        updateCount = 0;
    }

    void OnDisable()
    {
        buffer?.Dispose();
        buffer = null;
    }

    void Update()
    {
        int u = updateCount++;
        if (u < warmupStartFrame) return;

        if (u < benchmarkFrame)
        {
            OneDispatchThenWait(tag: u);
            return;
        }

        if (u == benchmarkFrame)
        {
            RunBenchmark();
            enabled = false;
        }
    }

    void RunBenchmark()
    {
        OneDispatchThenWait(tag: -1);

        int groups = Mathf.CeilToInt(elementCount / (float)threadsPerGroup);

        var swTotal = Stopwatch.StartNew();

        double minMs = double.PositiveInfinity;
        double maxMs = 0.0;
        double sumMs = 0.0;

        for (int i = 0; i < iterations; i++)
        {
            cs.SetInt("_Tag", i);

            var sw = Stopwatch.StartNew();

            cs.Dispatch(kernel, groups, 1, 1);

            // Dispatch → Request → WaitForCompletion を同一計測区間に含める
            var req = AsyncGPUReadback.Request(buffer);
            req.WaitForCompletion();

            sw.Stop();

            double ms = sw.Elapsed.TotalMilliseconds;
            sumMs += ms;
            if (ms < minMs) minMs = ms;
            if (ms > maxMs) maxMs = ms;

            // GetDataは呼ばない。完了とエラーのみ確認
            if (req.hasError)
                Debug.LogWarning($"[Bench] hasError==true at iter={i}");
        }

        swTotal.Stop();

        double totalMs = swTotal.Elapsed.TotalMilliseconds;
        double avgMs = sumMs / iterations;

        double bytes = (double)elementCount * sizeof(uint);
        double gib = bytes / (1024.0 * 1024.0 * 1024.0);
        double gibPerSec = gib / (avgMs / 1000.0); // 参考値（stall込み）

        Debug.Log(
            $"[Bench AsyncGPUReadback.WaitForCompletion] " +
            $"elementCount={elementCount} (bytes={bytes:0}) groups={groups} " +
            $"iterations={iterations} total={totalMs:0.###}ms " +
            $"avg={avgMs:0.###}ms min={minMs:0.###}ms max={maxMs:0.###}ms " +
            $"approxThroughput={gibPerSec:0.###} GiB/s " +
            $"(Update={benchmarkFrame})"
        );
    }

    void OneDispatchThenWait(int tag)
    {
        int groups = Mathf.CeilToInt(elementCount / (float)threadsPerGroup);
        cs.SetInt("_Tag", tag);

        cs.Dispatch(kernel, groups, 1, 1);

        var req = AsyncGPUReadback.Request(buffer);
        req.WaitForCompletion();

        if (req.hasError)
            Debug.LogWarning($"[Warmup] hasError==true tag={tag}");
    }
}
