using UnityEngine;

public class BenchGetDataAfterDispatch : MonoBehaviour
{
    [Header("Compute shader with kernel 'CSMain'")]
    public ComputeShader cs;

    [Header("Benchmark settings")]
    public int elementCount = 1024 * 1024; // 1M uint = 4MB
    public int threadsPerGroup = 256;      // must match [numthreads] in shader
    public int iterations = 50;            // N loop count

    [Header("Schedule")]
    public int warmupStartFrame = 0;
    public int benchmarkFrame = 100;

    [Header("Frame cap (best-effort)")]
    public bool capTo60Fps = true;

    private ComputeBuffer buffer;
    private int kernel;
    private uint[] managed;
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
            UnityEngine.Debug.LogError("ComputeShader not assigned.");
            enabled = false;
            return;
        }

        kernel = cs.FindKernel("CSMain");

        buffer = new ComputeBuffer(elementCount, sizeof(uint));
        managed = new uint[elementCount];

        cs.SetBuffer(kernel, "_Out", buffer);
        cs.SetInt("_Count", elementCount);

        updateCount = 0;
    }

    void OnDisable()
    {
        buffer?.Dispose();
        buffer = null;
        managed = null;
    }

    void Update()
    {
        int u = updateCount++;
        if (u < warmupStartFrame) return;

        if (u < benchmarkFrame)
        {
            OneDispatchThenGetData(tag: u);
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
        OneDispatchThenGetData(tag: -1);

        int groups = Mathf.CeilToInt(elementCount / (float)threadsPerGroup);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            cs.SetInt("_Tag", i);
            cs.Dispatch(kernel, groups, 1, 1);
            buffer.GetData(managed);
        }

        sw.Stop();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double perIterMs = totalMs / iterations;

        double bytesPerIter = (double)elementCount * sizeof(uint);
        double gibPerSec = (bytesPerIter / (1024.0 * 1024.0 * 1024.0)) / (perIterMs / 1000.0);

        UnityEngine.Debug.Log(
            $"[Bench] elementCount={elementCount} (bytes/iter={bytesPerIter:0}) " +
            $"iterations={iterations} total={totalMs:0.###}ms perIter={perIterMs:0.###}ms " +
            $"approxThroughput={gibPerSec:0.###} GiB/s (Update={benchmarkFrame})"
        );
    }

    void OneDispatchThenGetData(int tag)
    {
        int groups = Mathf.CeilToInt(elementCount / (float)threadsPerGroup);
        cs.SetInt("_Tag", tag);
        cs.Dispatch(kernel, groups, 1, 1);
        buffer.GetData(managed);
        GL.Flush();
    }
}
