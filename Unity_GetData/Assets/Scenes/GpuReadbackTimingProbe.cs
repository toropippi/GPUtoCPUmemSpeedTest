using System;
using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

public class GpuReadbackTimingProbe : MonoBehaviour
{
    [Header("Assign a ComputeShader that has kernel 'CSMain'")]
    public ComputeShader cs;

    [Header("Experiment toggles")]
    public bool useGLFlush = false;

    [Header("CPU load in Update (ms). 1..16 recommended")]
    [Range(0, 20)]
    public int cpuLoadMs = 1;

    [Header("Buffer length (elements)")]
    public int elementCount = 1024;

    [Header("Dispatch threads per group in shader must match")]
    public int threadsPerGroup = 64;

    private ComputeBuffer buffer;
    private int kernel;
    private uint frameCnt;

    // For detecting late callbacks vs current frame
    private float lastUpdateStartTime;

    void Awake()
    {
        // 60fps固定（狙い）
        // vSyncが有効だとtargetFrameRateが効かない/効きにくい場合があるため、vSyncを切るのが定石。
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 60;
    }
    void OnEnable()
    {
        if (cs == null)
        {
            Debug.LogError("ComputeShader is not assigned.");
            enabled = false;
            return;
        }

        kernel = cs.FindKernel("CSMain");
        buffer = new ComputeBuffer(elementCount, sizeof(uint), ComputeBufferType.Structured);

        // Initialize to known values (optional)
        var init = new uint[elementCount];
        for (int i = 0; i < init.Length; i++) init[i] = 0;
        buffer.SetData(init);

        cs.SetBuffer(kernel, "_Out", buffer);
        cs.SetInt("_Count", elementCount);

        frameCnt = 0;
    }

    void OnDisable()
    {
        buffer?.Dispose();
        buffer = null;
    }

    void Update()
    {
        frameCnt++;
        var StartTime = Time.realtimeSinceStartup * 60f;

        // --- CPU load emulation ---
        // Thread.Sleep is coarse; on Windows you may see jitter (timer granularity).
        if (cpuLoadMs > 0)
            Thread.Sleep(cpuLoadMs);
        
        // --- GPU dispatch ---
        int groups = Mathf.CeilToInt(elementCount / (float)threadsPerGroup);
        cs.SetInt("_FrameTag", unchecked((int)frameCnt));

        cs.Dispatch(kernel, groups, 1, 1);
        
        // --- Async readback request (after dispatch) ---
        // Capture frameCnt at request time to print it in callback.
        uint requestFrame = frameCnt;
        float requestTime = Time.realtimeSinceStartup * 60f;
        
        
        AsyncGPUReadback.Request(buffer, (AsyncGPUReadbackRequest req) =>
        {
            float cbTime = Time.realtimeSinceStartup;

            if (req.hasError)
            {
                Debug.LogError($"[ReadbackCB] ERROR. reqFrame={requestFrame} nowFrame={frameCnt} " +
                               $"tReq={requestTime:F6}s tCb={cbTime:F6}s (dt={cbTime - requestTime:F6}s) " +
                               $"useGLFlush={useGLFlush} cpuLoadMs={cpuLoadMs}");
                return;
            }

            // Read a tiny sample to ensure data is actually accessed (avoid dead-code assumptions)
            var data = req.GetData<uint>();
            uint sample0 = data.Length > 0 ? data[0] : 0;

            Debug.Log("r" + sample0 + "_" + cbTime * 60f);
        });
        
        
        // If enabled, flush queued driver commands to GPU
        if (useGLFlush)
        {
            GL.Flush();
            buffer.GetData(new uint[1], 0, 0, 1); // Dummy read to ensure flush has an effect
        }
        Debug.Log("" + frameCnt + "_" + StartTime);

        

    }
}
