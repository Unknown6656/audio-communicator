using System.Runtime.InteropServices;

using NAudio.Wave.SampleProviders;
using NAudio.Wave;
using NAudio;

using Unknown6656.Generics;
using System.Collections.Concurrent;
using NAudio.CoreAudioApi;
using System.Numerics;
using System.Security.Principal;
using System.Reflection.Metadata.Ecma335;

namespace Unknown6656.AudioCommunicator;


public static class Communicator
{
    public static async Task kek()
    {
        OutputWaveProvider provider = new(new OutputWaveConfiguration());
        WaveOut output = new();

        output.Init(provider);
        output.Play();

        for (byte b = 0; b < 0xff; b += 1)
        {
            //provider._data = b;

            Console.WriteLine($"{b:x2}");
            await Task.Delay(100);
            output.Stop();
            await Task.Delay(10);
            output.Play();
        }

        output.Stop();
    }

    public static async Task lol(int fft_size = 4096)
    {
        using MMDeviceEnumerator enumerator = new();
        using MMDevice device = enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active)
                                          .First(c => c.ID == "{0.0.1.00000000}.{af13e491-40f3-4fec-97f3-39174f1fa91e}");
        //using MMDevice device = WasapiCapture.GetDefaultCaptureDevice();

        ConcurrentQueue<float[]> queue = new();
        using WasapiCapture capture = new(device)
        {
            ShareMode = AudioClientShareMode.Shared,
            WaveFormat = device.AudioClient.MixFormat,// new WaveFormat(48000, 32, 1),
        };

        capture.DataAvailable += (_, args) =>
        {
            unsafe
            {
                float[] fbuffer = new float[args.BytesRecorded / sizeof(float)];

                fixed (byte* ptr = args.Buffer)
                {
                    float* fptr = (float*)ptr;

                    Parallel.For(0, fbuffer.Length, i => fbuffer[i] = fptr[i] is float f && float.IsFinite(f) ? f : 0f);
                }

                queue.Enqueue(fbuffer);
            }
        };
        capture.StartRecording();

        float[]? last = null;

        while (true)
            if (queue.TryDequeue(out float[]? fbuffer))
            {
                if (!queue.IsEmpty)
                    queue.Clear();

                int index = 0;

                if (last != null)
                    fbuffer = last.Concat(fbuffer).ToArray();

                for (; index < fbuffer.Length - fft_size; index += fft_size)
                {
                    float[] slice = fbuffer[index..(index + fft_size)];
                    (float frequency_hz, float amplitude)[] spectrum = FFT(slice, capture.WaveFormat.SampleRate, FFTWindowMode.Hamming);



                    // TODO


                    //Console.WriteLine(buffer[..10].Select(c => $"{c.X:F3}|{c.Y:F3}").StringJoin(" "));

                    foreach ((float frequency_hz, float amplitude) in spectrum)
                        Console.WriteLine($"{frequency_hz,8:F2}: {amplitude,10}");

                    Console.WriteLine();
                }

                last = index < fbuffer.Length - 1 ? fbuffer[index..] : null;
            }
            else
                await Task.Delay(100);

        capture.StopRecording();
        capture.Dispose();

        //byte[] buffer = new byte[8192];
        //int bytesRead;

        //do
        //{
        //    bytesRead = stream16.Read(buffer, 0, buffer.Length);
        //    converted.WriteData(buffer, 0, bytesRead);
        //} while (bytesRead != 0 && converted.Length < reader.Length);



    }


    public enum FFTWindowMode
    {
        Hamming,
        Hann,
        BlackmannHarris
    }

    public static (float frequency_hz, float amplitude)[] FFT(float[] waveform, int sample_rate, FFTWindowMode window = FFTWindowMode.Hamming)
    {
        int fft_size = waveform.Length;
        int m = (int)Math.Log2(fft_size);
        Func<double, double> window_func = window switch
        {
            FFTWindowMode.Hamming => f => .54 - .46 * Math.Cos(f),
            FFTWindowMode.Hann => f => .5 * (1 - Math.Cos(f)),
            FFTWindowMode.BlackmannHarris => f => .35875 - (.48829 * Math.Cos(f)) + (.14128 * Math.Cos(2 * f)) - (.01168 * Math.Cos(3 * f)),
            _ => throw new ArgumentOutOfRangeException(nameof(window)),
        };
        Complex[] spectrum = waveform.ToArray((f, i) => new Complex(f * window_func(i * Math.Tau / (fft_size - 1)), 0d));

        int n = 1 << m;
        int i2 = n >> 1;
        int i1;

        for (int i = 0, j = 0; i < n - 1; i++)
        {
            if (i < j)
                (spectrum[i], spectrum[j]) = (spectrum[j], spectrum[i]);

            int k = i2;

            while (k <= j)
            {
                j -= k;
                k >>= 1;
            }

            j += k;
        }

        float c1 = -1f;
        float c2 = 0f;
        int l2 = 1;

        for (int l = 0; l < m; ++l)
        {
            float u1 = 1f;
            float u2 = 0f;
            int l1 = l2;

            l2 <<= 1;

            for (int j = 0; j < l1; ++j)
            {
                for (int i = j; i < n; i += l2)
                {
                    i1 = i + l1;

                    Complex t = new(
                        u1 * spectrum[i1].Real - u2 * spectrum[i1].Imaginary,
                        u1 * spectrum[i1].Imaginary + u2 * spectrum[i1].Real
                    );

                    spectrum[i1] = spectrum[i] - t;
                    spectrum[i] += t;
                }

                float z = u1 * c1 - u2 * c2;
                u2 = u1 * c2 + u2 * c1;
                u1 = z;
            }

            c2 = -(float)Math.Sqrt((1 - c1) * .5);
            c1 = (float)Math.Sqrt((1 + c1) * .5);
        }

        for (int i = 0; i < n; i++)
            spectrum[i] /= n;

        return spectrum.Take(fft_size / 4).ToArray((c, i) => (i * sample_rate * 2f / fft_size, (float)c.Magnitude));
    }
}


/* low                                     mid                                     high
    |----'----|----'----|----'----|----'----|----'----|----'----|----'----|----'----|
    |    '    '    '    '    '    '    '    '    '    '    '    '    '    '    '    |
    |  |<-->|    |<-->|    |<-->|    |<-->|    |<-->|    |<-->|    |<-->|    |<-->| |
    |   CH-1      CH-2      CH-3      CH-4      CH-5      CH-6      CH-7      CH-8  |
    base                                                                            base + bandwidth
*/
public sealed record OutputWaveConfiguration(
    float MidFrequency = 8_000,
    float Bandwidth = 4_000,
    int SampleRate = 48000,
    int AudioChannelCount = 1
);

public sealed unsafe class OutputWaveProvider
    : IWaveProvider
{
    private volatile int _sample_number;
    public volatile byte _raw_data_byte;


    public static float Margin { get; set; } = 1.15f;
    public OutputWaveConfiguration Configuration { get; }
    public WaveFormat WaveFormat { get; }


    public OutputWaveProvider(OutputWaveConfiguration configuration)
    {
        Configuration = configuration;
        WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(configuration.SampleRate, configuration.AudioChannelCount);

        float max_bw = (2 - Margin) / (1 + .5f * Margin) * configuration.MidFrequency;

        if (configuration.Bandwidth > max_bw)
            throw new ArgumentException($"A bandwidth of {configuration.Bandwidth / 1000:F1} kHz is to large for a center frequency of {configuration.MidFrequency / 1000:F1} kHz. The bandwidth must be smaller than {max_bw / 1000:F1} kHz (including the margin of {Margin:F1}).", nameof(configuration));
    }


    public int Read(byte[] buffer, int offset, int count)
    {
        WaveBuffer wbuffer = new(buffer);
        float[] fbuffer = wbuffer.FloatBuffer;
        int sample_rate = WaveFormat.SampleRate;

        offset /= sizeof(float);

        for (int n = 0; n < count / sizeof(float); n++)
        {
            float phase = (float)_sample_number / sample_rate;

            fbuffer[n + offset] = .1f * GenerateParallelSignal(phase, _raw_data_byte);
            _sample_number = (_sample_number + 1) % sample_rate;
        }

        return count;
    }

    private float GenerateParallelSignal<T>(float phase, T data)
        where T : unmanaged
    {
        bool[] bits = new bool[sizeof(T) * 8];
        byte* ptr = (byte*)&data;

        for (int i = 0; i < bits.Length; ++i)
            bits[i] = ((ptr[i / 8] >> (i % 8)) & 1) != 0;

        return GenerateParallelSignal(phase, bits);
    }

    private float GenerateParallelSignal(float phase, bool[] bits, int bit_count)
    {
        Array.Resize(ref bits, bit_count);

        return GenerateParallelSignal(phase, bits);
    }

    private float GenerateParallelSignal(float phase, bool[] bits)
    {
        float lo = Configuration.MidFrequency - .5f * Configuration.Bandwidth;
        float hi = Configuration.MidFrequency + .5f * Configuration.Bandwidth;
        float step = Configuration.Bandwidth / (bits.Length * 4 + 1);
        (float f, float a)[] frequencies = new (float, float)[bits.Length + 2];

        for (int i = 0; i < bits.Length; ++i)
            frequencies[i] = (((bits[i] ? 3 : 1) + i * 4) * step + lo, 1);

        frequencies[^2] = (lo, .1f);
        frequencies[^1] = (hi, .5f);

        return GenerateSines(phase, frequencies);
    }

    private static float GenerateSines(float phase, IEnumerable<(float frequency, float amplitude)> waves)
    {
        double result = 0;

        foreach ((float frequency, float amplitude) in waves)
            result += amplitude * Math.Sin(2 * Math.PI * phase * frequency);

        return (float)result;
    }
}
