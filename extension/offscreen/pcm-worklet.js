// AudioWorkletProcessor: Downmixes stereo/multi-channel audio to mono, resamples from
// AudioContext sampleRate (e.g. 44100 or 48000 Hz) to 16000 Hz, and emits 100ms (1600-sample)
// Int16 and Float32 chunks.

class PCM16kWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.chunkSamples = 1600; // 100ms at 16,000 Hz (matches Pyannote 6,400-byte Float32 chunk)
    this.outBufferF32 = new Float32Array(this.chunkSamples);
    this.outWriteIndex = 0;
    this.fractionalPos = 0.0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const numChannels = input.length;
    const numFrames = input[0].length;
    const monoFrame = new Float32Array(numFrames);

    // 1. Downmix all channels to mono
    for (let i = 0; i < numFrames; i++) {
      let sum = 0.0;
      for (let c = 0; c < numChannels; c++) {
        sum += input[c][i];
      }
      monoFrame[i] = sum / numChannels;
    }

    // 2. Linear interpolation resampling from sampleRate -> 16000 Hz
    const step = sampleRate / this.targetSampleRate;
    while (this.fractionalPos < numFrames) {
      const idx = Math.floor(this.fractionalPos);
      const frac = this.fractionalPos - idx;
      const s0 = monoFrame[idx];
      const s1 = idx + 1 < numFrames ? monoFrame[idx + 1] : s0;
      const sample = s0 + frac * (s1 - s0);

      this.outBufferF32[this.outWriteIndex++] = Math.max(-1.0, Math.min(1.0, sample));

      if (this.outWriteIndex >= this.chunkSamples) {
        this.flushChunk();
      }
      this.fractionalPos += step;
    }
    this.fractionalPos -= numFrames;
    return true;
  }

  flushChunk() {
    const pcmInt16 = new Int16Array(this.chunkSamples);
    let sumSq = 0.0;
    for (let i = 0; i < this.chunkSamples; i++) {
      const s = this.outBufferF32[i];
      sumSq += s * s;
      pcmInt16[i] = s < 0 ? Math.max(-32768, Math.round(s * 32768)) : Math.min(32767, Math.round(s * 32767));
    }
    const rmsLinear = Math.sqrt(sumSq / this.chunkSamples);
    const f32Copy = new Float32Array(this.outBufferF32);

    this.port.postMessage(
      {
        pcmInt16Buffer: pcmInt16.buffer,
        pcmFloat32Buffer: f32Copy.buffer,
        rmsLinear,
      },
      [pcmInt16.buffer, f32Copy.buffer]
    );
    this.outWriteIndex = 0;
  }
}

registerProcessor('pcm-16k-worklet', PCM16kWorkletProcessor);
