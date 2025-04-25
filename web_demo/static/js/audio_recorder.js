// 将AudioWorklet处理逻辑转为字符串嵌入主文件
const workletCode = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (event) => {
      if (event.data === 'stop') {
        this.port.postMessage('prepare to stop');
        this.isStopped = true;
        if (this.buffer.length > 0 && this.buffer.length > this.targetSampleCount) {
          this.port.postMessage(new Int16Array(this.buffer));
          this.port.postMessage({'event':'stopped'});
          this.buffer = [];
        }
      }
    };
    this.buffer = [];
    this.targetSampleCount = 1024;
  }

  process(inputs) {
    const input = inputs[0];
    if (input.length > 0) {
      const inputData = input[0];
      // 优化数据转换
      const samples = inputData.map(sample => 
        Math.max(-32768, Math.min(32767, Math.round(sample * 32767)))
      );
      this.buffer.push(...samples);

      while (this.buffer.length >= this.targetSampleCount) {
        const pcmData = this.buffer.splice(0, this.targetSampleCount);
        this.port.postMessage(new Int16Array(pcmData));
        this.port.postMessage({'event':'sending'});
      }
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
`;
class PCMAudioRecorder {
  constructor() {
    this.audioContext = null;
    this.stream = null;
    this.currentSource = null;
    this.audioCallback = null;
  }

  async connect(audioCallback) {
    this.audioCallback = audioCallback;
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    }
    console.log('Current sample rate:', this.audioContext.sampleRate, 'Hz');

    // 生成动态worklet
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    try {
      await this.audioContext.audioWorklet.addModule(url);
      URL.revokeObjectURL(url); // 清除内存
    } catch (e) {
      console.error('Error loading AudioWorklet:', e);
      return;
    }

    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.currentSource = this.audioContext.createMediaStreamSource(this.stream);

    this.processorNode = new AudioWorkletNode(this.audioContext, 'pcm-processor');

    this.processorNode.port.onmessage = (event) => {
      if (event.data instanceof Int16Array) {
        this.audioCallback?.(event.data);
      } else if (event.data?.event === 'stopped') {
        console.log('Recorder stopped.');
      }
    };

    this.currentSource.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  stop() {
    if (this.processorNode) {
      this.processorNode.port.postMessage('stop');
      this.processorNode.disconnect();
      this.processorNode = null;
    }

    this.stream?.getTracks().forEach(track => track.stop());
    this.currentSource?.disconnect();

    if (this.audioContext) {
      this.audioContext.close().then(() => {
        this.audioContext = null;
      });
    }
  }
}

// 暴露到全局环境
window.PCMAudioRecorder = PCMAudioRecorder;