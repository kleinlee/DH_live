from typing import *
import logging
import time
import logging
import sherpa_onnx
import os
import asyncio
import numpy as np
import threading
logger = logging.getLogger(__file__)
_asr_engines = {}


class ASRResult:
    def __init__(self, text: str, finished: bool, idx: int):
        self.text = text
        self.finished = finished
        self.idx = idx

    def to_dict(self):
        return {"text": self.text, "finished": self.finished, "idx": self.idx}

import wave
class ASRStream:
    def __init__(self, recognizer: Union[sherpa_onnx.OnlineRecognizer | sherpa_onnx.OfflineRecognizer], sample_rate: int) -> None:
        self.recognizer = recognizer
        self.inbuf = asyncio.Queue()
        self.outbuf = asyncio.Queue()
        self.sample_rate = sample_rate
        self.is_closed = False
        self.online = isinstance(recognizer, sherpa_onnx.OnlineRecognizer)

    async def start(self):
        if self.online:
            asyncio.create_task(self.run_online())
        else:
            asyncio.create_task(self.run_offline())

    async def run_online(self):
        stream = self.recognizer.create_stream()
        last_result = ""
        segment_id = 0
        gggg = 0
        print('asr: start real-time recognizer')
        while not self.is_closed:
            samples = await self.inbuf.get()
            gggg += len(samples)
            print("AAAAA", gggg)

            if isinstance(samples, str) and samples == "vad":
                print("VAD detected")

                self.outbuf.put_nowait(
                    ASRResult("", True, -1))
                self.recognizer.reset(stream)
                continue

            stream.accept_waveform(self.sample_rate, samples)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            is_endpoint = self.recognizer.is_endpoint(stream)
            result = self.recognizer.get_result(stream)

            print("AAAAA2", is_endpoint, result)

            if result and (last_result != result):
                last_result = result
                print(f' > {segment_id}:{result}')
                self.outbuf.put_nowait(
                    ASRResult(result, False, segment_id))

            if is_endpoint:
                if result:
                    print(f'{segment_id}: {result}')
                    self.outbuf.put_nowait(
                        ASRResult(result, True, segment_id))
                    segment_id += 1
                self.recognizer.reset(stream)

    async def run_offline(self):
        vad = _asr_engines['vad']
        segment_id = 0
        st = None
        while not self.is_closed:
            samples = await self.inbuf.get()
            vad.accept_waveform(samples)
            while not vad.empty():
                if not st:
                    st = time.time()
                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, vad.front.samples)

                vad.pop()
                self.recognizer.decode_stream(stream)

                result = stream.result.text.strip()
                if result:
                    duration = time.time() - st
                    print(f'{segment_id}:{result} ({duration:.2f}s)')
                    self.outbuf.put_nowait(ASRResult(result, True, segment_id))
                    segment_id += 1
            st = None

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)

    async def write(self, pcm_bytes: bytes):
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples = pcm_data.astype(np.float32) / 32768.0
        self.inbuf.put_nowait(samples)

    async def vad_touched(self):
        samples = "vad"
        self.inbuf.put_nowait(samples)

    async def read(self) -> ASRResult:
        return await self.outbuf.get()


def create_zipformer(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    encoder = os.path.join(d, "encoder-epoch-99-avg-1.onnx")
    decoder = os.path.join(d, "decoder-epoch-99-avg-1.onnx")
    joiner = os.path.join(d, "joiner-epoch-99-avg-1.onnx")
    tokens = os.path.join(d, "tokens.txt")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        provider=args.asr_provider,
        num_threads=args.threads,
        sample_rate=samplerate,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,  # it essentially disables this rule
    )
    return recognizer


def create_sensevoice(samplerate: int, args) -> sherpa_onnx.OfflineRecognizer:
    d = os.path.join(args.models_root,
                     'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17')

    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=os.path.join(d, 'model.onnx'),
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        use_itn=True,
        debug=0,
        language=args.asr_lang,
    )
    return recognizer


def create_paraformer_trilingual(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-paraformer-trilingual-zh-cantonese-en')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(d, 'model.onnx'),
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer


def create_paraformer_en(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-paraformer-en')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(d, 'model.onnx'),
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        use_itn=True,
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer


def create_whisper(samplerate: int, args) -> sherpa_onnx.OfflineRecognizer:
    # Assuming model name like 'whisper-tiny.en', 'whisper-medium', etc.
    # and a directory structure like 'sherpa-onnx-whisper-tiny.en'
    model_dir_name = f'sherpa-onnx-{args.asr_model}'
    d = os.path.join(args.models_root, model_dir_name)
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    # Extract the base model name (e.g., 'medium.en', 'tiny')
    base_model_name = args.asr_model.replace('whisper-', '', 1)

    # Construct filenames using the base name
    # TODO: Decide whether to use int8 quantized models if available.
    #       This example prioritizes non-quantized models first.
    encoder_path = os.path.join(d, f'{base_model_name}-encoder.onnx')
    decoder_path = os.path.join(d, f'{base_model_name}-decoder.onnx')
    tokens_path = os.path.join(d, f'{base_model_name}-tokens.txt')  # Adjusted tokens filename too

    # Fallback to int8 models if non-quantized don't exist
    if not os.path.exists(encoder_path):
        encoder_path = os.path.join(d, f'{base_model_name}-encoder.int8.onnx')
    if not os.path.exists(decoder_path):
        decoder_path = os.path.join(d, f'{base_model_name}-decoder.int8.onnx')

    # Check if model files exist after potential fallback
    if not os.path.exists(encoder_path):
        raise ValueError(f"asr: encoder model not found at {encoder_path}")
    if not os.path.exists(decoder_path):
        raise ValueError(f"asr: decoder model not found at {decoder_path}")
    if not os.path.exists(tokens_path):
        raise ValueError(f"asr: tokens file not found at {tokens_path}")

    # Assuming a factory function like this exists in sherpa-onnx
    # --- IMPORTANT: Verify this factory function and parameters ---
    recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=encoder_path,
        decoder=decoder_path,
        tokens=tokens_path,
        num_threads=args.threads,
        language=args.asr_lang or "",  # Whisper might need language hint
        task="transcribe",  # or "translate"
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer


def load_asr_engine(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    cache_engine = _asr_engines.get(args.asr_model)
    if cache_engine:
        return cache_engine
    st = time.time()
    if args.asr_model == 'zipformer-bilingual':
        cache_engine = create_zipformer(samplerate, args)
    elif args.asr_model == 'sensevoice':
        cache_engine = create_sensevoice(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'paraformer-trilingual':
        cache_engine = create_paraformer_trilingual(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'paraformer-en':
        cache_engine = create_paraformer_en(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model.startswith('whisper-'):
        cache_engine = create_whisper(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)  # Use VAD for offline processing
    else:
        raise ValueError(f"asr: unknown model {args.asr_model}")
    _asr_engines[args.asr_model] = cache_engine
    print(f"asr: engine loaded in {time.time() - st:.2f}s")
    return cache_engine


def load_vad_engine(samplerate: int, args, min_silence_duration: float = 0.25, buffer_size_in_seconds: int = 100) -> sherpa_onnx.VoiceActivityDetector:
    config = sherpa_onnx.VadModelConfig()
    d = os.path.join(args.models_root, 'silero_vad')
    if not os.path.exists(d):
        raise ValueError(f"vad: model not found {d}")

    config.silero_vad.model = os.path.join(d, 'silero_vad.onnx')
    config.silero_vad.min_silence_duration = min_silence_duration
    config.sample_rate = samplerate
    config.provider = args.asr_provider
    config.num_threads = args.threads

    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=buffer_size_in_seconds)
    return vad


async def start_asr_stream(samplerate: int, args) -> ASRStream:
    """
    Start a ASR stream
    """
    print(samplerate, args)
    # 获取全局共享的ASR引擎
    asr_engine = ASREngineManager.get_engine()

    stream = ASRStream(asr_engine, samplerate)
    await stream.start()
    return stream

# 1. 全局模型管理类
class ASREngineManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.engine = None
            return cls._instance

    @classmethod
    def initialize(cls, samplerate, args):
        instance = cls()
        if instance.engine is None:  # 安全访问属性
            instance.engine = load_asr_engine(samplerate, args)

    @classmethod
    def get_engine(cls):
        instance = cls()  # 确保实例存在
        return instance.engine  # 安全访问属性