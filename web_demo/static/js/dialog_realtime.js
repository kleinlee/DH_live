let server_url = "http://localhost:8888/eb_stream"
let websocket_url = "ws://localhost:8888/asr?samplerate=16000"
let ws = null;   // ASR使用websocket双向流式连接
let isVoiceMode = true;                 // 默认使用语音模式

// 录音阶段
let asr_audio_recorder = new PCMAudioRecorder();
let isRecording = false;     // 标记当前录音是否向ws传输
let asr_input_text = "";     // 从ws接收到的ASR识别后的文本
let isNewASR = true;          // 开启新一轮的ASR,ASR返回文本要重新单独显示
let last_voice_time = null;   // 上一次检测到人声的时间
let last_3_voice_samples = [];
const VAD_SILENCE_DURATION = 800;  // 800ms不说话判定为讲话结束

// SSE 阶段（申请流式传输LLM+TTS的阶段）
let sse_startpoint = true;                // SSE传输开始标志
let sse_endpoint = false;                 // SSE传输结束标志
let sse_controller = null;                // SSE网络中断控制器，可用于打断传输
let sse_data_buffer = "";                 // SSE网络传输数据缓存区，用于存储不完整的 JSON 块

// 播放音频阶段
let isPlaying = false; // 标记是否正在播放音频
let audioQueue = []; // 存储待播放的音频数据
let audioContext; // 定义在全局以便在用户交互后创建或恢复


const toggleButton = document.getElementById('toggle-button');
const inputArea = document.getElementById('input-area');
const chatContainer = document.getElementById('chat-container');
const sendButton = document.getElementById('send-button');
const textInput = document.getElementById('text-input');
const voiceInputArea = document.getElementById('voice-input-area');
const voiceInputText = voiceInputArea.querySelector('span'); // 获取显示文字的 span 元素

document.addEventListener('DOMContentLoaded', function() {
  if (!window.isSecureContext) {
    alert('本项目使用了 WebCodecs API，该 API 仅在安全上下文（HTTPS 或 localhost）中可用。因此，在部署或测试时，请确保您的网页在 HTTPS 环境下运行，或者使用 localhost 进行本地测试。');
  }
});

// 初始设置为语音模式
function setVoiceMode() {
    isVoiceMode = true;
    toggleButton.innerHTML = '<i class="material-icons">keyboard</i>';
    textInput.style.display = 'none';
    sendButton.style.display = 'none';
    voiceInputArea.style.display = 'flex';
    voiceInputText.textContent = '点击重新开始对话'; // 恢复文字
}

// 初始设置为文字模式
function setTextMode() {
    isVoiceMode = false;
    toggleButton.innerHTML = '<i class="material-icons">mic</i>';
    textInput.style.display = 'block';
    sendButton.style.display = 'block';
    voiceInputArea.style.display = 'none';
}

// 切换输入模式
toggleButton.addEventListener('click', () => {
    console.log("toggleButton", isVoiceMode)
    if (isVoiceMode) {
        setTextMode();
    } else {
        setVoiceMode();
    }
});

async function running_audio_recorder() {
    await asr_audio_recorder.connect(async (pcmData) => {
            last_3_voice_samples.push(pcmData);
            if (last_3_voice_samples.length > 3) {
                last_3_voice_samples = last_3_voice_samples.slice(-3);
            }
            console.log('recording and send audio', pcmData.length, ws.readyState);

            // PCM数据处理,只取前 512 个 int16 数据
            const uint8Data = new Uint8Array(pcmData.buffer, 0, 512 * 2);
            const arrayBufferPtr = parent.Module._malloc(uint8Data.byteLength);
            parent.Module.HEAPU8.set(uint8Data, arrayBufferPtr);

            // VAD检测,speech_score(0-1)代表检测到人声的置信度
            const speech_score = parent.Module._getAudioVad(arrayBufferPtr, uint8Data.byteLength);
            parent.Module._free(arrayBufferPtr); // 确保内存释放

            // console.log('VAD Result:', speech_score);
            let current_time = Date.now();
            if (speech_score > 0.5)
            {
                if (!ws || ws.readyState !== WebSocket.OPEN)
                {
                    await asr_realtime_ws();
                }
                if (!isRecording)
                {
                    isRecording = true;
                    // 先发送两个历史语音，保证ASR不会遗漏首字符
                    if (last_3_voice_samples && last_3_voice_samples.length >= 2) {
                        ws.send(last_3_voice_samples[0]);
                        ws.send(last_3_voice_samples[1]);
                    }
                }
                ws.send(pcmData.buffer);
                last_voice_time = current_time;
            }
            else
            {
                if (isRecording)
                {
                    if (last_voice_time && (current_time - last_voice_time) > VAD_SILENCE_DURATION)
                    {
                        isRecording = false;
                        last_voice_time = null;
                        console.log("vad");
                        ws.send('vad');
                        await asr_audio_recorder.stop();
                    }
                    else
                    {
                        ws.send(pcmData.buffer);
                    }
                }
            }
        });
}

async function start_new_round() {
    isRecording = false;
    isNewASR = true;
    asr_input_text = "";

    if (isVoiceMode)
    {
        if (!ws)
        {
            await asr_realtime_ws();
        }
        await running_audio_recorder();
    }
}

async function asr_realtime_ws() {
    try {
        if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.close();
        }
        ws = new WebSocket(websocket_url);
        // Create a promise that resolves when the connection is open
        const connectionPromise = new Promise((resolve, reject) => {
            ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
        });
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Received data:', data);
            if (data.idx == -1) {
                console.log('asr round finished: ', asr_input_text);
                // ws.close();
                sendTextMessage(asr_input_text);
            }
            else {
                let text = data.text;
                let is_end = data.finished;
                console.log("addMessage: ", text, is_end, isNewASR);
                {
                    asr_input_text = text;
                    addMessage(text, true, isNewASR, true);
                    isNewASR = false;
                }
            }
        }
        ws.onclose = (event) => {
            console.log('WebSocket closed');
            ws = null;
        };

        await connectionPromise;
        console.log('ws connected and ready');

    } catch (error) {
        console.error('Error:', error);
        ws = null; // Ensure ws is set to null on error
        throw error;
    }
}

// 语音输入逻辑
voiceInputArea.addEventListener('click', async (event) => {
    event.preventDefault(); // 阻止默认行为
    console.log("voiceInputArea click")
    await user_abort();
    voiceInputText.textContent = '点击重新开始对话'; // 恢复文字
    await start_new_round();
});

// 文字输入逻辑
sendButton.addEventListener('click', (e) => {
    const icon = sendButton.querySelector('i.material-icons');
    // 检查是否存在图标且图标内容为 'stop'
    if (icon && icon.textContent.trim() === 'stop') {
        user_abort();
        return;
    }
    const inputValue = textInput.value.trim();
    if (inputValue) {
        addMessage(inputValue, true, true);
        sendTextMessage(inputValue);
    }
});
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const inputValue = textInput.value.trim();
        if (inputValue) {
            addMessage(inputValue, true, true);
            sendTextMessage(inputValue);
        }
    }
});

// 添加消息到聊天记录
function addMessage(message, isUser, isNew, replace=false) {
    if (isNew) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user' : 'ai');
        messageElement.innerHTML = `
            <div class="message-content">${message}</div>
        `;
        chatContainer.appendChild(messageElement);
    } else {
        // 直接操作 innerHTML 或使用 append 方法
        const lastMessageContent = chatContainer.lastElementChild.querySelector('.message-content');
        if (replace)
        {
            lastMessageContent.innerHTML = message;
        }
        else
        {
            lastMessageContent.innerHTML += message;
        }
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 初始设置为语音模式
setVoiceMode();

function initAudioContext() {
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } else if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

async function handleResponseStream(responseBody, isNewSession) {
    const reader = responseBody.getReader();
    const decoder = new TextDecoder();

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                return;
            }
            const chunk = decoder.decode(value, { stream: true });
            sse_data_buffer += chunk; // 将新数据追加到缓存区

            // 根据换行符拆分缓存区中的数据
            const chunks = sse_data_buffer.split("\n");
            // 处理完整的 JSON 块
            for (let i = 0; i < chunks.length - 1; i++) {
                try {
                    const data = JSON.parse(chunks[i]);
                    console.log("Received text:", data.text, sse_startpoint);
                    console.log("Received audio (Base64):", data.audio.length);
                    addMessage(data.text, false, sse_startpoint);
                    sse_startpoint = false;
                    // 将 Base64 音频数据转换为 Uint8Array
                    if (data.audio)
                    {
                        const audioUint8Array = base64ToUint8Array(data.audio);
                        audioQueue.push(audioUint8Array); // 将 Uint8Array 推入队列
                    }
                    sse_endpoint = data.endpoint;
                    playAudio();
                } catch (error) {
                    console.error("Error parsing chunk:", error);
                }
            }
            // 将最后一个不完整的块保留在缓存区中
            sse_data_buffer = chunks[chunks.length - 1];
        }
    } catch (error) {
        console.error('流处理异常:', error);
    }
}

// 发送文字消息
async function sendTextMessage(inputValue) {
    sendButton.innerHTML = '<i class="material-icons">stop</i>';
    initAudioContext();
    if (inputValue) {
        sse_controller = new AbortController();
        sse_startpoint = true;
        sse_endpoint = false;
        textInput.value = "";
        // 获取音色名称
        let characterName = "";
        const voiceDropdown = window.parent.document.getElementById('voiceDropdown');
        if (voiceDropdown) {
            const parsedValue = parseInt(voiceDropdown.value, 10);
            if (!isNaN(parsedValue)) {
                characterName = parsedValue;
            } else {
                characterName = ""; // 默认值
            }
        }
        let requestBody = {"input_mode": "text", 'prompt': inputValue, 'voice_id': characterName, 'voice_speed': "" }
        try {
            const response = await fetch(server_url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
                signal: sse_controller.signal
            });

            if (!response.ok) throw new Error(`HTTP错误 ${response.status}`);
            await handleResponseStream(response.body, true);
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('请求中止');
            } else {
                console.error('请求错误:', error);
            }
            start_new_round();
        }
    }
    else
    {
        start_new_round();
    }
}

// 将 Base64 转换为 Uint8Array
function base64ToUint8Array(base64) {
    const byteCharacters = atob(base64); // 解码 Base64
    const byteNumbers = new Uint8Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    return byteNumbers;
}

// 用户中断操作
async function user_abort() {
    if (isVoiceMode)
    {
        await asr_audio_recorder.stop();
        if (ws) {
            isASRActive = false;
            ws.close();
            ws = null;
        }
    }

    if (sse_controller)
    {
        sse_controller.abort();
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            console.log('AudioContext已关闭');
        });
    }
    parent.Module._clearAudio();
    audioQueue = []; // 清空音频队列
    isPlaying = false; // 标记音频播放结束
    sendButton.innerHTML = '<i class="material-icons">send</i>'; // 发送图标
}

// 播放音频
async function playAudio() {
    console.log("playAudio", audioQueue.length, isPlaying);
    if (!isPlaying) {
        if (audioQueue.length > 0) {
            let arrayBuffer = audioQueue.shift();
            isPlaying = true;
            console.log("playAudio arrayBuffer", audioQueue.length, isPlaying)

            const view = new Uint8Array(arrayBuffer);
            const arrayBufferPtr = parent.Module._malloc(arrayBuffer.byteLength);
            parent.Module.HEAPU8.set(view, arrayBufferPtr);
            console.log("buffer.byteLength", arrayBuffer.byteLength);
            parent.Module._setAudioBuffer(arrayBufferPtr, arrayBuffer.byteLength);
            parent.Module._free(arrayBufferPtr);

            // 将 Uint8Array 转换为 ArrayBuffer
            const arrayBuffer2 = arrayBuffer.buffer;
            // 解码ArrayBuffer为AudioBuffer
            audioContext.decodeAudioData(arrayBuffer2, function(audioBuffer) {
                // 创建BufferSource节点
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                // 连接到输出并播放
                source.connect(audioContext.destination);
                source.start(0);
                // 当音频播放结束时释放资源
                source.onended = async function() {
                    isPlaying = false;
                    await playAudio();
                    // audioContext.close();
                };
            }, function(error) {
                console.error('Decode audio error', error);
            });
        } else {
            if (sse_endpoint) {
                sendButton.innerHTML = '<i class="material-icons">send</i>'; // 发送图标
                await start_new_round();
            }
        }
    }
}