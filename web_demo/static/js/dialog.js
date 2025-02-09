let server_url = "http://localhost:8888/eb_stream"
let audioQueue = []; // 存储待播放的音频数据
let isPlaying = false; // 标记是否正在播放音频
let audioContext; // 定义在全局以便在用户交互后创建或恢复
let isEnding = false; // SSE传输已结束
let isChatting = false; // 本轮聊天进程
let isNewChat = true;
let controller;
let buffer = ""; // 缓存区，用于存储不完整的 JSON 块
let isRecording = false; // 标记当前是否正在录音

const toggleButton = document.getElementById('toggle-button');
const inputArea = document.getElementById('input-area');
const chatContainer = document.getElementById('chat-container');
const sendButton = document.getElementById('send-button');
const textInput = document.getElementById('text-input');
const voiceInputArea = document.getElementById('voice-input-area');
const voiceInputText = voiceInputArea.querySelector('span'); // 获取显示文字的 span 元素
let isVoiceMode = true;
let mediaRecorder;
let audioChunks = [];

// 初始设置为语音模式
function setVoiceMode() {
    isVoiceMode = true;
    toggleButton.innerHTML = '<i class="fas fa-keyboard"></i>';
    textInput.style.display = 'none';
    sendButton.style.display = 'none';
    voiceInputArea.style.display = 'flex';
}

// 初始设置为文字模式
function setTextMode() {
    isVoiceMode = false;
    toggleButton.innerHTML = '<i class="fas fa-microphone"></i>';
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

// 语音输入逻辑
voiceInputArea.addEventListener('click', async () => {
    event.preventDefault(); // 阻止默认行为

    if (isRecording) {
        // 如果正在录音，则停止录音
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            console.log('停止录音');
            isRecording = false;
            voiceInputText.textContent = '点击说话'; // 恢复文字
        }
    } else {
        // 如果未在录音，则开始录音
        if (controller) {
            user_abort();
        }

        if (isVoiceMode) {
            try {
                const stream = await getMediaStream();
                mediaRecorder = new MediaRecorder(stream);
                let startTime = Date.now(); // 记录录音开始时间

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const endTime = Date.now(); // 记录录音结束时间
                    const duration = (endTime - startTime) / 1000; // 计算录音时长（秒）

                    if (duration < 1.5) {
                        alert('录音时间过短');
                        audioChunks = []; // 清除录音数据
                    } else {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        audioChunks = [];
                        sendAudioMessage(audioBlob); // 直接发送 Blob 数据
                    }

                    stream.getTracks().forEach(track => track.stop()); // 停止录音
                    isRecording = false;
                    voiceInputText.textContent = '点击说话'; // 恢复文字
                };

                mediaRecorder.start();
                isRecording = true;
                voiceInputText.textContent = '点击结束'; // 修改文字
                console.log('开始录音');

                // 设置一个定时器，当录音时长超过 30 秒时自动停止录音
                const maxDuration = 30; // 最大录音时长（秒）
                setTimeout(() => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        console.log('录音超过 30 秒，自动停止并发送');
                    }
                }, maxDuration * 1000);

            } catch (error) {
                console.error('录音失败:', error);
                alert('无法访问麦克风，请确保已授予权限。');
            }
        }
    }
});

// 获取媒体流（兼容性处理）
async function getMediaStream() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        return await navigator.mediaDevices.getUserMedia({ audio: true });
    } else if (navigator.getUserMedia) { // 旧版API
        return new Promise((resolve, reject) => {
            navigator.getUserMedia({ audio: true }, resolve, reject);
        });
    } else {
        throw new Error('您的浏览器不支持录音功能');
    }
}

// 文字输入逻辑
sendButton.addEventListener('click', sendTextMessage);
textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendTextMessage();
    }
});

// 添加消息到聊天记录
function addMessage(message, isUser, isNew) {
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
        lastMessageContent.innerHTML += message;
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 初始设置为语音模式
setVoiceMode();

// 发送文字消息
function sendTextMessage() {
    if (isChatting) {
        user_abort();
        return;
    }
    isChatting = true;
    sendButton.innerHTML = '<i class="fas fa-stop"></i>'; // 停止图标
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } else if (audioContext.state === 'suspended') {
        audioContext.resume(); // 如果处于暂停状态，则恢复
    }

    const inputValue = textInput.value.trim();
    if (inputValue) {
        controller = new AbortController();
        addMessage(inputValue, true, true);
        isEnding = false;
        isNewChat = true;
        textInput.value = "";
        // 获取音色名称
        let characterName = "";
        const voiceDropdown = window.parent.document.getElementById('voiceDropdown');
        if (voiceDropdown) {
            characterName = voiceDropdown.value;
        }
        fetch(server_url, {
            method: 'post',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({"input_mode": "text", 'prompt': inputValue, 'voice_id': characterName, 'voice_speed': "" }),
            signal: controller.signal
        })
            .then(response => response.body)
            .then(body => {
                const reader = body.getReader();
                const decoder = new TextDecoder();
                function read() {
                    return reader.read().then(({ done, value }) => {
                        if (done) {
                            return;
                        }
                        const chunk = decoder.decode(value, { stream: true });
                        buffer += chunk; // 将新数据追加到缓存区

                        // 根据换行符拆分缓存区中的数据
                        const chunks = buffer.split("\n");
                        // 处理完整的 JSON 块
                        for (let i = 0; i < chunks.length - 1; i++) {
                            try {
                                const data = JSON.parse(chunks[i]);
                                console.log("Received text:", data.text, isNewChat);
                                console.log("Received audio (Base64):", data.audio.length);
                                addMessage(data.text, false, isNewChat);
                                isNewChat = false;
                                // 将 Base64 音频数据转换为 Uint8Array
                                const audioUint8Array = base64ToUint8Array(data.audio);
                                audioQueue.push(audioUint8Array); // 将 Uint8Array 推入队列
                                isEnding = data.endpoint;
                                playAudio();
                            } catch (error) {
                                console.error("Error parsing chunk:", error);
                            }
                        }

                        // 将最后一个不完整的块保留在缓存区中
                        buffer = chunks[chunks.length - 1];

                        return read();
                    });
                }
                return read();
            })
            .catch(error => {
                if (error.name === 'AbortError') {
                    console.log('Fetch aborted');
                } else {
                    console.error('Fetch error:', error);
                }
            });
    }
}

// 发送音频消息
function sendAudioMessage(audioBlob) {
    if (!audioBlob) {
        console.error('音频数据为空');
        return;
    }

    controller = new AbortController(); // 创建 AbortController

    // 将音频 Blob 转换为 Base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    isNewChat = true;
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } else if (audioContext.state === 'suspended') {
        audioContext.resume(); // 如果处于暂停状态，则恢复
    }
    reader.onloadend = function() {
        const base64Audio = reader.result.split(',')[1]; // 去掉 data URL 前缀

        // 获取音色名称
        let characterName = "";
        const voiceDropdown = window.parent.document.getElementById('voiceDropdown');
        if (voiceDropdown) {
            characterName = voiceDropdown.value;
        }

        const requestData = {
            input_mode: "audio",
            audio: base64Audio,
            voice_speed: "", // 可以根据需要调整
            voice_id: characterName // 可以根据需要调整
        };

        fetch(server_url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData),
            signal: controller.signal,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络响应错误');
            }
            return response.body;
        })
        .then(body => {
            const reader = body.getReader();
            const decoder = new TextDecoder();
            function read() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log('数据流读取完成');
                        return;
                    }
                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk; // 将新数据追加到缓存区

                    // 根据换行符拆分缓存区中的数据
                    const chunks = buffer.split("\n");

                    // 处理完整的 JSON 块
                    for (let i = 0; i < chunks.length - 1; i++) {
                        try {
                            const data = JSON.parse(chunks[i]);
                            // 如果data中包含key:prompt,那就解析出来
                            if (data.prompt) {
                                console.log("SSSSSS prompt", data.prompt);
                                addMessage(data.prompt, true, true); // 添加文本消息
                                continue;
                            }
                            console.log("Received text:", data.text, isNewChat);
                            console.log("Received audio (Base64):", data.audio.length);
                            addMessage(data.text, false, isNewChat); // 添加文本消息
                            isNewChat = false;

                            // 将 Base64 音频数据转换为 Uint8Array
                            const audioUint8Array = base64ToUint8Array(data.audio);
                            audioQueue.push(audioUint8Array); // 将 Uint8Array 推入队列
                            isEnding = data.endpoint;

                            playAudio(); // 播放音频
                        } catch (error) {
                            console.error("Error parsing chunk:", error);
                        }
                    }

                    // 将最后一个不完整的块保留在缓存区中
                    buffer = chunks[chunks.length - 1];

                    return read();
                });
            }
            return read();
        })
        .catch(error => {
            if (error.name === 'AbortError') {
                console.log('请求被中断');
            } else {
                console.error('请求错误:', error);
            }
        });
    };
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
function user_abort() {
    controller.abort();
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            console.log('AudioContext已关闭');
        });
    }
    audioQueue = []; // 清空音频队列
    isPlaying = false; // 标记音频播放结束
    isChatting = false;
    sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>'; // 发送图标
}

// 播放音频
function playAudio() {
    console.log("playAudio", audioQueue.length, isPlaying)
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
                source.onended = function() {
                    isPlaying = false;
                    playAudio();
                    // audioContext.close();
                };
            }, function(error) {
                console.error('Decode audio error', error);
            });
        } else {
            if (isEnding) {
                isChatting = false;
                sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>'; // 发送图标
            }
        }
    }
}