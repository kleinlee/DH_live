let audioQueue = []; // 存储待播放的音频数据
let isPlaying = false; // 标记是否正在播放音频
let audioContext; // 定义在全局以便在用户交互后创建或恢复
let audioChunkIndex = 0;
let audioChunkSize = 0;
let isEnding = false; // SSE传输已结束
let isChatting = false; // 本轮聊天进程
let isNewChat = true;
let text = "";
let charIndex = 0;
let controller;
let buffer = ""; // 缓存区，用于存储不完整的 JSON 块

document.getElementById('startButton').addEventListener('click', function() {
    window.parent.postMessage('startVideo', '*');
});

function runPrompt() {
    if (isChatting) {
        userAbort();
        return;
    }
    isChatting = true;
    pushButton.innerText = "停止";
    initializeAudioContext();

    const inputValue = document.getElementById('textInput').value;
    if (inputValue) {
        controller = new AbortController();
        addMessageToScreen('sender', inputValue, true);
        isEnding = false;
        isNewChat = true;
        document.getElementById('textInput').value = "";

        fetch('http://localhost:8888/eb_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: inputValue, voice_id: "", voice_speed: "" }),
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
                            console.log("Received text:", data.text);
                            console.log("Received audio (Base64):", data.audio.length);
                            addMessageToScreen('receiver', data.text, isNewChat);
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

function base64ToUint8Array(base64) {
    const byteCharacters = atob(base64); // 解码 Base64
    const byteNumbers = new Uint8Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    return byteNumbers;
}

function createMessageElement(text, messageType) {
    const message = document.createElement('div');
    message.classList.add('message', messageType);
    message.textContent = text;
    return message;
}

function addMessageToScreen(sender, text, isNew) {
    const chatContent = document.querySelector('.chat-content');
    if (isNew) {
        const senderMessage = createMessageElement(text, sender);
        chatContent.appendChild(senderMessage);
    } else {
        chatContent.lastElementChild.textContent += text;
    }
    chatContent.scrollTop = chatContent.scrollHeight;
}

function userAbort() {
    controller.abort();
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().then(() => {
            console.log('AudioContext已关闭');
        });
    }
    audioQueue = [];
    isPlaying = false;
    audioChunkIndex = 0;
    audioChunkSize = 0;
    text = "";
    charIndex = 0;
    isChatting = false;
    pushButton.innerText = "开始";
}

function initializeAudioContext() {
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    } else if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

function playAudio() {
    if (!isPlaying && audioQueue.length > 0) {
        const arrayBuffer = audioQueue.shift();
        isPlaying = true;

        const view = new Uint8Array(arrayBuffer);
        const arrayBufferPtr = parent.Module._malloc(arrayBuffer.byteLength);
        parent.Module.HEAPU8.set(view, arrayBufferPtr);
        parent.Module._setAudioBuffer(arrayBufferPtr, arrayBuffer.byteLength);
        parent.Module._free(arrayBufferPtr);

        audioContext.decodeAudioData(arrayBuffer.buffer, function(audioBuffer) {
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(0);
            source.onended = function() {
                isPlaying = false;
                playAudio();
            };
        }, function(error) {
            console.error('Decode audio error', error);
        });
    } else if (isEnding) {
        isChatting = false;
        pushButton.innerText = "发送";
    }
}