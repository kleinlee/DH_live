// 判断是否为iOS系统
const tag_ios = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
                (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

// 判断iOS版本是否是17以上
let tag_ios17 = false;
if (tag_ios) {
    // 从用户代理中提取iOS版本号
    const match = navigator.userAgent.match(/OS (\d+)_(\d+)_?(\d+)?/);
    if (match && match[1]) {
        const iosVersion = parseInt(match[1], 10);
        tag_ios17 = iosVersion >= 17;

        if (!tag_ios17) {
            alert("iOS系统目前不支持iOS17以下版本，请升级后再试");
        }
    } else {
        // 无法获取版本号的情况
        alert("无法检测您的iOS版本，请确保使用iOS17或更高版本");
    }
}

let fps_enabled = true; // 全局参数，控制是否显示FPS
let frameTimes = []; // 用于存储最近几帧的时间戳
let ctxEl = canvasEl.getContext("2d");
class VideoProcessor {
    constructor() {
        this.mp4box = MP4Box.createFile();
        this.videoTrack = null;
        this.videoDecoder = null;
        this.videoFrames = [];
        this.nbSampleTotal = 0;
        this.countSample = 0;
        this.isReverseAdded = false;

        // 绑定事件处理函数
        this.mp4box.onReady = this.handleReady.bind(this);
        this.mp4box.onSamples = this.handleSamples.bind(this);

        this.combinedData = null;
        this.offscreenCanvas = null;
        this.offscreenCtx = null;
    }

    async init(videoUrl, gzipUrl) {
        // 清空旧数据
        this.videoFrames = [];
        this.countSample = 0;
        this.isReverseAdded = false;
        this.combinedData = null;
        // 重置 MP4Box
        if (this.mp4box) {
            this.mp4box.stop(); // 停止解析
            this.mp4box.flush(); // 清空缓冲区
            this.mp4box = null; // 销毁旧实例
        }
        this.mp4box = MP4Box.createFile(); // 创建新实例

        // 重新绑定事件处理函数
        this.mp4box.onReady = this.handleReady.bind(this);
        this.mp4box.onSamples = this.handleSamples.bind(this);
        await this.fetchVideo(videoUrl);
        await this.fetchVideoUtilData(gzipUrl);
    }

    async fetchVideo(url) {
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        buffer.fileStart = 0;
        this.mp4box.appendBuffer(buffer);
        this.mp4box.flush();
    }

    async fetchVideoUtilData(gzipUrl) {
        // 从服务器加载 Gzip 压缩的 JSON 文件
        const response = await fetch(gzipUrl);
        const compressedData = await response.arrayBuffer();
        const decompressedData = pako.inflate(new Uint8Array(compressedData), { to: 'string' });
        this.combinedData = JSON.parse(decompressedData);
    }


    handleReady(info) {
        this.videoTrack = info.videoTracks[0];
        if (!this.videoTrack) return;

        this.mp4box.setExtractionOptions(this.videoTrack.id, 'video', { nbSamples: 100 });
        const { track_width: videoW, track_height: videoH } = this.videoTrack;

        // 假设canvas_video已在外部定义
        canvas_video.width = videoW;
        canvas_video.height = videoH;

        canvasEl.width = videoW;
        canvasEl.height = videoH;

        this.offscreenCanvas = new OffscreenCanvas(videoW, videoH);
        this.offscreenCtx = this.offscreenCanvas.getContext('2d', { alpha: false }); // 禁用 Alpha


        this.videoDecoder = new VideoDecoder({
            output: this.handleVideoFrame.bind(this),
            error: (e) => console.error("VideoDecoder error:", e)
        });

        this.nbSampleTotal = this.videoTrack.nb_samples;
        this.videoDecoder.configure({
            codec: this.videoTrack.codec,
            codedWidth: videoW,
            codedHeight: videoH,
            description: this.getExtradata()
        });

        this.mp4box.start();
    }

    handleVideoFrame(videoFrame) {
        if (tag_ios17)
        {
            createImageBitmap(videoFrame).then(img => {
                this.offscreenCtx.fillStyle = 'white';
                this.offscreenCtx.fillRect(0, 0, this.offscreenCanvas.width, this.offscreenCanvas.height);
                this.offscreenCtx.drawImage(img, 0, 0);
                return this.offscreenCanvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
            }).then(blob => {
                const sizeInMB = (blob.size / (1024 * 1024)).toFixed(2); // 保留两位小数
                console.log('Blob size:', `${sizeInMB} MB`);
                this.videoFrames.push({
                    blob,
                    duration: videoFrame.duration,
                    timestamp: videoFrame.timestamp
                });

                videoFrame.close();

                // 添加逆序帧逻辑
                if (this.videoFrames.length === this.nbSampleTotal && !this.isReverseAdded) {

                    this.sortVideoFrames();
                    this.videoFrames.push(...[...this.videoFrames].reverse());
                    this.isReverseAdded = true;
                    console.log(`Total frames: ${this.videoFrames.length}`);
                }
            });
        }
        else
        {
            createImageBitmap(videoFrame).then(img => {
            this.videoFrames.push({
                img,
                duration: videoFrame.duration,
                timestamp: videoFrame.timestamp
            });
            ctxEl.clearRect(0, 0, canvasEl.width, canvasEl.height);
            ctxEl.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
            videoFrame.close();

            // 添加逆序帧逻辑
            if (this.videoFrames.length === this.nbSampleTotal && !this.isReverseAdded) {

                this.sortVideoFrames();
                this.videoFrames.push(...[...this.videoFrames].reverse());
                this.isReverseAdded = true;
                console.log(`Total frames: ${this.videoFrames.length}`);
            }
        });
        }
    }

    handleSamples(trackId, ref, samples) {
        if (trackId !== this.videoTrack?.id) return;

        this.countSample += samples.length;
        for (const sample of samples) {
            const chunk = new EncodedVideoChunk({
                type: sample.is_sync ? "key" : "delta",
                timestamp: sample.cts,
                duration: sample.duration,
                data: sample.data
            });
            this.videoDecoder.decode(chunk);
        }

        if (this.countSample >= this.nbSampleTotal) {
            this.videoDecoder.flush();
        }
    }

    getExtradata() {
        const trak = this.mp4box.moov.traks[0];
        const entry = trak.mdia.minf.stbl.stsd.entries[0];
        const box = entry.avcC || entry.hvcC || entry.vpcC;
        if (!box) return;

        const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
        box.write(stream);
        return new Uint8Array(stream.buffer.slice(8));
    }
    sortVideoFrames() {
        this.videoFrames.sort((a, b) => a.timestamp - b.timestamp);
    }
}

let asset_dir = "assets";
let isPaused = false; // 标志位，控制是否暂停处理
// 获取 characterDropdown 元素
const characterDropdown = document.getElementById('characterDropdown');

// 检查元素是否存在
if (characterDropdown) {
    characterDropdown.addEventListener('change', async function() {
        isPaused = true;
        document.getElementById('startMessage').style.display = 'block';
        asset_dir = this.value;
        console.log('Selected character:', asset_dir);
        await videoProcessor.init(asset_dir + "/01.mp4", asset_dir + "/combined_data.json.gz");
        await loadCombinedData();
        await setupVertsBuffers();
        isPaused = false;
        // 启动绘制循环
        await processVideoFrames();
    });
} else {
    console.warn("characterDropdown 元素未找到，无法绑定事件监听器");
}
// 初始化处理器
const videoProcessor = new VideoProcessor();

let frameIndex = 0;
const frameInterval = 40;
let lastFrameTime = performance.now();

// 原始webgl渲染
const canvas_gl = document.getElementById('canvas_gl');
const gl = canvas_gl.getContext('webgl2', { antialias: false });

// 最终显示的画布
const canvas_video = document.getElementById('canvas_video');
const ctx_video = canvas_video.getContext('2d');

// 缩放到128x128
const resizedCanvas = document.createElement('canvas');
const resizedCtx = resizedCanvas.getContext('2d', { willReadFrequently: true });
resizedCanvas.width = 128;
resizedCanvas.height = 128;

// 创建一个像素缓冲区来存储读取的像素数据
const pixels_fbo = new Uint8Array(128 * 128 * 4);

let objData;
let dataSets = [];

let program;
let indexBuffer;
let positionBuffer;
const texture_bs = gl.createTexture();
var bs_array = new Float32Array(12);

let currentDataSetIndex;
let lastDataSetIndex = -1;

let imageDataPtr = null;
let imageDataGlPtr = null;
let bsPtr = null;

// 解析OBJ文件
function parseObjFile(text) {
    const vertices = [];
    const vt = [];
    const faces = [];
    const lines = text.split('\n');

    lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts[0] === 'v') {
            vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]),
                parseFloat(parts[4]), parseFloat(parts[5]));
        } else if (parts[0] === 'f') {
            const face = parts.slice(1).map(part => {
                const indices = part.split('/').map(index => parseInt(index, 10) - 1);
                return indices[0];
            });
            faces.push(...face);
        }
    });

    return { vertices, faces };
}

async function loadCombinedData() {
    try {
        let { json_data, ...WasmInputJson } = videoProcessor.combinedData;

        let jsonString = JSON.stringify(WasmInputJson);
        // 分配内存
        // 使用 TextEncoder 计算 UTF-8 字节长度
        function getUTF8Length(str) {
            const encoder = new TextEncoder();
            const encoded = encoder.encode(str);
            return encoded.length + 1; // +1 是为了包含 null 终止符
        }
        let lengthBytes = getUTF8Length(jsonString);

        let stringPointer = Module._malloc(lengthBytes);
        Module.stringToUTF8(jsonString, stringPointer, lengthBytes);
//        Module["asm"]["stringToUTF8"](jsonString, stringPointer, lengthBytes);
        console.log("Module._processJson");
        console.log(jsonString);
        console.log(lengthBytes);
        Module._processJson(stringPointer);

        // 释放内存
        Module._free(stringPointer);

        // 提取 jsonData
        dataSets = videoProcessor.combinedData.json_data;
        console.log('JSON data loaded successfully:', dataSets.length, 'sets.');

        // 将 dataSets 的内容逆序并加到原列表后面
        dataSets = dataSets.concat(dataSets.slice().reverse());
        console.log('DataSets after adding reversed content:', dataSets.length, 'sets.');

        // 提取 objData
        objData = parseObjFile(videoProcessor.combinedData.face3D_obj.join('\n'));
        console.log('OBJ data loaded successfully:', objData.vertices.length, 'vertices,', objData.faces.length, 'faces.');
    } catch (error) {
        console.error('Error loading the combined data:', error);
        throw error;
    }
}


async function init_gl() {
        // WebGL Shaders
        const vertexShaderSource = `#version 300 es
            layout(location = 0) in vec3 a_position;
            layout(location = 1) in vec2 a_texture;
            uniform float bsVec[12];
            uniform mat4 gProjection;
            uniform mat4 gWorld0;
            uniform sampler2D texture_bs;
            uniform vec2 vertBuffer[209];
            out vec2 v_texture;
            out vec2 v_bias;

            vec4 calculateMorphPosition(vec3 position, vec2 textureCoord) {
                vec4 tmp_Position2 = vec4(position, 1.0);
                if (textureCoord.x < 3.0 && textureCoord.x >= 0.0) {
                    vec3 morphSum = vec3(0.0);
                    for (int i = 0; i < 6; i++) {
                        ivec2 coord = ivec2(int(textureCoord.y), i);
                        vec3 morph = texelFetch(texture_bs, coord, 0).xyz * 2.0 - 1.0;
                        morphSum += bsVec[i] * morph;
                    }
                    tmp_Position2.xyz += morphSum;
                }
                else if (textureCoord.x == 4.0) {
                    // lower teeth
                    vec3 morphSum = vec3(0.0, (bsVec[0] + bsVec[1]) / 2.7 + 6.0, 0.0);
                    tmp_Position2.xyz += morphSum;
                }
                return tmp_Position2;
            }

            void main() {
                mat4 gWorld = gWorld0;

                vec4 tmp_Position2 = calculateMorphPosition(a_position, a_texture);
                vec4 tmp_Position = gWorld * tmp_Position2;

                v_bias = vec2(0.0, 0.0);
                if (a_texture.x == -1.0f) {
                    v_bias = vec2(0.0, 0.0);
                }
                else if (a_texture.y < 209.0f) {
                    vec4 vert_new = gProjection * vec4(tmp_Position.x, tmp_Position.y, tmp_Position.z, 1.0);
                    v_bias = vert_new.xy - (vertBuffer[int(a_texture.y)].xy / 128.0 * 2.0 - 1.0);
                }

                if (a_texture.x >= 3.0f) {
                    gl_Position = gProjection * vec4(tmp_Position.x, tmp_Position.y, 500.0, 1.0);
                }
                else {
                    gl_Position = gProjection * vec4(tmp_Position.x, tmp_Position.y, tmp_Position.z, 1.0);
                }

                v_texture = a_texture;
            }
        `;

        const fragmentShaderSource = `#version 300 es
            precision mediump float;
            in mediump vec2 v_texture;
            in mediump vec2 v_bias;
            out vec4 out_color;

            void main() {
                if (v_texture.x == 2.0f) {
                    out_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
                else if (v_texture.x > 2.0f && v_texture.x < 2.1f) {
                    out_color = vec4(0.5f, 0.0, 0.0, 1.0);
                }
                else if (v_texture.x == 3.0f) {
                    out_color = vec4(0.0, 1.0, 0.0, 1.0);
                }
                else if (v_texture.x == 4.0f) {
                    out_color = vec4(0.0, 0.0, 1.0, 1.0);
                }
                else if (v_texture.x > 3.0f && v_texture.x < 4.0f) {
                    out_color = vec4(0.0, 0.0, 0.0, 1.0);
                }
                else {
                    vec2 wrap = (v_bias.xy + 1.0) / 2.0;
                    out_color = vec4(wrap.xy, 0.5, 1.0);
                }
            }
        `;

        // Compile shaders and link program
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexShaderSource);
        gl.compileShader(vertexShader);

        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentShaderSource);
        gl.compileShader(fragmentShader);

        program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        gl.useProgram(program);

        // Set up vertex data
        positionBuffer = gl.createBuffer();

        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(objData.vertices), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);

        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);

        indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(objData.faces), gl.STATIC_DRAW);

        var image = new Image();
        image.onload = function () {
            gl.bindTexture(gl.TEXTURE_2D, texture_bs);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
//            gl.bindTexture(gl.TEXTURE_2D, null);

            gl.activeTexture(gl.TEXTURE0);
            gl.uniform1i(gl.getUniformLocation(program, 'texture_bs'), 0);
        };
        image.src = 'common/bs_texture_halfFace.png';
}
async function setupVertsBuffers() {
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(objData.vertices), gl.STATIC_DRAW);
}

async function newVideoTask() {
    await videoProcessor.init("assets/01.mp4", "assets/combined_data.json.gz");
    // 加载 combined_data.json.gz
    await loadCombinedData();
    await init_gl();
    await setupVertsBuffers();
    initMemory();
    // 启动绘制循环
    await processVideoFrames();
    document.getElementById('startMessage').style.display = 'none';
}

function cerateOrthoMatrix()
{
    const orthoMatrix = new Float32Array(16);

// 定义正交投影参数
const left = 0;
const right = 128;
const bottom = 0;
const top = 128;
const near = 1000;
const far = -1000;

// 计算各轴跨度
const rl = right - left;    // 128
const tb = top - bottom;    // 128
const fn = far - near;      // -2000

// 列主序填充正交投影矩阵
// 第一列 (x)
orthoMatrix[0] = 2 / rl;    // 2/128 = 0.015625
orthoMatrix[1] = 0;
orthoMatrix[2] = 0;
orthoMatrix[3] = 0;

// 第二列 (y)
orthoMatrix[4] = 0;
orthoMatrix[5] = 2 / tb;    // 2/128 = 0.015625
orthoMatrix[6] = 0;
orthoMatrix[7] = 0;

// 第三列 (z)
orthoMatrix[8] = 0;
orthoMatrix[9] = 0;
orthoMatrix[10] = -2 / fn;  // -2/-2000 = 0.001
orthoMatrix[11] = 0;

// 第四列 (平移)
orthoMatrix[12] = -(right + left) / rl;  // -128/128 = -1
orthoMatrix[13] = -(top + bottom) / tb;  // -128/128 = -1
orthoMatrix[14] = -(far + near) / fn;    // -(-1000+1000)/-2000 = 0
orthoMatrix[15] = 1;
return orthoMatrix;
}

function render(mat_world, subPoints, bsArray) {
    if (isPaused) {
        // 如果暂停，直接返回，不处理帧
        return;
    }
    gl.useProgram(program);
    const worldMatUniformLocation = gl.getUniformLocation(program, "gWorld0");
    gl.uniformMatrix4fv(worldMatUniformLocation, false, mat_world);

    gl.uniform2fv(gl.getUniformLocation(program, "vertBuffer"), subPoints);
    gl.uniform1fv(gl.getUniformLocation(program, "bsVec"), bsArray);

    const projectionUniformLocation = gl.getUniformLocation(program, "gProjection");
    const orthoMatrix = cerateOrthoMatrix();
    gl.uniformMatrix4fv(projectionUniformLocation, false, orthoMatrix);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.frontFace(gl.CW);
    gl.clearColor(0.5, 0.5, 0.5, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    gl.drawElements(gl.TRIANGLES, objData.faces.length, gl.UNSIGNED_SHORT, 0);

    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels_fbo);
}



async function processVideoFrames() {
    if (isPaused) {
        // 如果暂停，直接返回，不处理帧
        return;
    }
    // 检查视频帧是否已经解码完成
    if (videoProcessor.videoFrames.length === 0 || videoProcessor.videoFrames.length < videoProcessor.nbSampleTotal) {
        console.log('Waiting for video frames to load...', videoProcessor.videoFrames.length, videoProcessor.nbSampleTotal);
        setTimeout(processVideoFrames, 100); // 等待100毫秒后再次检查
        return;
    }
    if (frameIndex >= videoProcessor.videoFrames.length) {
        frameIndex = 0; // 重新开始
    }

    if (tag_ios17)
    {
        const { blob, duration, timestamp } = videoProcessor.videoFrames[frameIndex];
        const img = await createImageBitmap(blob);
        ctx_video.drawImage(img, 0, 0, canvas_video.width, canvas_video.height);
        img.close(); // 及时释放内存
    }
    else
    {
        const { img, duration, timestamp } = videoProcessor.videoFrames[frameIndex];
        ctx_video.drawImage(img, 0, 0, canvas_video.width, canvas_video.height);
    }

    // 计算并显示FPS
    if (fps_enabled) {
        const currentTime = performance.now();
        const deltaTime = currentTime - lastFrameTime;
        frameTimes.push(currentTime);

        // 只保留最近1秒的帧时间
        while (frameTimes.length > 0 && currentTime - frameTimes[0] > 1000) {
            frameTimes.shift();
        }

        const fps = frameTimes.length;
        ctx_video.fillStyle = 'white';
        ctx_video.font = '16px Arial';
        ctx_video.textAlign = 'right';
        ctx_video.fillText(`FPS: ${fps}`, canvas_video.width - 10, 20);
    }

    processDataSet(frameIndex);
    frameIndex++;

    const currentTime = performance.now();
    const deltaTime = currentTime - lastFrameTime;
    const delay = Math.max(0, frameInterval - deltaTime);
    lastFrameTime = currentTime + delay;
    setTimeout(processVideoFrames, delay);
}

async function initMemory() {
    const imageDataSize = 128 * 128 * 4; // RGBA
    imageDataPtr = Module._malloc(imageDataSize);
    imageDataGlPtr = Module._malloc(imageDataSize);
    bsPtr = Module._malloc(12 * 4); // 12 floats for blend shape
}
async function processDataSet(currentDataSetIndex) {
    if (isPaused) {
        // 如果暂停，直接返回，不处理帧
        return;
    }
    const dataSet = dataSets[currentDataSetIndex];
    const rect = dataSet.rect;

    const currentpoints = dataSets[currentDataSetIndex].points;

    const matrix = new Float32Array(16);
    matrix.set(currentpoints.slice(0, 16));

    const subPoints = currentpoints.slice(16);
    Module._updateBlendShape(bsPtr, 12 * 4);
    const bsArray = new Float32Array(Module.HEAPU8.buffer, bsPtr, 12);

    render(matrix, subPoints, bsArray);
    // console.log("bsArray", bsArray);
    resizedCtx.drawImage(canvas_video, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], 0, 0, 128, 128);

    const imageData = resizedCtx.getImageData(0, 0, 128, 128);
    Module.HEAPU8.set(imageData.data, imageDataPtr);

    Module.HEAPU8.set(pixels_fbo, imageDataGlPtr);

    Module._processImage(imageDataPtr, 128, 128, imageDataGlPtr, 128, 128);
    const result = Module.HEAPU8.subarray(imageDataPtr, imageDataPtr + imageData.data.length);
    imageData.data.set(result);

    resizedCtx.putImageData(imageData, 0, 0);
    ctx_video.drawImage(resizedCanvas, 0, 0, 128, 128, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
}
