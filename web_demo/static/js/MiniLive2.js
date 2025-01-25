// video 读取，使用MP4Box方式读取原始视频帧，才能和cv2读取保持完全一致
const mp4url = 'assets/01.mp4';

const mp4box = MP4Box.createFile();
let videoTrack = null;
let videoDecoder = null;
const videoFrames = [];
let nbSampleTotal = 0;
let countSample = 0;

let index = 0;
let lastTime = performance.now(); // 记录上一次绘制的时间戳
const frameInterval = 40; // 每帧的时间间隔（40ms对应25fps）

const getExtradata = () => {
    const entry = mp4box.moov.traks[0].mdia.minf.stbl.stsd.entries[0];
    const box = entry.avcC ?? entry.hvcC ?? entry.vpcC;
    if (box != null) {
        const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
        box.write(stream);
        return new Uint8Array(stream.buffer.slice(8));
    }
};

mp4box.onReady = function (info) {
    videoTrack = info.videoTracks[0];

    if (videoTrack != null) {
        mp4box.setExtractionOptions(videoTrack.id, 'video');
    }

    const videoW = videoTrack.track_width;
    const videoH = videoTrack.track_height;

    canvas_video.width = videoW;
    canvas_video.height = videoH;

    videoDecoder = new VideoDecoder({
        output: (videoFrame) => {
            createImageBitmap(videoFrame).then((img) => {
                videoFrames.push({
                    img,
                    duration: videoFrame.duration,
                    timestamp: videoFrame.timestamp
                });
                videoFrame.close();
            });
        },
        error: (err) => {
            console.error('videoDecoder错误：', err);
        }
    });

    nbSampleTotal = videoTrack.nb_samples;

    videoDecoder.configure({
        codec: videoTrack.codec,
        codedWidth: videoW,
        codedHeight: videoH,
        description: getExtradata()
    });

    mp4box.start();
};

mp4box.onSamples = function (trackId, ref, samples) {
    if (videoTrack.id === trackId) {
        mp4box.stop();

        countSample += samples.length;

        for (const sample of samples) {
            console.log('Sample duration:', sample); // 检查 duration 是否正确
            const type = sample.is_sync ? 'key' : 'delta';

            const chunk = new EncodedVideoChunk({
                type,
                timestamp: sample.cts,
                duration: sample.duration,
                data: sample.data
            });

            videoDecoder.decode(chunk);
        }

        if (countSample === nbSampleTotal) {
            videoDecoder.flush();
        }
    }
};

fetch(mp4url)
    .then(res => res.arrayBuffer())
    .then(buffer => {
        buffer.fileStart = 0;
        mp4box.appendBuffer(buffer);
        mp4box.flush();
    });

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
const texture_bs = gl.createTexture();
var bs_array = new Float32Array(12);

const mat4 = glMatrix.mat4;

let currentDataSetIndex;
let lastDataSetIndex = -1;

// 监听来自 iframe 的消息
window.addEventListener('message', function (event) {
    document.getElementById('startMessage').style.display = 'none';
    if (!videoFrames.length) {
        console.error('视频解码尚未完成，请稍等');
        return;
    }

    // 启动绘制循环
    processVideoFrames();
});

async function loadRefData() {
    const response = await fetch('assets/ref_data.txt');
    if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
    }
    const fileContent = await response.text();
    var refData = fileContent.split('\n').map(line => parseFloat(line.trim()));
    const expectedSize = 18 * 14 * 20;
    refData = refData.slice(0, expectedSize);
    const refDataSize = refData.length;
    if (refDataSize !== expectedSize) {
        throw new Error(`Invalid refData size: expected ${expectedSize}, got ${refData.length}`);
    }
    console.log(refData.length, refData)
    // Convert refData to Float32Array
    const floatArray = new Float32Array(refData);
    const floatArrayBytes = floatArray.byteLength;

    var refDataPtr = Module._malloc(floatArrayBytes);
    Module.HEAPF32.set(floatArray, refDataPtr / 4);
    Module._setRefData(refDataPtr, floatArrayBytes);
//    Module._free(refDataPtr);
}

// 加载JSON数据
async function loadJsonData() {
    try {
        const response = await fetch('assets/json_data.json');
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        dataSets = await response.json();
        console.log('Data loaded successfully:', dataSets.length, 'sets.');
    } catch (error) {
        console.error('Error loading the file:', error);
    }
}

// 加载OBJ文件
async function loadObjFile(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
    }
    const text = await response.text();
    const { vertices, faces } = parseObjFile(text);
    return { vertices, faces };
}

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

async function init_gl() {
    loadJsonData();
    loadRefData();
    objData = await loadObjFile('assets/face3D.obj');

    {
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
        const positionLocation = gl.getAttribLocation(program, "a_position");
        const positionBuffer = gl.createBuffer();

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
            gl.bindTexture(gl.TEXTURE_2D, null);
        };
        image.src = 'common/bs_texture_halfFace.png';
    }
}

function render(mat_world, subPoints, bsArray) {
    gl.useProgram(program);
    const worldMatUniformLocation = gl.getUniformLocation(program, "gWorld0");
    gl.uniformMatrix4fv(worldMatUniformLocation, false, mat_world);

    gl.uniform2fv(gl.getUniformLocation(program, "vertBuffer"), subPoints);
    gl.uniform1fv(gl.getUniformLocation(program, "bsVec"), bsArray);

    const projectionUniformLocation = gl.getUniformLocation(program, "gProjection");
    const orthoMatrix = mat4.create();
    mat4.ortho(orthoMatrix, 0, 128, 0, 128, 1000, -1000);
    gl.uniformMatrix4fv(projectionUniformLocation, false, orthoMatrix);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.frontFace(gl.CW);
    gl.clearColor(0.5, 0.5, 0.5, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture_bs);
    gl.uniform1i(gl.getUniformLocation(program, 'texture_bs'), 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    gl.drawElements(gl.TRIANGLES, objData.faces.length, gl.UNSIGNED_SHORT, 0);

    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels_fbo);
}

async function processVideoFrames() {
    if (index >= videoFrames.length) {
        index = 0; // 重新开始
    }
    const { img, duration, timestamp } = videoFrames[index];
    ctx_video.drawImage(img, 0, 0, canvas_video.width, canvas_video.height);
    processDataSet(index);

    console.log("draw", index, duration, timestamp);
    index++;

    const currentTime = performance.now();
    const deltaTime = currentTime - lastTime;
    const delay = Math.max(0, frameInterval - deltaTime);
    lastTime = currentTime + delay;
    setTimeout(processVideoFrames, delay);
}

async function processDataSet(currentDataSetIndex) {
    const dataSet = dataSets[currentDataSetIndex];
    const rect = dataSet.rect;

    const currentpoints = dataSets[currentDataSetIndex].points;
    console.log("video.currentTime 1111", currentDataSetIndex);

    let points = currentpoints;

    let matrix = mat4.create();
    mat4.set(
        matrix,
        points[0], points[1], points[2], points[3],
        points[4], points[5], points[6], points[7],
        points[8], points[9], points[10], points[11],
        points[12], points[13], points[14], points[15]
    );

    const subPoints = points.slice(16);
    const bsPtr = allocateMemory(12 * 4);
    Module._updateBlendShape(bsPtr, 12 * 4);
    const bsArray = new Float32Array(Module.HEAPU8.buffer, bsPtr, 12);

    render(matrix, subPoints, bsArray);
    console.log("bsArray", bsArray);
    resizedCtx.drawImage(canvas_video, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], 0, 0, 128, 128);

    const imageData = resizedCtx.getImageData(0, 0, 128, 128);
    const imageDataPtr = allocateMemory(imageData.data.length);
    Module.HEAPU8.set(imageData.data, imageDataPtr);

    const imageDataGlPtr = allocateMemory(pixels_fbo.length);
    Module.HEAPU8.set(pixels_fbo, imageDataGlPtr);

    Module._processImage(imageDataPtr, 128, 128, imageDataGlPtr, 128, 128);
    const result = Module.HEAPU8.subarray(imageDataPtr, imageDataPtr + imageData.data.length);
    imageData.data.set(result);

    resizedCtx.putImageData(imageData, 0, 0);
    ctx_video.drawImage(resizedCanvas, 0, 0, 128, 128, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);

    freeMemory(imageDataPtr);
    freeMemory(imageDataGlPtr);
    freeMemory(bsPtr);
}

function allocateMemory(size) {
    const ptr = Module._malloc(size);
    if (ptr === 0) throw new Error('Failed to allocate memory');
    return ptr;
}

function freeMemory(ptr) {
    if (ptr !== null && ptr !== 0) {
        Module._free(ptr);
    }
}
