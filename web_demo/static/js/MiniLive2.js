// ==================== 重要配置项 ====================
let CONFIG = {
    showFPS: false,              // 是否显示 FPS
    chromaKeyEnabled: true,     // 是否开启绿幕扣除
    backgroundVideoSrc: "background/bg.mp4",  // 背景视频路径（开启绿幕扣除时使用）
    videoSrc: "assets/01.mp4",                // 默认视频文件路径
    dataSrc: "assets/combined_data.json.gz",   // 默认数据文件路径
    // 绿幕抠图参数配置
    chromaKey: {
        keyColor: { r: 0.0, g: 1.0, b: 0.0 },  // 要抠除的颜色（默认绿色）
        similarity: 0.4,      // 相似度阈值：值越小，抠除范围越严格
        smoothness: 0.1,      // 平滑度：值越大，边缘过渡越平滑
        spill: 0.5           // 溢色抑制：减少绿色溢出到保留区域
    }
};
// ==================================================
const model_size = 184;
let frameTimes = [];
const VIDEO_FPS = 25;
const FRAME_INTERVAL = 1000 / VIDEO_FPS;
const MODULO_N = 16;
const THRESHOLD = 128;

let chromaKeyCanvas = null;
let chromaKeyGl = null;
let chromaKeyProgram = null;
let chromaKeyTextures = { foreground: null };

let frameIndexCanvas = null;
let frameIndexCtx = null;

class VideoProcessor {
    constructor() {
        this.video = null;
        this.combinedData = null;
        this.lastFrameTime = 0;
    }

    async init(videoUrl, gzipUrl) {
        if (this.video) {
            this.video.pause();
            this.video.src = '';
            this.video = null;
        }

        this.video = document.createElement('video');
        this.video.src = videoUrl;
        this.video.loop = true;
        this.video.muted = true;
        this.video.playsInline = true;

        await new Promise((resolve, reject) => {
            this.video.onloadedmetadata = () => {
                const videoW = this.video.videoWidth;
                const videoH = this.video.videoHeight;
                
                canvas_video.width = videoW;
                canvas_video.height = videoH;
                canvasEl.width = videoW;
                canvasEl.height = videoH;
                
                resolve();
            };
            this.video.onerror = reject;
        });

        await this.fetchVideoUtilData(gzipUrl);
    }

    async fetchVideoUtilData(gzipUrl) {
        const response = await fetch(gzipUrl);
        const compressedData = await response.arrayBuffer();
        const decompressedData = pako.inflate(new Uint8Array(compressedData), { to: 'string' });
        this.combinedData = JSON.parse(decompressedData);
    }

    decodeModuloFromPixels(pixelData) {
        let detected = 0;

        for (let i = 0; i < 4; i++) {
            let r = pixelData[i * 4];
            let bitValue = (r > THRESHOLD) ? 1 : 0;

            switch(i) {
                case 0:
                    if (bitValue) detected |= (1 << 1);
                    break;
                case 1:
                    if (bitValue) detected |= (1 << 0);
                    break;
                case 2:
                    if (bitValue) detected |= (1 << 3);
                    break;
                case 3:
                    if (bitValue) detected |= (1 << 2);
                    break;
            }
        }

        return detected;
    }

    getAccurateFrameIndex(pixelData) {
        return this.decodeModuloFromPixels(pixelData);
    }

    findRealFrame(roughFrame, exactModulo) {
        let candidateFrame = roughFrame;
        if (candidateFrame % MODULO_N !== exactModulo) {
            for (let offset = -7; offset <= 7; offset++) {
                let candidate = candidateFrame + offset;
                if (candidate >= 0 && candidate % MODULO_N === exactModulo) {
                    return candidate;
                }
            }
            console.warn("未找到匹配帧号，返回修正值:", candidateFrame);
            return candidateFrame;
        }
        return candidateFrame;
    }

    getCurrentFrameIndex(pixelData) {
        if (!this.video) return 0;
        
        const roughFrameByTime = Math.floor(this.video.currentTime * VIDEO_FPS);
        const moduloFromPixel = this.getAccurateFrameIndex(pixelData);
        // console.log("roughFrameByTime", roughFrameByTime, "moduloFromPixel", moduloFromPixel);
        return this.findRealFrame(roughFrameByTime, moduloFromPixel);
    }

    play() {
        if (this.video) {
            this.video.play();
        }
    }

    pause() {
        if (this.video) {
            this.video.pause();
        }
    }
}

function setupBackgroundVideo() {
    const bgVideo = document.getElementById('background_video');
    if (bgVideo) {
        bgVideo.src = CONFIG.backgroundVideoSrc;
        bgVideo.load();
        bgVideo.style.display = 'block';
        bgVideo.play().catch(e => console.log('背景视频自动播放被阻止:', e));
    }
}

function hideBackgroundVideo() {
    const bgVideo = document.getElementById('background_video');
    if (bgVideo) {
        bgVideo.pause();
        bgVideo.style.display = 'none';
    }
}

function initChromaKeyGL() {
    chromaKeyCanvas = document.createElement('canvas');
    chromaKeyCanvas.width = canvas_video.width;
    chromaKeyCanvas.height = canvas_video.height;
    
    console.log('初始化绿幕抠图 WebGL, 画布尺寸:', chromaKeyCanvas.width, 'x', chromaKeyCanvas.height);
    console.log('绿幕抠图参数:', CONFIG.chromaKey);
    
    chromaKeyGl = chromaKeyCanvas.getContext('webgl2', { 
        antialias: false, 
        alpha: true,
        premultipliedAlpha: false 
    });
    
    const vertexShaderSource = `#version 300 es
        in vec2 a_position;
        in vec2 a_texCoord;
        out vec2 v_texCoord;
        
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texCoord = a_texCoord;
        }
    `;
    
    const fragmentShaderSource = `#version 300 es
        precision highp float;
        
        in vec2 v_texCoord;
        uniform sampler2D u_foreground;
        uniform vec3 u_keyColor;
        uniform float u_similarity;
        uniform float u_smoothness;
        uniform float u_spill;
        out vec4 outColor;
        
        void main() {
            vec4 fg = texture(u_foreground, v_texCoord);
            
            vec3 keyColor = u_keyColor;
            
            float diff = distance(fg.rgb, keyColor);
            float mask = smoothstep(u_similarity, u_similarity + u_smoothness, diff);
            
            float spill = max(0.0, fg.g - max(fg.r, fg.b)) * u_spill;
            vec3 color = fg.rgb - spill * keyColor;
            
            float edgeMask = smoothstep(u_similarity * 0.5, u_similarity, diff);
            color = mix(color, fg.rgb, edgeMask * 0.5);
            
            outColor = vec4(color, mask);
        }
    `;
    
    const vertexShader = chromaKeyGl.createShader(chromaKeyGl.VERTEX_SHADER);
    chromaKeyGl.shaderSource(vertexShader, vertexShaderSource);
    chromaKeyGl.compileShader(vertexShader);
    if (!chromaKeyGl.getShaderParameter(vertexShader, chromaKeyGl.COMPILE_STATUS)) {
        console.error('顶点着色器编译失败:', chromaKeyGl.getShaderInfoLog(vertexShader));
    }
    
    const fragmentShader = chromaKeyGl.createShader(chromaKeyGl.FRAGMENT_SHADER);
    chromaKeyGl.shaderSource(fragmentShader, fragmentShaderSource);
    chromaKeyGl.compileShader(fragmentShader);
    if (!chromaKeyGl.getShaderParameter(fragmentShader, chromaKeyGl.COMPILE_STATUS)) {
        console.error('片段着色器编译失败:', chromaKeyGl.getShaderInfoLog(fragmentShader));
    }
    
    chromaKeyProgram = chromaKeyGl.createProgram();
    chromaKeyGl.attachShader(chromaKeyProgram, vertexShader);
    chromaKeyGl.attachShader(chromaKeyProgram, fragmentShader);
    chromaKeyGl.linkProgram(chromaKeyProgram);
    if (!chromaKeyGl.getProgramParameter(chromaKeyProgram, chromaKeyGl.LINK_STATUS)) {
        console.error('着色器程序链接失败:', chromaKeyGl.getProgramInfoLog(chromaKeyProgram));
    } else {
        console.log('绿幕抠图着色器初始化成功');
    }
    
    const positions = new Float32Array([
        -1, -1,  0, 1,
         1, -1,  1, 1,
        -1,  1,  0, 0,
         1,  1,  1, 0,
    ]);
    
    const buffer = chromaKeyGl.createBuffer();
    chromaKeyGl.bindBuffer(chromaKeyGl.ARRAY_BUFFER, buffer);
    chromaKeyGl.bufferData(chromaKeyGl.ARRAY_BUFFER, positions, chromaKeyGl.STATIC_DRAW);
    
    const posLoc = chromaKeyGl.getAttribLocation(chromaKeyProgram, 'a_position');
    const texLoc = chromaKeyGl.getAttribLocation(chromaKeyProgram, 'a_texCoord');
    
    chromaKeyGl.enableVertexAttribArray(posLoc);
    chromaKeyGl.vertexAttribPointer(posLoc, 2, chromaKeyGl.FLOAT, false, 16, 0);
    chromaKeyGl.enableVertexAttribArray(texLoc);
    chromaKeyGl.vertexAttribPointer(texLoc, 2, chromaKeyGl.FLOAT, false, 16, 8);
    
    chromaKeyTextures.foreground = chromaKeyGl.createTexture();
    
    chromaKeyGl.bindTexture(chromaKeyGl.TEXTURE_2D, chromaKeyTextures.foreground);
    chromaKeyGl.texParameteri(chromaKeyGl.TEXTURE_2D, chromaKeyGl.TEXTURE_WRAP_S, chromaKeyGl.CLAMP_TO_EDGE);
    chromaKeyGl.texParameteri(chromaKeyGl.TEXTURE_2D, chromaKeyGl.TEXTURE_WRAP_T, chromaKeyGl.CLAMP_TO_EDGE);
    chromaKeyGl.texParameteri(chromaKeyGl.TEXTURE_2D, chromaKeyGl.TEXTURE_MIN_FILTER, chromaKeyGl.LINEAR);
    chromaKeyGl.texParameteri(chromaKeyGl.TEXTURE_2D, chromaKeyGl.TEXTURE_MAG_FILTER, chromaKeyGl.LINEAR);
}

function processChromaKey(foregroundSource) {
    if (!chromaKeyGl || !chromaKeyProgram) {
        return foregroundSource;
    }
    
    chromaKeyGl.viewport(0, 0, chromaKeyCanvas.width, chromaKeyCanvas.height);
    chromaKeyGl.clearColor(0, 0, 0, 0);
    chromaKeyGl.clear(chromaKeyGl.COLOR_BUFFER_BIT);
    
    chromaKeyGl.useProgram(chromaKeyProgram);
    
    chromaKeyGl.activeTexture(chromaKeyGl.TEXTURE0);
    chromaKeyGl.bindTexture(chromaKeyGl.TEXTURE_2D, chromaKeyTextures.foreground);
    chromaKeyGl.texImage2D(chromaKeyGl.TEXTURE_2D, 0, chromaKeyGl.RGBA, chromaKeyGl.RGBA, chromaKeyGl.UNSIGNED_BYTE, foregroundSource);
    
    chromaKeyGl.uniform1i(chromaKeyGl.getUniformLocation(chromaKeyProgram, 'u_foreground'), 0);
    
    // 使用 CONFIG 中的绿幕参数
    chromaKeyGl.uniform3f(
        chromaKeyGl.getUniformLocation(chromaKeyProgram, 'u_keyColor'), 
        CONFIG.chromaKey.keyColor.r, 
        CONFIG.chromaKey.keyColor.g, 
        CONFIG.chromaKey.keyColor.b
    );
    chromaKeyGl.uniform1f(
        chromaKeyGl.getUniformLocation(chromaKeyProgram, 'u_similarity'), 
        CONFIG.chromaKey.similarity
    );
    chromaKeyGl.uniform1f(
        chromaKeyGl.getUniformLocation(chromaKeyProgram, 'u_smoothness'), 
        CONFIG.chromaKey.smoothness
    );
    chromaKeyGl.uniform1f(
        chromaKeyGl.getUniformLocation(chromaKeyProgram, 'u_spill'), 
        CONFIG.chromaKey.spill
    );
    
    chromaKeyGl.drawArrays(chromaKeyGl.TRIANGLE_STRIP, 0, 4);
    
    return chromaKeyCanvas;
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
        
        if (chromaKeyCanvas) {
            chromaKeyCanvas.width = canvas_video.width;
            chromaKeyCanvas.height = canvas_video.height;
        }
        
        isPaused = false;
        videoProcessor.play();
        processVideoFrames();
    });
} else {
    console.warn("characterDropdown 元素未找到，无法绑定事件监听器");
}
// 初始化处理器
const videoProcessor = new VideoProcessor();

const canvas_gl = document.getElementById('canvas_gl');
const gl = canvas_gl.getContext('webgl2', { antialias: false });

// 最终显示的画布
const canvas_video = document.getElementById('canvas_video');
// const ctx_video = canvas_video.getContext('2d');
const ctx_video = canvas_video.getContext('2d', { 
    alpha: true,
    willReadFrequently: false 
});

// 缩放到model_size
const resizedCanvas = document.createElement('canvas');
// const resizedCtx = resizedCanvas.getContext('2d', { willReadFrequently: true });
const resizedCtx = resizedCanvas.getContext('2d', { 
    alpha: true,
    willReadFrequently: true 
});
resizedCanvas.width = model_size;
resizedCanvas.height = model_size;

// 创建一个像素缓冲区来存储读取的像素数据
const pixels_fbo = new Uint8Array(model_size * model_size * 4);

// 预创建离屏 canvas，用于锁定当前视频帧，避免处理过程中视频继续播放导致不同步
const lockedFrameCanvas = document.createElement('canvas');
const lockedFrameCtx = lockedFrameCanvas.getContext('2d', { willReadFrequently: true });

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
        const version_valid = Module._processJson(stringPointer);

        // 释放内存
        Module._free(stringPointer);

        if (version_valid == 0) {
            alert("DH_live前端版本不对，请检查")
        }

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
                    float z_ = (bsVec[0] + bsVec[1])/ 3.9;
                    z_ = max(z_, 0.0);
                    vec3 morphSum = vec3(0.0, (bsVec[0] + bsVec[1]) / 3.0 + 6.0, z_);
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
                    v_bias = vert_new.xy - (vertBuffer[int(a_texture.y)].xy / 184.0 * 2.0 - 1.0);
                }
                gl_Position = gProjection * vec4(tmp_Position.xyz, 1.0);
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
                else if (v_texture.x == -2.0f) {
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
    await videoProcessor.init(CONFIG.videoSrc, CONFIG.dataSrc);
    await loadCombinedData();
    await init_gl();
    await setupVertsBuffers();
    initMemory();
    
    if (CONFIG.chromaKeyEnabled) {
        initChromaKeyGL();
        setupBackgroundVideo();
    } else {
        hideBackgroundVideo();
    }
    
    videoProcessor.play();
    processVideoFrames();
    document.getElementById('startMessage').style.display = 'none';
}

function cerateOrthoMatrix()
{
    const orthoMatrix = new Float32Array(16);

// 定义正交投影参数
const left = 0;
const right = model_size;
const bottom = 0;
const top = model_size;
const near = 1000;
const far = -1000;

// 计算各轴跨度
const rl = right - left;
const tb = top - bottom;
const fn = far - near;

// 列主序填充正交投影矩阵
// 第一列 (x)
orthoMatrix[0] = 2 / rl;
orthoMatrix[1] = 0;
orthoMatrix[2] = 0;
orthoMatrix[3] = 0;

// 第二列 (y)
orthoMatrix[4] = 0;
orthoMatrix[5] = 2 / tb;
orthoMatrix[6] = 0;
orthoMatrix[7] = 0;

// 第三列 (z)
orthoMatrix[8] = 0;
orthoMatrix[9] = 0;
orthoMatrix[10] = -2 / fn;
orthoMatrix[11] = 0;

// 第四列 (平移)
orthoMatrix[12] = -(right + left) / rl;
orthoMatrix[13] = -(top + bottom) / tb;
orthoMatrix[14] = -(far + near) / fn;
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
    // gl.enable(gl.BLEND);
    // gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
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
        return;
    }

    const currentTime = performance.now();
    const deltaTime = currentTime - videoProcessor.lastFrameTime;

    if (deltaTime >= FRAME_INTERVAL) {
        videoProcessor.lastFrameTime = currentTime - (deltaTime % FRAME_INTERVAL);

        // 一次性将当前视频帧绘制到离屏 canvas，锁定当前帧，避免后续处理中视频继续播放导致不同步
        if (lockedFrameCanvas.width !== canvas_video.width || 
            lockedFrameCanvas.height !== canvas_video.height) {
            lockedFrameCanvas.width = canvas_video.width;
            lockedFrameCanvas.height = canvas_video.height;
        }
        lockedFrameCtx.drawImage(videoProcessor.video, 0, 0, canvas_video.width, canvas_video.height);

        if (!frameIndexCanvas) {
            frameIndexCanvas = document.createElement('canvas');
            frameIndexCanvas.width = 2;
            frameIndexCanvas.height = 2;
            frameIndexCtx = frameIndexCanvas.getContext('2d', { willReadFrequently: true });
        }
        
        // 从锁定的帧 canvas 读取帧序号像素
        frameIndexCtx.clearRect(0, 0, 2, 2);
        frameIndexCtx.drawImage(lockedFrameCanvas, 
            canvas_video.width - 2, 0, 2, 2,
            0, 0, 2, 2);
        const pixelData = frameIndexCtx.getImageData(0, 0, 2, 2);
        const currentFrameIndex = videoProcessor.getCurrentFrameIndex(pixelData.data);

        ctx_video.clearRect(0, 0, canvas_video.width, canvas_video.height);

        if (CONFIG.chromaKeyEnabled && chromaKeyGl) {
            const chromaKeyResult = processChromaKey(lockedFrameCanvas);
            ctx_video.drawImage(chromaKeyResult, 0, 0, canvas_video.width, canvas_video.height);
        } else {
            ctx_video.drawImage(lockedFrameCanvas, 0, 0, canvas_video.width, canvas_video.height);
        }
        // console.log("currentFrameIndex", currentFrameIndex, videoProcessor.video.currentTime);

        processDataSet(currentFrameIndex);

        if (CONFIG.showFPS) {
            frameTimes.push(currentTime);

            while (frameTimes.length > 0 && currentTime - frameTimes[0] > 1000) {
                frameTimes.shift();
            }

            const fps = frameTimes.length;
            ctx_video.fillStyle = 'white';
            ctx_video.font = '16px Arial';
            ctx_video.textAlign = 'right';
            ctx_video.fillText(`FPS: ${fps}`, canvas_video.width - 10, 20);
        }
    }

    requestAnimationFrame(processVideoFrames);
}

async function initMemory() {
    const imageDataSize = model_size * model_size * 4; // RGBA
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
    resizedCtx.clearRect(0, 0, model_size, model_size);
    resizedCtx.drawImage(canvas_video, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1], 0, 0, model_size, model_size);

    const imageData = resizedCtx.getImageData(0, 0, model_size, model_size);
    Module.HEAPU8.set(imageData.data, imageDataPtr);

    Module.HEAPU8.set(pixels_fbo, imageDataGlPtr);

    Module._processImage(imageDataPtr, model_size, model_size, imageDataGlPtr, currentDataSetIndex);
    const result = Module.HEAPU8.subarray(imageDataPtr, imageDataPtr + imageData.data.length);
    imageData.data.set(result);

    resizedCtx.putImageData(imageData, 0, 0);
    ctx_video.drawImage(resizedCanvas, 0, 0, model_size, model_size, rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
}