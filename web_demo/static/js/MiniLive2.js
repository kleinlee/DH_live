const audioSource = document.getElementById('audioSource');
const audio = document.getElementById('audio');

        const video = document.getElementById('video');
        const canvas_gl = document.getElementById('canvas_gl');
        const gl = canvas_gl.getContext('webgl2',{ antialias: false });

        const canvas_video = document.getElementById('canvas_video');
        const ctx_video = canvas_video.getContext('2d');
        
        // 缩放到128x128
        const resizedCanvas = document.createElement('canvas');
        const resizedCtx = resizedCanvas.getContext('2d', {willReadFrequently: true});

        // 创建一个像素缓冲区来存储读取的像素数据
        const pixels_fbo = new Uint8Array(128 * 128 * 4);


        let objData;
        let dataSets = [];

        let program;
        let indexBuffer;
        const texture_bs = gl.createTexture();;
        var bs_array = new Float32Array(12);

        const mat4 = glMatrix.mat4;

document.getElementById('playButton').addEventListener('click', function() {
    video.play();
//    document.getElementById('startMessage').style.display = 'none';
});

video.addEventListener('loadedmetadata', () => {
//     resizeCanvas();
    video.addEventListener('play', processVideoFrames);
//    resizeCanvas();
    console.log("loadedmetadata", video.videoWidth, video.videoHeight)
    canvas_video.width = video.videoWidth;
    canvas_video.height = video.videoHeight;
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
//    console.log(line)
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'v') {
              vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]),
              parseFloat(parts[4]), parseFloat(parts[5]));
//        vertices.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
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
    else if (textureCoord.x == 4.0)
    {
        // lower teeth
        vec3 morphSum = vec3(0.0, (bsVec[0] + bsVec[1])/ 2.7 + 6.0, 0.0);
        tmp_Position2.xyz += morphSum;
    }
    return tmp_Position2;
}

            void main() {
                mat4 gWorld = gWorld0;

    vec4 tmp_Position2 = calculateMorphPosition(a_position, a_texture);
    vec4 tmp_Position = gWorld * tmp_Position2;

    v_bias = vec2(0.0, 0.0);
    if (a_texture.x == -1.0f)
    {
        v_bias = vec2(0.0, 0.0);
    }
    else if (a_texture.y < 209.0f)
    {
        vec4 vert_new = gProjection * vec4(tmp_Position.x, tmp_Position.y, tmp_Position.z, 1.0);
        v_bias = vert_new.xy - (vertBuffer[int(a_texture.y)].xy/128.0 * 2.0 - 1.0);
    }

    if (a_texture.x >= 3.0f)
    {
        gl_Position = gProjection * vec4(tmp_Position.x, tmp_Position.y, 500.0, 1.0);
    }
    else
    {
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
void main()
{
    if (v_texture.x == 2.0f)
    {
        out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else if (v_texture.x > 2.0f && v_texture.x < 2.1f)
    {
        out_color = vec4(0.5f, 0.0, 0.0, 1.0);
    }
    else if (v_texture.x == 3.0f)
    {
        out_color = vec4(0.0, 1.0, 0.0, 1.0);
    }
    else if (v_texture.x == 4.0f)
    {
        out_color = vec4(0.0, 0.0, 1.0, 1.0);
    }
    else if (v_texture.x > 3.0f && v_texture.x < 4.0f)
    {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
        vec2 wrap = (v_bias.xy + 1.0)/2.0;
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
        // 启用顶点属性数组
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);

        // 启用纹理坐标属性数组
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
//        gl.enableVertexAttribArray(positionLocation);
//        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);


        indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(objData.faces), gl.STATIC_DRAW);


        // 假设已经有了WebGL上下文gl

var image = new Image();
image.onload = function() {
            gl.bindTexture(gl.TEXTURE_2D, texture_bs);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.bindTexture(gl.TEXTURE_2D, null);
};
image.src = 'assets/bs_texture_halfFace.png';

    }
}
//init();



        function render(mat_world, subPoints, bsArray) {
//            console.log("render")
            gl.useProgram(program);
            const worldMatUniformLocation = gl.getUniformLocation(program, "gWorld0");
            gl.uniformMatrix4fv(worldMatUniformLocation, false, mat_world);

            gl.uniform2fv(gl.getUniformLocation(program, "vertBuffer"), subPoints);
            gl.uniform1fv(gl.getUniformLocation(program, "bsVec"), bsArray);

            // 获取着色器程序中uniform变量的位置
            const projectionUniformLocation = gl.getUniformLocation(program, "gProjection");

            // 使用 gl-matrix 创建正交投影矩阵
            const orthoMatrix = mat4.create();

            mat4.ortho(orthoMatrix, 0, 128, 0, 128, 1000, -1000);
            // console.log(orthoMatrix)
            // 将正交投影矩阵传递给着色器
            gl.uniformMatrix4fv(projectionUniformLocation, false, orthoMatrix);

            gl.enable(gl.DEPTH_TEST);
            // 启用混合
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            // 启用面剔除
            gl.enable(gl.CULL_FACE);
            gl.cullFace(gl.BACK);  // 剔除背面
            gl.frontFace(gl.CW);   // 通常顶点顺序是顺时针
            // 设置清屏颜色
            gl.clearColor(0.5, 0.5, 0.5, 0);
            // 清屏
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture_bs);
            gl.uniform1i(gl.getUniformLocation(program, 'texture_bs'), 0);

            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);

            // Render to canvas
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);

            const width = gl.drawingBufferWidth;
            const height = gl.drawingBufferHeight;
            gl.drawElements(gl.TRIANGLES, objData.faces.length, gl.UNSIGNED_SHORT, 0);

            // 读取像素数据
            gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels_fbo);
        }

function resizeCanvas() {
            const screenWidth = window.innerWidth;
            const screenHeight = window.innerHeight;

            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;
            const videoAspectRatio = videoWidth / videoHeight;

            let canvasWidth = screenWidth;
            let canvasHeight = screenWidth / videoAspectRatio;

            if (canvasHeight > screenHeight) {
                canvasHeight = screenHeight;
                canvasWidth = screenHeight * videoAspectRatio;
            }

            canvas_video.width = canvasWidth;
            canvas_video.height = canvasHeight;

            // Center the canvas
            canvas_video.style.position = 'absolute';
            canvas_video.style.left = (screenWidth - canvasWidth) / 2 + 'px';
            canvas_video.style.top = (screenHeight - canvasHeight) / 2 + 'px';
        }



async function processVideoFrames() {
    let lastDataSetIndex = -1; // 初始化为一个不可能的索引值
    let isProcessing = false; // 标志位

    let lastVideoTime = 0; // 初始化为一个不可能的索引值

    const frameCallback = async (currentTime) => {
      if (!video.paused && !video.ended && !isProcessing) {
        isProcessing = true;

        try {
          // 计算当前数据集索引
          const currentDataSetIndex = Math.floor(video.currentTime * 25);

//          console.log("currentDataSetIndex", currentDataSetIndex, video.currentTime, video.currentTime - lastVideoTime)
          lastVideoTime = video.currentTime
          if (lastDataSetIndex !== currentDataSetIndex && currentDataSetIndex < dataSets.length - 1) {
            lastDataSetIndex = currentDataSetIndex;
//            console.log("currentDataSetIndex", currentDataSetIndex, lastDataSetIndex, video.currentTime, video.currentTime - lastVideoTime)
            // 清除画布并绘制当前视频帧到canvas
            ctx_video.clearRect(0, 0, canvas_video.width, canvas_video.height);
            ctx_video.drawImage(video, 0, 0, canvas_video.width, canvas_video.height);
            // 处理当前数据集
            if (currentDataSetIndex < dataSets.length - 1) {

                // 创建一个ArrayBuffer来存储浮点数
                const floatArraySize = 12;
                const floatArrayBytes = floatArraySize * 4; // 每个浮点数占用4个字节

                var bsPtr = Module._malloc(floatArrayBytes);
                Module._updateBlendShape(bsPtr, floatArrayBytes);
                // 从Wasm内存中取出数据
                var bsArray = new Float32Array(Module.HEAPU8.buffer, bsPtr, floatArraySize);
//                console.log("bs", bsArray);



              const dataSet = dataSets[currentDataSetIndex];
              const rect = dataSet.rect;

              const currentTimeStamp = 0.04 * currentDataSetIndex;
              const nextTimeStamp = 0.04 * (currentDataSetIndex + 1);
              const currentpoints = dataSets[currentDataSetIndex].points;
              const nextpoints = dataSets[currentDataSetIndex + 1].points;

//              console.log("rect", rect)

              // 线性插值计算
              const t = (video.currentTime - currentTimeStamp) / (nextTimeStamp - currentTimeStamp);
              let points = currentpoints.map((xi, index) => (1-t) * xi + t * nextpoints[index]);
              // 创建一个新的 mat4 对象
              let matrix = mat4.create();

              mat4.set(matrix,
              points[0], points[1], points[2], points[3],
              points[4], points[5], points[6], points[7],
              points[8], points[9], points[10], points[11],
              points[12], points[13], points[14], points[15]
              );
//              console.log(matrix);
              const subPoints = points.slice(16);
              render(matrix, subPoints, bsArray);

              // 创建临时画布用于裁剪、缩放和绘点
              const tempCanvas = document.createElement('canvas');
              const tempCtx = tempCanvas.getContext('2d');

              // 获取rect区域图像数据并绘制到临时画布
              tempCanvas.width = rect[2] - rect[0];
              tempCanvas.height = rect[3] - rect[1];
              tempCtx.drawImage(
                video,
                rect[0],
                rect[1],
                rect[2] - rect[0],
                rect[3] - rect[1],
                0,
                0,
                tempCanvas.width,
                tempCanvas.height
              );

              // 缩放到128x128
              resizedCanvas.width = 128;
              resizedCanvas.height = 128;
              resizedCtx.drawImage(tempCanvas, 0, 0, 128, 128);

              // 获取128x128图像数据并处理
              const imageData = resizedCtx.getImageData(0, 0, 128, 128);


              var data = imageData.data;
              var imageDataPtr = Module._malloc(data.length);
              Module.HEAPU8.set(data, imageDataPtr);


              var imageDataGlPtr = Module._malloc(pixels_fbo.length);
              Module.HEAPU8.set(pixels_fbo, imageDataGlPtr);
              const processedData = Module._processImage(imageDataPtr, 128, 128, imageDataGlPtr, 128, 128);
              var result = Module.HEAPU8.subarray(imageDataPtr, imageDataPtr + data.length);
              imageData.data.set(result);

              // 更新 canvas 上的图像显示
              resizedCtx.putImageData(imageData, 0, 0);
              // 恢复图像到原始尺寸
              tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
              tempCtx.drawImage(resizedCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

              // 将临时画布的内容放回原画布
              ctx_video.drawImage(tempCanvas, rect[0], rect[1]);

              // 释放分配的内存
              Module._free(imageDataPtr);
              imageDataPtr = null;
              Module._free(imageDataGlPtr);

              Module._free(bsPtr);
            }
          }

          isProcessing = false; // 处理完成后将标志位置为false
        } catch (error) {
          console.error('Error processing frame:', error);
          isProcessing = false; // 即使出错也要将标志位置为false
        }

        requestAnimationFrame(frameCallback);
      }
    };

    requestAnimationFrame(frameCallback);
  }

document.getElementById('uploadButton').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        readWavFile(file);
    }
});

function readWavFile(file) {
    const reader = new FileReader();

    reader.onload = function(event) {
        const arrayBuffer = event.target.result;
        const dataView = new DataView(arrayBuffer);

        // Check if the file is a valid WAV file
        if (dataView.getUint32(0, true) !== 0x46464952 || dataView.getUint32(8, true) !== 0x45564157) {
            alert('Not a valid WAV file');
            return;
        }

        // Get the PCM data chunk
        const chunkSize = dataView.getUint32(4, true);
        const format = dataView.getUint32(12, true);
        const subChunk1Size = dataView.getUint32(16, true);
        const audioFormat = dataView.getUint16(20, true);
        const numChannels = dataView.getUint16(22, true);
        const sampleRate = dataView.getUint32(24, true);
        const byteRate = dataView.getUint32(28, true);
        const blockAlign = dataView.getUint16(32, true);
        const bitsPerSample = dataView.getUint16(34, true);

        // Find the data chunk
        let dataOffset = 36;
        while (dataOffset < arrayBuffer.byteLength) {
            const chunkId = dataView.getUint32(dataOffset, true);
            const chunkSize = dataView.getUint32(dataOffset + 4, true);
            if (chunkId === 0x61746164) { // "data" chunk
                const data = new Uint16Array(arrayBuffer, dataOffset + 8, chunkSize / 2);
                console.log('PCM Data:', data);
                // Convert PCM data to Uint8Array
                const view = new Uint8Array(arrayBuffer);

                // Allocate memory in WebAssembly heap
                const arrayBufferPtr = Module._malloc(arrayBuffer.byteLength);

                // Copy data to WebAssembly heap
                Module.HEAPU8.set(view, arrayBufferPtr);

                // Call WebAssembly module's C function
                console.log("buffer.byteLength", arrayBuffer.byteLength);
                Module._setAudioBuffer(arrayBufferPtr, arrayBuffer.byteLength);

                // Free the allocated memory
                Module._free(arrayBufferPtr);

                // Set the audio source and play it when the video starts

                audioSource.src = URL.createObjectURL(new Blob([arrayBuffer], { type: 'audio/wav' }));
                audio.load();
                audio.play();

                break;
            }
            dataOffset += 8 + chunkSize;
        }
    };

    reader.readAsArrayBuffer(file);
}