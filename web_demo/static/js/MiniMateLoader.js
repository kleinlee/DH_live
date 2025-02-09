document.addEventListener('DOMContentLoaded', function () {
  init();
});

async function init()
{
            const spinner = document.querySelector('#loadingSpinner');
            const screen = document.querySelector('#screen');
            const showUi = () => {
                spinner.style.display = 'none';
                screen.style.display = 'block';
            }
            const instance = await qtLoad({
                    qt: {
                        onLoaded: () => showUi(),
                        entryFunction: window.createQtAppInstance,
                        containerElements: [screen],
                    }
                });
            await newVideoTask();
            document.getElementById('screen2').style.display = 'block';
}


async function qtLoad(config)
{
    const throwIfEnvUsedButNotExported = (instance, config) =>
    {
        const environment = config.environment;
        if (!environment || Object.keys(environment).length === 0)
            return;
        const isEnvExported = typeof instance.ENV === 'object';
        if (!isEnvExported)
            throw new Error('ENV must be exported if environment variables are passed');
    };

    const throwIfFsUsedButNotExported = (instance, config) =>
    {
        const environment = config.environment;
        if (!environment || Object.keys(environment).length === 0)
            return;
        const isFsExported = typeof instance.FS === 'object';
        if (!isFsExported)
            throw new Error('FS must be exported if preload is used');
    };

    if (typeof config !== 'object')
        throw new Error('config is required, expected an object');
    if (typeof config.qt !== 'object')
        throw new Error('config.qt is required, expected an object');
    if (typeof config.qt.entryFunction !== 'function')
        config.qt.entryFunction = window.createQtAppInstance;

    config.qt.qtdir ??= 'qt';
    config.qt.preload ??= [];

    config.qtContainerElements = config.qt.containerElements;
    delete config.qt.containerElements;
    config.qtFontDpi = config.qt.fontDpi;
    delete config.qt.fontDpi;

    // Used for rejecting a failed load's promise where emscripten itself does not allow it,
    // like in instantiateWasm below. This allows us to throw in case of a load error instead of
    // hanging on a promise to entry function, which emscripten unfortunately does.
    let circuitBreakerReject;
    const circuitBreaker = new Promise((_, reject) => { circuitBreakerReject = reject; });

    // If module async getter is present, use it so that module reuse is possible.
    if (config.qt.module) {
        config.instantiateWasm = async (imports, successCallback) =>
        {
            try {
                const module = await config.qt.module;
                successCallback(
                    await WebAssembly.instantiate(module, imports), module);
            } catch (e) {
                circuitBreakerReject(e);
            }
        }
    }

    const qtPreRun = (instance) => {
        // Copy qt.environment to instance.ENV
        throwIfEnvUsedButNotExported(instance, config);
        for (const [name, value] of Object.entries(config.qt.environment ?? {}))
            instance.ENV[name] = value;

        // Copy self.preloadData to MEMFS
        const makeDirs = (FS, filePath) => {
            const parts = filePath.split("/");
            let path = "/";
            for (let i = 0; i < parts.length - 1; ++i) {
                const part = parts[i];
                if (part == "")
                    continue;
                path += part + "/";
                try {
                    FS.mkdir(path);
                } catch (error) {
                    const EEXIST = 20;
                    if (error.errno != EEXIST)
                        throw error;
                }
            }
        }
        throwIfFsUsedButNotExported(instance, config);
        for ({destination, data} of self.preloadData) {
            makeDirs(instance.FS, destination);
            instance.FS.writeFile(destination, new Uint8Array(data));
        }
    }

    if (!config.preRun)
        config.preRun = [];
    config.preRun.push(qtPreRun);

    config.onRuntimeInitialized = () => config.qt.onLoaded?.();

    const originalLocateFile = config.locateFile;
    config.locateFile = filename =>
    {
        const originalLocatedFilename = originalLocateFile ? originalLocateFile(filename) : filename;
        if (originalLocatedFilename.startsWith('libQt6'))
            return `${config.qt.qtdir}/lib/${originalLocatedFilename}`;
        return originalLocatedFilename;
    }

    const originalOnExit = config.onExit;
    config.onExit = code => {
        originalOnExit?.();
        config.qt.onExit?.({
            code,
            crashed: false
        });
    }

    const originalOnAbort = config.onAbort;
    config.onAbort = text =>
    {
        originalOnAbort?.();

        aborted = true;
        config.qt.onExit?.({
            text,
            crashed: true
        });
    };

    const fetchPreloadFiles = async () => {
        const fetchJson = async path => (await fetch(path)).json();
        const fetchArrayBuffer = async path => (await fetch(path)).arrayBuffer();
        const loadFiles = async (paths) => {
            const source = paths['source'].replace('$QTDIR', config.qt.qtdir);
            return {
                destination: paths['destination'],
                data: await fetchArrayBuffer(source)
            };
        }
        const fileList = (await Promise.all(config.qt.preload.map(fetchJson))).flat();
        self.preloadData = (await Promise.all(fileList.map(loadFiles))).flat();
    }

    await fetchPreloadFiles();

    // Call app/emscripten module entry function. It may either come from the emscripten
    // runtime script or be customized as needed.
    let instance;
    try {
        instance = await Promise.race(
            [circuitBreaker, config.qt.entryFunction(config)]);
    } catch (e) {
        config.qt.onExit?.({
            text: e.message,
            crashed: true
        });
        throw e;
    }

    return instance;
}

// Compatibility API. This API is deprecated,
// and will be removed in a future version of Qt.
function QtLoader(qtConfig) {

    const warning = 'Warning: The QtLoader API is deprecated and will be removed in ' +
                    'a future version of Qt. Please port to the new qtLoad() API.';
    console.warn(warning);

    let emscriptenConfig = qtConfig.moduleConfig || {}
    qtConfig.moduleConfig = undefined;
    const showLoader = qtConfig.showLoader;
    qtConfig.showLoader = undefined;
    const showError = qtConfig.showError;
    qtConfig.showError = undefined;
    const showExit = qtConfig.showExit;
    qtConfig.showExit = undefined;
    const showCanvas = qtConfig.showCanvas;
    qtConfig.showCanvas = undefined;
    if (qtConfig.canvasElements) {
        qtConfig.containerElements = qtConfig.canvasElements
        qtConfig.canvasElements = undefined;
    } else {
        qtConfig.containerElements = qtConfig.containerElements;
        qtConfig.containerElements = undefined;
    }
    emscriptenConfig.qt = qtConfig;

    let qtloader = {
        exitCode: undefined,
        exitText: "",
        loadEmscriptenModule: _name => {
            try {
                qtLoad(emscriptenConfig);
            } catch (e) {
                showError?.(e.message);
            }
        }
    }

    qtConfig.onLoaded = () => {
        showCanvas?.();
    }

    qtConfig.onExit = exit => {
        qtloader.exitCode = exit.code
        qtloader.exitText = exit.text;
        showExit?.();
    }

    showLoader?.("Loading");

    return qtloader;
};
