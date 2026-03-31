/// Main application — glue between WASM module, renderer, camera, and UI.

import { Camera } from './camera.js';
import { Renderer } from './renderer.js';
import HatchingModule from '../hatching.js';

// --- Globals ---
let Module = null;
let renderer = null;
const camera = new Camera();
let meshLoaded = false;
let lightAngle = 0.4;
let animFrameId = null;

// Keys currently held.
const keysDown = new Set();

// Shader parameters (synced from sliders).
const params = {
    blackThreshold: 0.05,
    whiteThreshold: 0.60,
    shadingAmount: 1.0,
    stripeFrequency: 1,
    lineFrequency: 40,
    perpendicular: false,
    lightDir: [0, 0, 0],
};

// --- DOM refs ---
const canvas = document.getElementById('canvas');
const dropOverlay = document.getElementById('drop-overlay');
const computingOverlay = document.getElementById('computing-overlay');
const statusEl = document.getElementById('status');
const meshInfoEl = document.getElementById('mesh-info');

// --- WebGL2 init ---
const gl = canvas.getContext('webgl2', { antialias: true });
if (!gl) {
    statusEl.textContent = 'WebGL2 not supported.';
    throw new Error('WebGL2 not available');
}

// --- Renderer ---
renderer = new Renderer(gl);
if (!renderer.init()) {
    statusEl.textContent = 'Shader compilation failed.';
    throw new Error('Shader init failed');
}

// --- Resize handling ---
function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

// --- Camera interaction ---
let mouseDown = 0; // bitmask: 1 = left, 2 = right
let lastX = 0, lastY = 0;

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 0) mouseDown |= 1;
    if (e.button === 2) mouseDown |= 2;
    lastX = e.clientX;
    lastY = e.clientY;
});

window.addEventListener('mouseup', (e) => {
    if (e.button === 0) mouseDown &= ~1;
    if (e.button === 2) mouseDown &= ~2;
});

canvas.addEventListener('mousemove', (e) => {
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    if (mouseDown & 1) camera.rotate(dx, dy);
    if (mouseDown & 2) camera.pan(dx, dy);
});

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camera.zoom(-e.deltaY * 0.01);
}, { passive: false });

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// --- Keyboard for light rotation ---
window.addEventListener('keydown', (e) => keysDown.add(e.key));
window.addEventListener('keyup', (e) => keysDown.delete(e.key));

// --- Sliders ---
function bindSlider(id, key, valId, format) {
    const el = document.getElementById(id);
    const valEl = document.getElementById(valId);
    el.addEventListener('input', () => {
        const v = parseFloat(el.value);
        params[key] = v;
        valEl.textContent = format(v);
    });
}

bindSlider('dark-threshold', 'blackThreshold', 'val-dark', v => v.toFixed(2));
bindSlider('bright-threshold', 'whiteThreshold', 'val-bright', v => v.toFixed(2));
bindSlider('shading-amount', 'shadingAmount', 'val-shading', v => v.toFixed(2));
bindSlider('stripe-freq', 'stripeFrequency', 'val-freq', v => v.toFixed(0));

// Line frequency triggers recompute.
const lineFreqEl = document.getElementById('line-freq');
const lineFreqValEl = document.getElementById('val-line-freq');
lineFreqEl.addEventListener('change', () => {
    const v = parseFloat(lineFreqEl.value);
    params.lineFrequency = v;
    lineFreqValEl.textContent = v.toFixed(0);
    if (meshLoaded) recompute();
});
lineFreqEl.addEventListener('input', () => {
    lineFreqValEl.textContent = parseFloat(lineFreqEl.value).toFixed(0);
});

// Perpendicular toggle triggers recompute.
const perpEl = document.getElementById('perpendicular');
perpEl.addEventListener('change', () => {
    params.perpendicular = perpEl.checked;
    if (meshLoaded) recompute();
});

// --- Prevent browser from handling dropped files ---
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());

// --- Drag and drop (on the whole canvas wrapper) ---
const canvasWrap = document.getElementById('canvas-wrap');

canvasWrap.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('hidden');
    dropOverlay.classList.add('drag-active');
});

canvasWrap.addEventListener('dragover', (e) => {
    e.preventDefault();
});

canvasWrap.addEventListener('dragleave', (e) => {
    // Only hide when leaving the wrapper itself, not its children.
    if (e.relatedTarget && canvasWrap.contains(e.relatedTarget)) return;
    e.preventDefault();
    if (meshLoaded) {
        dropOverlay.classList.add('hidden');
    }
    dropOverlay.classList.remove('drag-active');
});

canvasWrap.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('drag-active');
    if (meshLoaded) dropOverlay.classList.add('hidden');
    const file = e.dataTransfer.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => loadMesh(reader.result, file.name);
        reader.readAsText(file);
    }
});

// --- Load + compute ---
function loadMesh(objText, name) {
    if (!Module) {
        statusEl.textContent = 'WASM module not loaded yet.';
        return;
    }

    statusEl.textContent = `Loading ${name}...`;

    if (!Module.loadOBJ(objText)) {
        statusEl.textContent = `Failed to parse ${name}`;
        return;
    }

    const nv = Module.getMeshNumVertices();
    const nf = Module.getMeshNumFaces();
    const ne = Module.getMeshNumEdges();
    const chi = Module.getMeshEulerCharacteristic();
    meshInfoEl.innerHTML = `<b>${name}</b><br>V=${nv}  F=${nf}  E=${ne}  &chi;=${chi}`;

    // Setup camera.
    const center = Module.getMeshCenter();
    const extent = Module.getMeshExtent();
    camera.lookAt([center[0], center[1], center[2]], extent * 2.5);

    meshLoaded = true;
    dropOverlay.classList.add('hidden');
    recompute();
}

function recompute() {
    if (!Module) return;
    computingOverlay.classList.add('active');
    statusEl.textContent = 'Computing...';

    // Defer to let the overlay paint.
    setTimeout(() => {
        const ok = Module.compute(params.lineFrequency, 0.0, 0.0, params.perpendicular);
        computingOverlay.classList.remove('active');

        if (!ok) {
            statusEl.textContent = 'Computation failed.';
            return;
        }

        // Get vertex buffer from WASM and copy it (the WASM memory can move).
        const wasmBuf = Module.getVertexBuffer();
        const vertexData = new Float32Array(wasmBuf);
        const numVerts = Module.getNumVertices();

        renderer.uploadMesh(vertexData, numVerts);
        statusEl.textContent = 'Ready.';
    }, 30);
}

// --- Render loop ---
function frame() {
    animFrameId = requestAnimationFrame(frame);

    // Light rotation via arrow keys.
    if (keysDown.has('ArrowLeft')) lightAngle -= 0.02;
    if (keysDown.has('ArrowRight')) lightAngle += 0.02;

    // Compute light direction in camera space, same as desktop.
    const view = camera.viewMatrix();
    // Extract camera axes from view matrix (row-major in the matrix, but
    // stored column-major in the Float32Array).
    const camRight = [view[0], view[4], view[8]];
    const camUp = [view[1], view[5], view[9]];
    const camFwd = [-view[2], -view[6], -view[10]];

    const lx = Math.sin(lightAngle);
    const ly = 0.5;
    const lz = Math.cos(lightAngle);
    const ld = [
        lx * camRight[0] + ly * camUp[0] - lz * camFwd[0],
        lx * camRight[1] + ly * camUp[1] - lz * camFwd[1],
        lx * camRight[2] + ly * camUp[2] - lz * camFwd[2],
    ];
    const ll = Math.sqrt(ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]);
    params.lightDir = [ld[0] / ll, ld[1] / ll, ld[2] / ll];

    // Clear and draw.
    resize(); // handle DPR changes
    gl.clearColor(1, 1, 1, 1);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const aspect = canvas.width / canvas.height;
    renderer.render(camera, aspect, params);
}

// --- Init WASM ---
async function init() {
    try {
        // HatchingModule is defined by Emscripten's MODULARIZE option.
        Module = await HatchingModule();
        statusEl.textContent = 'Ready. Drop an OBJ file to begin.';

        // Try loading a default mesh.
        try {
            const resp = await fetch('bunny.obj');
            if (resp.ok) {
                const text = await resp.text();
                loadMesh(text, 'bunny.obj');
            }
        } catch {
            // No default mesh — that's fine.
        }

        frame();
    } catch (err) {
        statusEl.textContent = 'Failed to load WASM module: ' + err.message;
        console.error(err);
    }
}

init();
