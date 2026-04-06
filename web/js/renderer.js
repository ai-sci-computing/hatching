/// WebGL2 renderer for hatching — ports the desktop OpenGL renderer.

const VERT_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in float a_alpha;
layout(location = 3) in vec3 a_bary;
layout(location = 4) in float a_face_index;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_position;
out vec3 v_normal;
out float v_alpha;
out vec3 v_bary;
flat out float v_face_index;

void main() {
    v_position = a_position;
    v_normal = a_normal;
    v_alpha = a_alpha;
    v_bary = a_bary;
    v_face_index = a_face_index;
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

in vec3 v_position;
in vec3 v_normal;
in float v_alpha;
in vec3 v_bary;
flat in float v_face_index;

uniform vec3 u_eye;
uniform vec3 u_light_dir;
uniform float u_stripe_frequency;
uniform float u_black_threshold;
uniform float u_white_threshold;
uniform float u_shading_amount;
out vec4 frag_color;

float lArg(int n, float ti, float tj, float tk) {
    if (n == 0) return 0.0;
    float pi_n = 3.14159265359 * float(n);

    if (tk <= ti && tk <= tj) {
        return pi_n / 3.0 * (1.0 + (tj - ti) / (1.0 - 3.0 * tk));
    } else if (ti <= tj && ti <= tk) {
        return pi_n / 3.0 * (3.0 + (tk - tj) / (1.0 - 3.0 * ti));
    } else {
        return pi_n / 3.0 * (5.0 + (ti - tk) / (1.0 - 3.0 * tj));
    }
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_eye - v_position);
    vec3 L = normalize(u_light_dir);

    if (dot(N, V) < 0.0) N = -N;

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = dot(N, V);

    float tone;
    if (NdotL <= u_black_threshold) {
        tone = 0.0;
    } else if (NdotL >= u_white_threshold) {
        tone = 1.0;
    } else {
        tone = (NdotL - u_black_threshold) / (u_white_threshold - u_black_threshold);
    }

    int n_ijk = int(round(v_face_index));
    float alpha = v_alpha + lArg(n_ijk, v_bary.x, v_bary.y, v_bary.z);
    float V_tex = alpha * u_stripe_frequency / 3.14159265359;

    float sawtooth = fract(V_tex);
    float triangle = abs(2.0 * sawtooth - 1.0);

    float dp = length(vec2(dFdx(V_tex), dFdy(V_tex)));
    float edge = dp * 2.0;

    float silhouette = 1.0 - smoothstep(0.0, 0.25, abs(NdotV));

    vec3 dNdx = dFdx(v_normal);
    vec3 dNdy = dFdy(v_normal);
    float contour_strength = length(dNdx) + length(dNdy);
    float contour = smoothstep(1.0, 2.5, contour_strength);
    float edge_mask = max(silhouette, contour);

    vec3 paper_color = vec3(1.0, 1.0, 1.0);
    vec3 ink_color = vec3(0.05, 0.05, 0.1);

    vec3 color;
    if (tone <= 0.0) {
        color = ink_color;
    } else if (tone >= 1.0) {
        color = paper_color;
    } else {
        float light_width = mix(0.0, 0.95, 1.0 - tone);
        float shade_width = mix(0.475, light_width, u_shading_amount);

        float hatch;
        if (shade_width < 0.01) {
            hatch = 1.0;
        } else {
            hatch = smoothstep(shade_width - edge, shade_width + edge, triangle);
        }
        color = mix(ink_color, paper_color, hatch);
    }

    color = mix(color, ink_color, edge_mask);
    frag_color = vec4(color, 1.0);
}
`;

function compileShader(gl, type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function linkProgram(gl, vert, frag) {
    const prog = gl.createProgram();
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(prog));
        gl.deleteProgram(prog);
        return null;
    }
    return prog;
}

export class Renderer {
    constructor(gl) {
        this.gl = gl;
        this.program = null;
        this.vao = null;
        this.vbo = null;
        this.numVertices = 0;
        this.uniforms = {};
    }

    init() {
        const gl = this.gl;
        const vert = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
        const frag = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
        if (!vert || !frag) return false;
        this.program = linkProgram(gl, vert, frag);
        if (!this.program) return false;

        // Cache uniform locations.
        const names = [
            'u_view', 'u_projection', 'u_eye', 'u_light_dir',
            'u_stripe_frequency', 'u_black_threshold', 'u_white_threshold',
            'u_shading_amount',
        ];
        for (const name of names) {
            this.uniforms[name] = gl.getUniformLocation(this.program, name);
        }

        this.vao = gl.createVertexArray();
        this.vbo = gl.createBuffer();
        return true;
    }

    uploadMesh(vertexData, numVertices) {
        const gl = this.gl;
        this.numVertices = numVertices;

        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.STATIC_DRAW);

        const stride = 11 * 4; // 11 floats * 4 bytes
        // location 0: position (3 floats)
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
        // location 1: normal (3 floats)
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * 4);
        // location 2: alpha (1 float)
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, 6 * 4);
        // location 3: bary (3 floats)
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 3, gl.FLOAT, false, stride, 7 * 4);
        // location 4: face_index (1 float)
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 1, gl.FLOAT, false, stride, 10 * 4);

        gl.bindVertexArray(null);
    }

    render(camera, aspect, params) {
        const gl = this.gl;
        if (!this.program || this.numVertices === 0) return;

        gl.useProgram(this.program);

        const view = camera.viewMatrix();
        const proj = camera.projectionMatrix(aspect);
        const eye = camera.position();

        gl.uniformMatrix4fv(this.uniforms.u_view, false, view);
        gl.uniformMatrix4fv(this.uniforms.u_projection, false, proj);
        gl.uniform3f(this.uniforms.u_eye, eye[0], eye[1], eye[2]);
        gl.uniform3f(this.uniforms.u_light_dir, params.lightDir[0], params.lightDir[1], params.lightDir[2]);
        gl.uniform1f(this.uniforms.u_stripe_frequency, params.stripeFrequency);
        gl.uniform1f(this.uniforms.u_black_threshold, params.blackThreshold);
        gl.uniform1f(this.uniforms.u_white_threshold, params.whiteThreshold);
        gl.uniform1f(this.uniforms.u_shading_amount, params.shadingAmount);

        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.TRIANGLES, 0, this.numVertices);
        gl.bindVertexArray(null);
    }

    cleanup() {
        const gl = this.gl;
        if (this.vbo) gl.deleteBuffer(this.vbo);
        if (this.vao) gl.deleteVertexArray(this.vao);
        if (this.program) gl.deleteProgram(this.program);
    }
}
