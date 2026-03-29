#version 410 core

/// Hatching fragment shader.
///
/// Evaluates a periodic function of the stripe texture coordinate alpha to
/// produce hatching lines.  Combines with basic Phong-like shading for
/// depth cues.

in vec3 v_position;
in vec3 v_normal;
in float v_alpha;

uniform vec3 u_eye;
uniform float u_stripe_frequency; // visual frequency multiplier
uniform float u_stripe_width;     // 0..1, relative width of dark stripes

out vec4 frag_color;

void main() {
    // --- Shading ---
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_eye - v_position);
    vec3 L = normalize(vec3(0.5, 1.0, 0.3)); // fixed light direction

    float NdotL = max(dot(N, L), 0.0);
    float ambient = 0.15;
    float diffuse = 0.65 * NdotL;

    // Half-vector specular.
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.2;

    float shading = ambient + diffuse + spec;

    // --- Stripe pattern ---
    // Evaluate cos(frequency * alpha).  The stripe pattern paper outputs
    // alpha such that plugging into any 2pi-periodic function yields
    // globally continuous stripes.
    float pattern = cos(u_stripe_frequency * v_alpha);

    // Convert to binary hatching via smoothstep.
    // stripe_width controls the black-to-white ratio.
    float threshold = 1.0 - 2.0 * u_stripe_width;
    float fw = fwidth(pattern) * 1.5; // anti-aliasing
    float hatch = smoothstep(threshold - fw, threshold + fw, pattern);

    // Combine: white paper with dark hatching lines.
    vec3 paper_color = vec3(1.0, 0.98, 0.95);
    vec3 ink_color = vec3(0.05, 0.05, 0.1);

    vec3 base = mix(ink_color, paper_color, hatch);
    vec3 color = base * shading;

    frag_color = vec4(color, 1.0);
}
