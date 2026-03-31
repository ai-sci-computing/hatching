#version 410 core

/// NPR hatching shader: shading via stripe density, silhouette & contour edges.

in vec3 v_position;
in vec3 v_normal;
in float v_alpha;       // linearly interpolated alpha (corner-corrected)
in vec3 v_bary;         // interpolated barycentric coordinates
flat in float v_face_index;  // face winding number n_ijk (flat = per-face)

uniform vec3 u_eye;
uniform vec3 u_light_dir;
uniform float u_stripe_frequency;
uniform float u_stripe_width;
out vec4 frag_color;

/// lArg interpolant (Eq. 11, Knoeppel 2015).
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

    // Two-sided normal.
    if (dot(N, V) < 0.0) N = -N;

    // --- Lighting for hatching density ---
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = dot(N, V);

    // Tone: 0 = dark (dense hatching), 1 = bright (no hatching).
    // Remap with contrast so bright areas clear out fully.
    float tone = smoothstep(0.0, 0.6, NdotL);

    // --- Stripe pattern ---
    int n_ijk = int(round(v_face_index));
    float alpha = v_alpha + lArg(n_ijk, v_bary.x, v_bary.y, v_bary.z);

    float V_tex = alpha * u_stripe_frequency / 3.14159265359;

    float sawtooth = fract(V_tex);
    float triangle = abs(2.0 * sawtooth - 1.0);

    // Antialiasing.
    float dp = length(vec2(dFdx(V_tex), dFdy(V_tex)));
    float edge = dp * 2.0;

    // Modulate stripe width by shading.
    // shade_width: 0 in bright areas (no stripes), up to ~0.95 in darkest
    // areas (nearly solid ink).  The full range gives bright-to-almost-black.
    float shade_width = mix(0.0, 0.95, 1.0 - tone);

    // In bright areas, force pure paper (no stripes at all).
    float hatch;
    if (shade_width < 0.01) {
        hatch = 1.0;
    } else {
        hatch = smoothstep(shade_width - edge, shade_width + edge, triangle);
    }

    // --- Silhouette & contour edges ---
    // Silhouette: where surface is nearly edge-on to the camera.
    float silhouette = 1.0 - smoothstep(0.0, 0.25, abs(NdotV));

    // Contour: detect sharp changes in normal via screen-space derivatives.
    vec3 dNdx = dFdx(v_normal);
    vec3 dNdy = dFdy(v_normal);
    float contour_strength = length(dNdx) + length(dNdy);
    float contour = smoothstep(1.0, 2.5, contour_strength);

    // --- Compose ---
    vec3 paper_color = vec3(1.0, 1.0, 1.0);
    vec3 ink_color = vec3(0.05, 0.05, 0.1);

    // Start with paper, apply hatching stripes.
    vec3 color = mix(ink_color, paper_color, hatch);

    // Draw silhouette and contour edges in ink.
    float edge_mask = max(silhouette, contour);
    color = mix(color, ink_color, edge_mask);

    frag_color = vec4(color, 1.0);
}
