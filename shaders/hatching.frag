#version 410 core

/// Hatching fragment shader with singularity handling via lArg (Eq. 11).

in vec3 v_position;
in vec3 v_normal;
in float v_alpha;       // linearly interpolated alpha (corner-corrected)
in vec3 v_bary;         // interpolated barycentric coordinates
flat in float v_face_index;  // face winding number n_ijk (flat = per-face)

uniform vec3 u_eye;
uniform float u_stripe_frequency;
uniform float u_stripe_width;

out vec4 frag_color;

/// lArg interpolant (Eq. 11, Knoeppel 2015).
/// Provides a continuous map to the unit circle that winds n times
/// around the barycenter of the triangle.
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
    // --- Shading ---
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_eye - v_position);
    vec3 L = normalize(vec3(0.5, 1.0, 0.3));

    float NdotL = max(dot(N, L), 0.0);
    float ambient = 0.15;
    float diffuse = 0.65 * NdotL;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.2;

    float shading = ambient + diffuse + spec;

    // --- Stripe pattern with singularity handling ---
    // The vertex alpha values have been adjusted by subtracting the lArg
    // corner values (Algorithm 7 lines 28-29).  The GPU linearly interpolates
    // these adjusted values.  We add lArg evaluated at the fragment's
    // barycentric coordinates to get the correct nonlinear interpolation
    // near singularities (zeros of psi).
    int n_ijk = int(round(v_face_index));
    float alpha = v_alpha + lArg(n_ijk, v_bary.x, v_bary.y, v_bary.z);

    // Convert to stripe units (period = 1) and apply visual frequency.
    float V_tex = alpha * u_stripe_frequency / (2.0 * 3.14159265359);

    // Triangle wave via fract (handles integer jumps, symmetric under negation).
    float sawtooth = fract(V_tex);
    float triangle = abs(2.0 * sawtooth - 1.0);

    // Antialiasing via screen-space derivatives.
    float dp = length(vec2(dFdx(V_tex), dFdy(V_tex)));
    float edge = dp * 2.0;

    float threshold = 1.0 - u_stripe_width;
    float hatch = smoothstep(threshold - edge, threshold + edge, triangle);

    // Combine: white paper with dark hatching lines.
    vec3 paper_color = vec3(1.0, 0.98, 0.95);
    vec3 ink_color = vec3(0.05, 0.05, 0.1);

    vec3 base = mix(ink_color, paper_color, hatch);
    vec3 color = base * shading;

    frag_color = vec4(color, 1.0);
}
