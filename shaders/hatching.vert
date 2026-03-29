#version 410 core

/// Hatching vertex shader.
/// Passes position, normal, and stripe texture coordinate to the fragment
/// shader.

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in float a_alpha;

uniform mat4 u_view;
uniform mat4 u_projection;

out vec3 v_position;
out vec3 v_normal;
out float v_alpha;

void main() {
    v_position = a_position;
    v_normal = a_normal;
    v_alpha = a_alpha;
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);
}
