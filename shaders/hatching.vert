#version 410 core

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
