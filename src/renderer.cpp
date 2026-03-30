/// @file renderer.cpp
/// @brief OpenGL renderer implementation.

#include "renderer.h"

#include <glad/gl.h>

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace hatching {

Renderer::Renderer() = default;
Renderer::~Renderer() { cleanup(); }

static std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static unsigned int compile_single_shader(unsigned int type,
                                           const std::string& src) {
    unsigned int shader = glCreateShader(type);
    const char* c_src = src.c_str();
    glShaderSource(shader, 1, &c_src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static unsigned int link_program(unsigned int vert, unsigned int frag) {
    unsigned int prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    int success;
    glGetProgramiv(prog, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(prog, 512, nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
        glDeleteProgram(prog);
        return 0;
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

bool Renderer::compile_shader(const std::string& vert_src,
                               const std::string& frag_src) {
    auto vert = compile_single_shader(GL_VERTEX_SHADER, vert_src);
    auto frag = compile_single_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!vert || !frag) return false;
    shader_program_ = link_program(vert, frag);
    return shader_program_ != 0;
}

bool Renderer::compile_line_shader() {
    const char* vert_src = R"(
#version 410 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);
}
)";
    const char* frag_src = R"(
#version 410 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";
    auto vert = compile_single_shader(GL_VERTEX_SHADER, vert_src);
    auto frag = compile_single_shader(GL_FRAGMENT_SHADER, frag_src);
    if (!vert || !frag) return false;
    line_program_ = link_program(vert, frag);
    return line_program_ != 0;
}

bool Renderer::init(const std::string& shader_dir) {
    std::string vert_src = read_file(shader_dir + "/hatching.vert");
    std::string frag_src = read_file(shader_dir + "/hatching.frag");
    if (vert_src.empty() || frag_src.empty()) {
        std::cerr << "Failed to read shader files from " << shader_dir
                  << std::endl;
        return false;
    }
    bool ok = compile_shader(vert_src, frag_src);
    ok = compile_line_shader() && ok;
    return ok;
}

void Renderer::upload_mesh(const TriangleMesh& mesh,
                            const StripePattern& pattern) {
    // Vertex layout: pos(3) + normal(3) + alpha(1) + bary(3) + n_ijk(1) = 11
    int nf = mesh.num_faces();
    int num_verts = nf * 3;
    const int stride = 11;
    std::vector<float> vdata(num_verts * stride);

    for (int f = 0; f < nf; ++f) {
        for (int k = 0; k < 3; ++k) {
            int v = mesh.F(f, k);
            Eigen::Vector3d vn = mesh.vertex_normal(v);
            int base = (f * 3 + k) * stride;
            vdata[base + 0] = static_cast<float>(mesh.V(v, 0));
            vdata[base + 1] = static_cast<float>(mesh.V(v, 1));
            vdata[base + 2] = static_cast<float>(mesh.V(v, 2));
            vdata[base + 3] = static_cast<float>(vn.x());
            vdata[base + 4] = static_cast<float>(vn.y());
            vdata[base + 5] = static_cast<float>(vn.z());
            vdata[base + 6] = static_cast<float>(pattern.alpha(f, k));
            // Barycentric coordinates: (1,0,0), (0,1,0), (0,0,1).
            vdata[base + 7] = (k == 0) ? 1.0f : 0.0f;
            vdata[base + 8] = (k == 1) ? 1.0f : 0.0f;
            vdata[base + 9] = (k == 2) ? 1.0f : 0.0f;
            // Face index (winding number of psi around this face).
            vdata[base + 10] = static_cast<float>(pattern.face_index_raw(f));
        }
    }

    std::vector<unsigned int> indices(num_verts);
    for (int i = 0; i < num_verts; ++i) indices[i] = i;
    num_indices_ = num_verts;

    if (!vao_) {
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenBuffers(1, &ebo_);
    }

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<long>(vdata.size() * sizeof(float)),
                 vdata.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<long>(indices.size() * sizeof(unsigned int)),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          reinterpret_cast<void*>(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          reinterpret_cast<void*>(7 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride * sizeof(float),
                          reinterpret_cast<void*>(10 * sizeof(float)));
    glEnableVertexAttribArray(4);
    glBindVertexArray(0);
}

void Renderer::upload_field(const TriangleMesh& mesh,
                             const MeshGeometry& geom,
                             const DirectionField& field) {
    int nv = mesh.num_vertices();

    double avg_edge = 0;
    for (int e = 0; e < mesh.num_edges(); ++e) {
        avg_edge += mesh.edge_length(e);
    }
    avg_edge /= mesh.num_edges();
    double line_len = avg_edge * 0.4;

    // For a line field (n=2), draw the line segment through both directions
    // at each vertex.  The angle is arg(u)/2, and both arg(u)/2 and
    // arg(u)/2 + pi are valid — drawing the full line covers both.
    std::vector<float> vdata;
    vdata.reserve(nv * 2 * 6);

    for (int v = 0; v < nv; ++v) {
        if (std::abs(field.u(v)) < 1e-15) continue;

        Eigen::Vector3d pos = mesh.V.row(v);
        Eigen::Vector3d N = mesh.vertex_normal(v);

        int ref_nbr = -1;
        for (auto& [nbr, th] : geom.theta[v]) {
            if (std::abs(th) < 1e-10) { ref_nbr = nbr; break; }
        }
        if (ref_nbr < 0) continue;

        Eigen::Vector3d ref_dir_vec = mesh.V.row(ref_nbr) - mesh.V.row(v);
        Eigen::Vector3d ref_dir = ref_dir_vec.normalized();
        ref_dir = (ref_dir - ref_dir.dot(N) * N).normalized();
        Eigen::Vector3d perp = N.cross(ref_dir).normalized();

        // arg(u)/2 gives one of the two line directions.
        double angle = std::arg(field.u(v)) / 2.0;
        Eigen::Vector3d dir =
            std::cos(angle) * ref_dir + std::sin(angle) * perp;

        Eigen::Vector3d p = pos + N * avg_edge * 0.005;
        float r = 0.1f, g = 0.3f, b = 0.9f;

        Eigen::Vector3d p0 = p - dir * line_len;
        Eigen::Vector3d p1 = p + dir * line_len;

        auto push_vertex = [&](const Eigen::Vector3d& pt) {
            vdata.push_back(static_cast<float>(pt.x()));
            vdata.push_back(static_cast<float>(pt.y()));
            vdata.push_back(static_cast<float>(pt.z()));
            vdata.push_back(r);
            vdata.push_back(g);
            vdata.push_back(b);
        };

        push_vertex(p0);
        push_vertex(p1);
    }

    num_field_lines_ = static_cast<int>(vdata.size()) / 6;

    if (!field_vao_) {
        glGenVertexArrays(1, &field_vao_);
        glGenBuffers(1, &field_vbo_);
    }

    glBindVertexArray(field_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, field_vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<long>(vdata.size() * sizeof(float)),
                 vdata.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Renderer::render(const Camera& camera, float aspect,
                       float stripe_frequency, float stripe_width,
                       bool show_field) {
    Eigen::Matrix4f view = camera.view_matrix();
    Eigen::Matrix4f proj = camera.projection_matrix(aspect);
    Eigen::Vector3d cam_pos = camera.position();

    // Draw mesh with hatching.
    if (shader_program_ && vao_) {
        glUseProgram(shader_program_);
        glUniformMatrix4fv(
            glGetUniformLocation(shader_program_, "u_view"), 1, GL_FALSE,
            view.data());
        glUniformMatrix4fv(
            glGetUniformLocation(shader_program_, "u_projection"), 1,
            GL_FALSE, proj.data());
        glUniform3f(glGetUniformLocation(shader_program_, "u_eye"),
                    static_cast<float>(cam_pos.x()),
                    static_cast<float>(cam_pos.y()),
                    static_cast<float>(cam_pos.z()));
        glUniform1f(
            glGetUniformLocation(shader_program_, "u_stripe_frequency"),
            stripe_frequency);
        glUniform1f(
            glGetUniformLocation(shader_program_, "u_stripe_width"),
            stripe_width);

        glBindVertexArray(vao_);
        glDrawElements(GL_TRIANGLES, num_indices_, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    // Draw direction field lines.
    if (show_field && line_program_ && field_vao_ && num_field_lines_ > 0) {
        glUseProgram(line_program_);
        glUniformMatrix4fv(
            glGetUniformLocation(line_program_, "u_view"), 1, GL_FALSE,
            view.data());
        glUniformMatrix4fv(
            glGetUniformLocation(line_program_, "u_projection"), 1,
            GL_FALSE, proj.data());

        glLineWidth(1.5f);
        glBindVertexArray(field_vao_);
        glDrawArrays(GL_LINES, 0, num_field_lines_);
        glBindVertexArray(0);
    }
}

void Renderer::cleanup() {
    if (vao_) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
    if (vbo_) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
    if (ebo_) { glDeleteBuffers(1, &ebo_); ebo_ = 0; }
    if (shader_program_) { glDeleteProgram(shader_program_); shader_program_ = 0; }
    if (field_vao_) { glDeleteVertexArrays(1, &field_vao_); field_vao_ = 0; }
    if (field_vbo_) { glDeleteBuffers(1, &field_vbo_); field_vbo_ = 0; }
    if (line_program_) { glDeleteProgram(line_program_); line_program_ = 0; }
}

} // namespace hatching
