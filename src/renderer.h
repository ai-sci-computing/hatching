#pragma once

/// @file renderer.h
/// @brief OpenGL renderer for the hatching application.

#include "camera.h"
#include "direction_field.h"
#include "geometry.h"
#include "stripe_pattern.h"
#include "triangle_mesh.h"

#include <Eigen/Core>
#include <string>

namespace hatching {

/// @brief OpenGL renderer that displays a mesh with hatching stripes.
class Renderer {
public:
    Renderer();
    ~Renderer();

    /// @brief Initialize OpenGL resources (shaders, buffers).
    bool init(const std::string& shader_dir);

    /// @brief Upload mesh data and stripe pattern to the GPU.
    void upload_mesh(const TriangleMesh& mesh, const StripePattern& pattern);

    /// @brief Upload direction field for visualization as line segments.
    void upload_field(const TriangleMesh& mesh, const MeshGeometry& geom,
                      const DirectionField& field);

    /// @brief Render one frame.
    void render(const Camera& camera, float aspect, float stripe_frequency,
                float black_threshold, float white_threshold,
                float shading_amount, bool show_field,
                float light_x = 0.5f, float light_y = 1.0f, float light_z = 0.3f);

    /// @brief Clean up GPU resources.
    void cleanup();

private:
    unsigned int shader_program_ = 0;
    unsigned int line_program_ = 0;
    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    unsigned int ebo_ = 0;
    int num_indices_ = 0;

    unsigned int field_vao_ = 0;
    unsigned int field_vbo_ = 0;
    int num_field_lines_ = 0;

    bool compile_shader(const std::string& vert_src,
                        const std::string& frag_src);
    bool compile_line_shader();
};

} // namespace hatching
