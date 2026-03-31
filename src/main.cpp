/// @file main.cpp
/// @brief Hatching application entry point.
///
/// Loads a triangle mesh, computes a curvature-aligned direction field and
/// stripe pattern, then renders with an interactive hatching shader.

#include "camera.h"
#include "direction_field.h"
#include "geometry.h"
#include "renderer.h"
#include "stripe_pattern.h"
#include "triangle_mesh.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace hatching;

// ---- Globals for GLFW callbacks ----
static Camera g_camera;
static bool g_mouse_left = false;
static bool g_mouse_right = false;
static double g_mouse_x = 0, g_mouse_y = 0;

static void mouse_button_callback(GLFWwindow* /*window*/, int button,
                                   int action, int /*mods*/) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse_left = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_mouse_right = (action == GLFW_PRESS);
    }
}

static void cursor_pos_callback(GLFWwindow* /*window*/, double x, double y) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    double dx = x - g_mouse_x;
    double dy = y - g_mouse_y;
    g_mouse_x = x;
    g_mouse_y = y;

    if (g_mouse_left) {
        g_camera.rotate(dx, dy);
    }
    if (g_mouse_right) {
        g_camera.pan(dx, dy);
    }
}

static void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/,
                              double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_camera.zoom(yoffset);
}

int main(int argc, char* argv[]) {
    // --- Determine mesh path ---
    std::string mesh_path;
    if (argc > 1) {
        mesh_path = argv[1];
    } else {
        // Default: look for bunny.obj next to the executable or in CWD.
        if (std::filesystem::exists("bunny.obj")) {
            mesh_path = "bunny.obj";
        } else {
            std::cerr << "Usage: hatching <mesh.obj>" << std::endl;
            return 1;
        }
    }

    // --- Load mesh ---
    std::cout << "Loading mesh: " << mesh_path << std::endl;
    TriangleMesh mesh;
    if (!mesh.load_obj(mesh_path)) {
        std::cerr << "Failed to load mesh: " << mesh_path << std::endl;
        return 1;
    }
    std::cout << "  V=" << mesh.num_vertices() << " F=" << mesh.num_faces()
              << " E=" << mesh.num_edges()
              << " chi=" << mesh.euler_characteristic() << std::endl;

    // --- Compute direction field and stripe pattern ---
    float param_s = 0.0f;
    float param_lambda = 0.0f;
    float param_frequency = 40.0f;
    int param_stripe_freq = 1;
    float param_black_threshold = 0.05f;
    float param_white_threshold = 0.6f;
    float param_shading_amount = 1.0f;

    std::cout << "Computing geometry..." << std::endl;
    MeshGeometry geom = compute_geometry(mesh, 2);

    std::cout << "Computing direction field..." << std::endl;
    DirectionField field = compute_direction_field(mesh, param_s, param_lambda);

    int sing_sum = field.singularity_index.sum();
    int pos_sing = (field.singularity_index.array() > 0).count();
    int neg_sing = (field.singularity_index.array() < 0).count();
    std::cout << "  Singularities: " << pos_sing << " positive, " << neg_sing
              << " negative, sum=" << sing_sum << std::endl;

    std::cout << "Computing stripe pattern..." << std::endl;
    StripePattern pattern =
        compute_stripe_pattern(mesh, field, geom, param_frequency);
    std::cout << "  Done." << std::endl;

    // --- Initialize GLFW and OpenGL ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window =
        glfwCreateWindow(1280, 800, "Hatching", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    int glad_version = gladLoadGL(glfwGetProcAddress);
    if (!glad_version) {
        std::cerr << "Failed to initialize glad" << std::endl;
        glfwTerminate();
        return 1;
    }
    std::cout << "OpenGL " << GLAD_VERSION_MAJOR(glad_version) << "."
              << GLAD_VERSION_MINOR(glad_version) << std::endl;

    // --- Setup callbacks ---
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // --- Setup camera ---
    Eigen::Vector3d center = mesh.V.colwise().mean();
    double extent =
        (mesh.V.rowwise() - center.transpose()).rowwise().norm().maxCoeff();
    g_camera.look_at(center, extent * 2.5);

    // --- Setup Dear ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 410");
    ImGui::StyleColorsDark();

    // --- Setup renderer ---
    Renderer renderer;

    // Find shader directory: try next to executable, then CWD, then source dir.
    std::string shader_dir = "shaders";
    if (!std::filesystem::exists(shader_dir + "/hatching.vert")) {
        // Try next to the executable (CMake copies shaders there).
        std::filesystem::path exe_dir =
            std::filesystem::path(argv[0]).parent_path();
        if (std::filesystem::exists(exe_dir / "shaders" / "hatching.vert")) {
            shader_dir = (exe_dir / "shaders").string();
        }
    }

    if (!renderer.init(shader_dir)) {
        std::cerr << "Failed to init renderer (shaders)" << std::endl;
        // Continue anyway — we can still show the UI.
    }

    renderer.upload_mesh(mesh, pattern);
    renderer.upload_field(mesh, geom, field);

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // --- Main loop ---
    bool show_field = false;
    bool perpendicular = false;
    bool needs_recompute = false;
    float light_angle = 0.4f; // radians, rotates around Y axis in camera space

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Arrow keys rotate light in camera space.
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            light_angle -= 0.02f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            light_angle += 0.02f;

        // ImGui frame.
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Hatching Parameters");

        ImGui::Text("Mesh: %s", mesh_path.c_str());
        ImGui::Text("V=%d  F=%d  E=%d  chi=%d", mesh.num_vertices(),
                     mesh.num_faces(), mesh.num_edges(),
                     mesh.euler_characteristic());
        ImGui::Text("Singularities: %d+ %d- (sum=%d)", pos_sing, neg_sing,
                     sing_sum);
        ImGui::Separator();

        ImGui::Text("Shading");
        ImGui::SliderFloat("dark threshold", &param_black_threshold, 0.0f, 1.0f);
        ImGui::SliderFloat("bright threshold", &param_white_threshold, 0.0f, 1.0f);
        ImGui::SliderFloat("shading amount", &param_shading_amount, 0.0f, 1.0f);
        ImGui::Separator();

        ImGui::Text("Rendering");
        ImGui::SliderInt("stripe frequency", &param_stripe_freq, 1, 10);
        ImGui::Separator();

        ImGui::Text("Direction Field");
        int freq_int = static_cast<int>(param_frequency);
        if (ImGui::SliderInt("line frequency", &freq_int, 5, 200)) {
            param_frequency = static_cast<float>(freq_int);
            needs_recompute = true;
        }
        if (ImGui::Checkbox("perpendicular", &perpendicular)) {
            needs_recompute = true;
        }

        if (needs_recompute) {
            needs_recompute = false;
            DirectionField field_for_pattern = field;
            if (perpendicular) field_for_pattern.u = -field_for_pattern.u;
            pattern = compute_stripe_pattern(mesh, field_for_pattern, geom, param_frequency);
            renderer.upload_mesh(mesh, pattern);
        }

        ImGui::End();
        ImGui::Render();

        // Render.
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = static_cast<float>(fb_w) / static_cast<float>(fb_h);

        // Light direction in camera space: slightly from above, rotatable.
        // Extract camera axes from the view matrix.
        Eigen::Matrix4f V_mat = g_camera.view_matrix();
        Eigen::Vector3d cam_right(V_mat(0,0), V_mat(0,1), V_mat(0,2));
        Eigen::Vector3d cam_up(V_mat(1,0), V_mat(1,1), V_mat(1,2));
        Eigen::Vector3d cam_forward(-V_mat(2,0), -V_mat(2,1), -V_mat(2,2));

        // Light in camera space: rotate around Y, tilted slightly up.
        float lx = std::sin(light_angle);
        float ly = 0.5f; // slightly from above
        float lz = std::cos(light_angle);
        // Transform to world space.
        Eigen::Vector3d light_dir = lx * cam_right + ly * cam_up - lz * cam_forward;
        light_dir.normalize();

        renderer.render(g_camera, aspect, static_cast<float>(param_stripe_freq),
                        param_black_threshold, param_white_threshold,
                        param_shading_amount, show_field,
                        static_cast<float>(light_dir.x()),
                        static_cast<float>(light_dir.y()),
                        static_cast<float>(light_dir.z()));

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup.
    renderer.cleanup();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
