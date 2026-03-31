/// @file bridge.cpp
/// @brief Emscripten bindings for hatching_core → JavaScript.
///
/// Exposes a simple API via embind:
///   loadOBJ(string) → bool
///   compute(frequency, s, lambda) → bool
///   getVertexBuffer() → Float32Array  (pos3 + normal3 + alpha1 + bary3 + n_ijk1 = 11 floats per vertex)
///   getNumVertices() → int

#include "direction_field.h"
#include "geometry.h"
#include "stripe_pattern.h"
#include "triangle_mesh.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <sstream>
#include <string>
#include <vector>

using namespace hatching;

// Global state — one mesh at a time.
static TriangleMesh g_mesh;
static std::vector<float> g_vertex_buffer;
static int g_num_render_verts = 0;

/// Parse OBJ from a string (tinyobjloader reads files; we use a simple parser
/// here since the data comes from JS as a string).
static bool parse_obj_string(const std::string& obj_data, TriangleMesh& mesh) {
    std::istringstream stream(obj_data);
    std::string line;

    std::vector<double> verts;
    std::vector<int> faces;

    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ls(line);
        std::string prefix;
        ls >> prefix;

        if (prefix == "v") {
            double x, y, z;
            ls >> x >> y >> z;
            verts.push_back(x);
            verts.push_back(y);
            verts.push_back(z);
        } else if (prefix == "f") {
            // Handle "f v", "f v/vt", "f v/vt/vn", "f v//vn".
            std::vector<int> face_verts;
            std::string token;
            while (ls >> token) {
                int vi = std::stoi(token.substr(0, token.find('/')));
                face_verts.push_back(vi - 1); // OBJ is 1-indexed.
            }
            // Triangulate simple convex polygons (fan from vertex 0).
            for (size_t i = 1; i + 1 < face_verts.size(); ++i) {
                faces.push_back(face_verts[0]);
                faces.push_back(face_verts[i]);
                faces.push_back(face_verts[i + 1]);
            }
        }
    }

    if (verts.empty() || faces.empty()) return false;

    int nv = static_cast<int>(verts.size()) / 3;
    int nf = static_cast<int>(faces.size()) / 3;

    mesh.V.resize(nv, 3);
    for (int i = 0; i < nv; ++i) {
        mesh.V(i, 0) = verts[3 * i + 0];
        mesh.V(i, 1) = verts[3 * i + 1];
        mesh.V(i, 2) = verts[3 * i + 2];
    }

    mesh.F.resize(nf, 3);
    for (int i = 0; i < nf; ++i) {
        mesh.F(i, 0) = faces[3 * i + 0];
        mesh.F(i, 1) = faces[3 * i + 1];
        mesh.F(i, 2) = faces[3 * i + 2];
    }

    mesh.build_topology();
    return true;
}

static bool parse_off_string(const std::string& off_data, TriangleMesh& mesh) {
    std::istringstream stream(off_data);
    std::string header;
    stream >> header;
    if (header != "OFF") return false;

    int nv, nf, ne;
    stream >> nv >> nf >> ne;
    if (nv <= 0 || nf <= 0) return false;

    mesh.V.resize(nv, 3);
    for (int i = 0; i < nv; ++i) {
        stream >> mesh.V(i, 0) >> mesh.V(i, 1) >> mesh.V(i, 2);
    }

    std::vector<int> faces;
    for (int i = 0; i < nf; ++i) {
        int fv;
        stream >> fv;
        std::vector<int> verts(fv);
        for (int j = 0; j < fv; ++j) stream >> verts[j];
        for (int j = 1; j + 1 < fv; ++j) {
            faces.push_back(verts[0]);
            faces.push_back(verts[j]);
            faces.push_back(verts[j + 1]);
        }
    }

    int ntri = static_cast<int>(faces.size()) / 3;
    mesh.F.resize(ntri, 3);
    for (int i = 0; i < ntri; ++i) {
        mesh.F(i, 0) = faces[3 * i + 0];
        mesh.F(i, 1) = faces[3 * i + 1];
        mesh.F(i, 2) = faces[3 * i + 2];
    }

    mesh.build_topology();
    return true;
}

bool loadMesh(const std::string& data, const std::string& filename) {
    g_mesh = TriangleMesh{};
    g_vertex_buffer.clear();
    g_num_render_verts = 0;

    // Detect format from extension or content.
    bool is_off = false;
    if (filename.size() >= 4 &&
        filename.substr(filename.size() - 4) == ".off") {
        is_off = true;
    } else if (data.size() >= 3 && data.substr(0, 3) == "OFF") {
        is_off = true;
    }

    if (is_off) {
        return parse_off_string(data, g_mesh);
    }
    return parse_obj_string(data, g_mesh);
}

bool compute(double frequency, double s, double lambda, bool perpendicular) {
    if (g_mesh.num_faces() == 0) return false;

    MeshGeometry geom = compute_geometry(g_mesh, 2);
    DirectionField field = compute_direction_field(g_mesh, s, lambda);
    if (perpendicular) field.u = -field.u;
    Eigen::Vector3d bb_lo = g_mesh.V.colwise().minCoeff();
    Eigen::Vector3d bb_hi = g_mesh.V.colwise().maxCoeff();
    double diameter = (bb_hi - bb_lo).norm();
    double freq_scale = 2.0 * M_PI / diameter;
    StripePattern pattern = compute_stripe_pattern(g_mesh, field, geom, frequency * freq_scale);

    // Build vertex buffer — same layout as renderer.cpp.
    // Stride: pos(3) + normal(3) + alpha(1) + bary(3) + n_ijk(1) = 11
    int nf = g_mesh.num_faces();
    int num_verts = nf * 3;
    const int stride = 11;
    g_vertex_buffer.resize(num_verts * stride);

    for (int f = 0; f < nf; ++f) {
        for (int k = 0; k < 3; ++k) {
            int v = g_mesh.F(f, k);
            Eigen::Vector3d vn = g_mesh.vertex_normal(v);
            int base = (f * 3 + k) * stride;
            g_vertex_buffer[base + 0] = static_cast<float>(g_mesh.V(v, 0));
            g_vertex_buffer[base + 1] = static_cast<float>(g_mesh.V(v, 1));
            g_vertex_buffer[base + 2] = static_cast<float>(g_mesh.V(v, 2));
            g_vertex_buffer[base + 3] = static_cast<float>(vn.x());
            g_vertex_buffer[base + 4] = static_cast<float>(vn.y());
            g_vertex_buffer[base + 5] = static_cast<float>(vn.z());
            g_vertex_buffer[base + 6] = static_cast<float>(pattern.alpha(f, k));
            g_vertex_buffer[base + 7] = (k == 0) ? 1.0f : 0.0f;
            g_vertex_buffer[base + 8] = (k == 1) ? 1.0f : 0.0f;
            g_vertex_buffer[base + 9] = (k == 2) ? 1.0f : 0.0f;
            g_vertex_buffer[base + 10] = static_cast<float>(pattern.face_index_raw(f));
        }
    }

    g_num_render_verts = num_verts;
    return true;
}

emscripten::val getVertexBuffer() {
    return emscripten::val(emscripten::typed_memory_view(
        g_vertex_buffer.size(), g_vertex_buffer.data()));
}

int getNumVertices() {
    return g_num_render_verts;
}

int getMeshNumVertices() {
    return g_mesh.num_vertices();
}

int getMeshNumFaces() {
    return g_mesh.num_faces();
}

int getMeshNumEdges() {
    return g_mesh.num_edges();
}

int getMeshEulerCharacteristic() {
    return g_mesh.euler_characteristic();
}

emscripten::val getMeshCenter() {
    Eigen::Vector3d center = g_mesh.V.colwise().mean();
    auto result = emscripten::val::array();
    result.call<void>("push", center.x());
    result.call<void>("push", center.y());
    result.call<void>("push", center.z());
    return result;
}

double getMeshExtent() {
    Eigen::Vector3d center = g_mesh.V.colwise().mean();
    return (g_mesh.V.rowwise() - center.transpose()).rowwise().norm().maxCoeff();
}

EMSCRIPTEN_BINDINGS(hatching_web) {
    emscripten::function("loadMesh", &loadMesh);
    emscripten::function("compute", &compute);
    emscripten::function("getVertexBuffer", &getVertexBuffer);
    emscripten::function("getNumVertices", &getNumVertices);
    emscripten::function("getMeshNumVertices", &getMeshNumVertices);
    emscripten::function("getMeshNumFaces", &getMeshNumFaces);
    emscripten::function("getMeshNumEdges", &getMeshNumEdges);
    emscripten::function("getMeshEulerCharacteristic", &getMeshEulerCharacteristic);
    emscripten::function("getMeshCenter", &getMeshCenter);
    emscripten::function("getMeshExtent", &getMeshExtent);
}
