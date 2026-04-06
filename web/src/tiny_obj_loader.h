/// @file tiny_obj_loader.h
/// @brief Stub for Emscripten build — provides the tinyobj types so
///        triangle_mesh.cpp compiles. load_obj() won't work at runtime
///        (the web bridge parses OBJ strings directly in bridge.cpp).
#pragma once

#include <string>
#include <vector>

namespace tinyobj {

struct ObjReaderConfig {
    bool triangulate = true;
};

struct index_t {
    int vertex_index = -1;
    int normal_index = -1;
    int texcoord_index = -1;
};

struct mesh_t {
    std::vector<index_t> indices;
    std::vector<unsigned char> num_face_vertices;
};

struct shape_t {
    std::string name;
    mesh_t mesh;
};

struct attrib_t {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texcoords;
};

class ObjReader {
public:
    bool ParseFromFile(const std::string& /*path*/,
                       const ObjReaderConfig& /*config*/ = {}) {
        return false; // Not supported in web build.
    }
    const attrib_t& GetAttrib() const { return attrib_; }
    const std::vector<shape_t>& GetShapes() const { return shapes_; }
private:
    attrib_t attrib_;
    std::vector<shape_t> shapes_;
};

} // namespace tinyobj
