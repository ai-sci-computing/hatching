/// @file triangle_mesh.cpp
/// @brief Implementation of TriangleMesh.

#include "triangle_mesh.h"

#include <tiny_obj_loader.h>

#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>

namespace hatching {

// ---------------------------------------------------------------------------
// OBJ loading
// ---------------------------------------------------------------------------

bool TriangleMesh::load_obj(const std::string& path) {
    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;

    if (!reader.ParseFromFile(path, config)) {
        return false;
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    // Vertices.
    int nv = static_cast<int>(attrib.vertices.size()) / 3;
    V.resize(nv, 3);
    for (int i = 0; i < nv; ++i) {
        V(i, 0) = attrib.vertices[3 * i + 0];
        V(i, 1) = attrib.vertices[3 * i + 1];
        V(i, 2) = attrib.vertices[3 * i + 2];
    }

    // Faces — collect from all shapes.
    int total_faces = 0;
    for (const auto& shape : shapes) {
        total_faces +=
            static_cast<int>(shape.mesh.num_face_vertices.size());
    }

    F.resize(total_faces, 3);
    int fi = 0;
    for (const auto& shape : shapes) {
        int index_offset = 0;
        for (int f = 0;
             f < static_cast<int>(shape.mesh.num_face_vertices.size());
             ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                return false; // non-triangle face
            }
            F(fi, 0) = shape.mesh.indices[index_offset + 0].vertex_index;
            F(fi, 1) = shape.mesh.indices[index_offset + 1].vertex_index;
            F(fi, 2) = shape.mesh.indices[index_offset + 2].vertex_index;
            index_offset += fv;
            ++fi;
        }
    }

    build_topology();
    return true;
}

bool TriangleMesh::load_off(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string header;
    file >> header;
    if (header != "OFF") return false;

    int nv, nf, ne;
    file >> nv >> nf >> ne;
    if (nv <= 0 || nf <= 0) return false;

    V.resize(nv, 3);
    for (int i = 0; i < nv; ++i) {
        file >> V(i, 0) >> V(i, 1) >> V(i, 2);
    }

    // Read faces, triangulating polygons with a fan from vertex 0.
    std::vector<Eigen::Vector3i> tris;
    tris.reserve(nf);
    for (int i = 0; i < nf; ++i) {
        int fv;
        file >> fv;
        std::vector<int> verts(fv);
        for (int j = 0; j < fv; ++j) {
            file >> verts[j];
        }
        for (int j = 1; j + 1 < fv; ++j) {
            tris.push_back({verts[0], verts[j], verts[j + 1]});
        }
    }

    F.resize(static_cast<int>(tris.size()), 3);
    for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
        F.row(i) = tris[i];
    }

    build_topology();
    return true;
}

bool TriangleMesh::load(const std::string& path) {
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".off") {
        return load_off(path);
    }
    return load_obj(path);
}

// ---------------------------------------------------------------------------
// Topology construction
// ---------------------------------------------------------------------------

void TriangleMesh::build_topology() {
    const int nv = num_vertices();
    const int nf = num_faces();

    // Reset.
    edges.clear();
    edge_map_.clear();
    VF.assign(nv, {});
    VE.assign(nv, {});

    FE.resize(nf, 3);

    // Pass 1: collect edges and vertex-face adjacency.
    for (int f = 0; f < nf; ++f) {
        int v[3] = {F(f, 0), F(f, 1), F(f, 2)};

        for (int k = 0; k < 3; ++k) {
            VF[v[k]].push_back(f);
        }

        // Edge opposite to local vertex k connects v[(k+1)%3] and v[(k+2)%3].
        for (int k = 0; k < 3; ++k) {
            int a = v[(k + 1) % 3];
            int b = v[(k + 2) % 3];
            int64_t key = edge_key(a, b);

            auto it = edge_map_.find(key);
            if (it == edge_map_.end()) {
                int ei = static_cast<int>(edges.size());
                int lo = std::min(a, b);
                int hi = std::max(a, b);
                edges.push_back({lo, hi});
                edge_map_[key] = ei;
                FE(f, k) = ei;
            } else {
                FE(f, k) = it->second;
            }
        }
    }

    const int ne = num_edges();

    // Build vertex-edge adjacency.
    VE.assign(nv, {});
    for (int e = 0; e < ne; ++e) {
        VE[edges[e].v0].push_back(e);
        VE[edges[e].v1].push_back(e);
    }

    // Build edge-face adjacency.
    edge_faces.assign(ne, {-1, -1});
    for (int f = 0; f < nf; ++f) {
        for (int k = 0; k < 3; ++k) {
            int ei = FE(f, k);
            if (edge_faces[ei][0] == -1) {
                edge_faces[ei][0] = f;
            } else {
                edge_faces[ei][1] = f;
            }
        }
    }

    // Boundary flags.
    is_boundary_edge.assign(ne, false);
    is_boundary_vertex.assign(nv, false);
    for (int e = 0; e < ne; ++e) {
        if (edge_faces[e][1] == -1) {
            is_boundary_edge[e] = true;
            is_boundary_vertex[edges[e].v0] = true;
            is_boundary_vertex[edges[e].v1] = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

int TriangleMesh::find_edge(int v0, int v1) const {
    auto it = edge_map_.find(edge_key(v0, v1));
    return (it != edge_map_.end()) ? it->second : -1;
}

int TriangleMesh::local_index(int f, int v) const {
    for (int k = 0; k < 3; ++k) {
        if (F(f, k) == v) return k;
    }
    return -1;
}

double TriangleMesh::tip_angle(int f, int k) const {
    Eigen::Vector3d pi = V.row(F(f, k));
    Eigen::Vector3d pj = V.row(F(f, (k + 1) % 3));
    Eigen::Vector3d pk = V.row(F(f, (k + 2) % 3));

    Eigen::Vector3d u = (pj - pi).normalized();
    Eigen::Vector3d v = (pk - pi).normalized();

    double d = u.dot(v);
    // Clamp for numerical safety.
    d = std::max(-1.0, std::min(1.0, d));
    return std::acos(d);
}

double TriangleMesh::face_area(int f) const {
    Eigen::Vector3d pi = V.row(F(f, 0));
    Eigen::Vector3d pj = V.row(F(f, 1));
    Eigen::Vector3d pk = V.row(F(f, 2));
    Eigen::Vector3d e1 = pj - pi;
    Eigen::Vector3d e2 = pk - pi;
    return 0.5 * e1.cross(e2).norm();
}

double TriangleMesh::edge_length(int e) const {
    Eigen::Vector3d d = V.row(edges[e].v1) - V.row(edges[e].v0);
    return d.norm();
}

double TriangleMesh::edge_length_verts(int v0, int v1) const {
    Eigen::Vector3d d = V.row(v1) - V.row(v0);
    return d.norm();
}

Eigen::Vector3d TriangleMesh::face_normal(int f) const {
    Eigen::Vector3d pi = V.row(F(f, 0));
    Eigen::Vector3d pj = V.row(F(f, 1));
    Eigen::Vector3d pk = V.row(F(f, 2));
    Eigen::Vector3d e1 = pj - pi;
    Eigen::Vector3d e2 = pk - pi;
    Eigen::Vector3d n = e1.cross(e2);
    double len = n.norm();
    return (len > 1e-15) ? (n / len).eval() : Eigen::Vector3d::UnitZ();
}

Eigen::Vector3d TriangleMesh::vertex_normal(int v) const {
    Eigen::Vector3d n = Eigen::Vector3d::Zero();
    for (int f : VF[v]) {
        double a = face_area(f);
        n += a * face_normal(f);
    }
    double len = n.norm();
    return (len > 1e-15) ? (n / len).eval() : Eigen::Vector3d::UnitZ();
}

double TriangleMesh::total_area() const {
    double area = 0.0;
    for (int f = 0; f < num_faces(); ++f) {
        area += face_area(f);
    }
    return area;
}

int TriangleMesh::num_boundary_loops() const {
    // Count boundary edges, then trace loops.
    std::set<int> boundary_verts;
    std::unordered_map<int, int> next_boundary; // v -> next boundary vertex

    for (int e = 0; e < num_edges(); ++e) {
        if (!is_boundary_edge[e]) continue;

        int va = edges[e].v0;
        int vb = edges[e].v1;

        // Determine orientation from the single adjacent face.
        int f = edge_faces[e][0];
        int la = local_index(f, va);
        int lb = local_index(f, vb);

        // The boundary edge should be traversed opposite to the face
        // orientation. In face f, the edge goes from F(f,(k+1)%3) to
        // F(f,(k+2)%3) for the edge opposite vertex k. The boundary
        // traversal is the reverse.
        // Simpler: if va appears before vb in the face winding, the
        // boundary goes vb -> va.
        if ((la + 1) % 3 == lb) {
            // face order is va, vb => boundary goes vb -> va
            next_boundary[vb] = va;
        } else {
            next_boundary[va] = vb;
        }

        boundary_verts.insert(va);
        boundary_verts.insert(vb);
    }

    if (boundary_verts.empty()) return 0;

    // Trace loops.
    std::set<int> visited;
    int loops = 0;
    for (int start : boundary_verts) {
        if (visited.count(start)) continue;
        int cur = start;
        while (!visited.count(cur)) {
            visited.insert(cur);
            auto it = next_boundary.find(cur);
            if (it == next_boundary.end()) break;
            cur = it->second;
        }
        ++loops;
    }
    return loops;
}

} // namespace hatching
