#pragma once

/// @file triangle_mesh.h
/// @brief Triangle mesh with topology and basic geometric queries.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace hatching {

/// @brief Oriented triangle mesh with edge list and adjacency structures.
///
/// Stores vertex positions (V) and face vertex indices (F, CCW-oriented).
/// After calling build_topology(), provides an explicit edge list with
/// canonical orientation (v0 < v1), per-face edge indices, vertex-face and
/// vertex-edge adjacency, and boundary flags.
class TriangleMesh {
public:
    // === Geometry ===
    Eigen::MatrixXd V; ///< Vertex positions, |V| x 3.
    Eigen::MatrixXi F; ///< Face vertex indices (CCW), |F| x 3.

    // === Topology (populated by build_topology()) ===

    /// @brief An edge defined by two vertex indices with v0 < v1.
    struct Edge {
        int v0, v1;
    };

    std::vector<Edge> edges;              ///< All mesh edges.
    Eigen::MatrixXi FE;                   ///< Per-face edge indices, |F| x 3.
                                           ///< FE(f, k) is the edge opposite
                                           ///< vertex F(f, k).
    std::vector<std::vector<int>> VF;     ///< Vertex -> incident face indices.
    std::vector<std::vector<int>> VE;     ///< Vertex -> incident edge indices.
    std::vector<std::array<int, 2>> edge_faces; ///< Edge -> adjacent faces.
                                                 ///< Second entry is -1 for
                                                 ///< boundary edges.
    std::vector<bool> is_boundary_vertex; ///< True for vertices on the boundary.
    std::vector<bool> is_boundary_edge;   ///< True for edges on the boundary.

    // === Construction ===

    /// @brief Load a mesh from an OBJ file.
    /// @param path Path to the .obj file.
    /// @return True on success.
    bool load_obj(const std::string& path);

    /// @brief Build edge list and adjacency from V and F.
    ///
    /// Must be called after V and F are populated (either via load_obj or
    /// by setting them directly). Populates edges, FE, VF, VE, edge_faces,
    /// and boundary flags.
    void build_topology();

    // === Queries ===

    int num_vertices() const { return static_cast<int>(V.rows()); }
    int num_faces() const { return static_cast<int>(F.rows()); }
    int num_edges() const { return static_cast<int>(edges.size()); }

    /// @brief Euler characteristic V - E + F.
    int euler_characteristic() const {
        return num_vertices() - num_edges() + num_faces();
    }

    /// @brief Find the edge index for the edge connecting v0 and v1.
    /// @return Edge index, or -1 if no such edge exists.
    int find_edge(int v0, int v1) const;

    /// @brief Local index (0, 1, or 2) of vertex v in face f.
    /// @return Local index, or -1 if v is not a vertex of f.
    int local_index(int f, int v) const;

    /// @brief Tip angle (in radians) at vertex F(f, k) in face f.
    double tip_angle(int f, int k) const;

    /// @brief Area of face f.
    double face_area(int f) const;

    /// @brief Length of edge e.
    double edge_length(int e) const;

    /// @brief Length of the edge connecting vertices v0 and v1.
    double edge_length_verts(int v0, int v1) const;

    /// @brief Unit normal of face f (CCW orientation).
    Eigen::Vector3d face_normal(int f) const;

    /// @brief Area-weighted average normal at vertex v.
    Eigen::Vector3d vertex_normal(int v) const;

    /// @brief Total surface area.
    double total_area() const;

    /// @brief Number of boundary loops (0 for closed meshes).
    int num_boundary_loops() const;

private:
    std::unordered_map<int64_t, int> edge_map_;

    static int64_t edge_key(int v0, int v1) {
        if (v0 > v1) std::swap(v0, v1);
        return (static_cast<int64_t>(v0) << 32) | static_cast<int64_t>(v1);
    }
};

} // namespace hatching
