/// @file test_mesh.cpp
/// @brief Tests for TriangleMesh: topology, geometry queries, OBJ loading.

#include "triangle_mesh.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace hatching;

// ---------------------------------------------------------------------------
// Helper: build a regular tetrahedron.
// ---------------------------------------------------------------------------
static TriangleMesh make_tetrahedron() {
    TriangleMesh m;
    m.V.resize(4, 3);
    m.V << 1, 1, 1, //
        1, -1, -1,   //
        -1, 1, -1,   //
        -1, -1, 1;

    m.F.resize(4, 3);
    // CCW orientation (outward-facing normals).
    m.F << 0, 1, 2, //
        0, 3, 1,     //
        0, 2, 3,     //
        1, 3, 2;

    m.build_topology();
    return m;
}

// ---------------------------------------------------------------------------
// Helper: build a single triangle.
// ---------------------------------------------------------------------------
static TriangleMesh make_single_triangle() {
    TriangleMesh m;
    m.V.resize(3, 3);
    m.V << 0, 0, 0, //
        1, 0, 0,     //
        0, 1, 0;

    m.F.resize(1, 3);
    m.F << 0, 1, 2;

    m.build_topology();
    return m;
}

// ===========================================================================
// Topology tests
// ===========================================================================

TEST(MeshTopology, TetrahedronCounts) {
    auto m = make_tetrahedron();
    EXPECT_EQ(m.num_vertices(), 4);
    EXPECT_EQ(m.num_faces(), 4);
    EXPECT_EQ(m.num_edges(), 6);
}

TEST(MeshTopology, TetrahedronEuler) {
    auto m = make_tetrahedron();
    // Closed genus-0 surface: chi = 2.
    EXPECT_EQ(m.euler_characteristic(), 2);
}

TEST(MeshTopology, TetrahedronNoBoundary) {
    auto m = make_tetrahedron();
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_FALSE(m.is_boundary_edge[e]);
    }
    for (int v = 0; v < m.num_vertices(); ++v) {
        EXPECT_FALSE(m.is_boundary_vertex[v]);
    }
    EXPECT_EQ(m.num_boundary_loops(), 0);
}

TEST(MeshTopology, SingleTriangleCounts) {
    auto m = make_single_triangle();
    EXPECT_EQ(m.num_vertices(), 3);
    EXPECT_EQ(m.num_faces(), 1);
    EXPECT_EQ(m.num_edges(), 3);
    EXPECT_EQ(m.euler_characteristic(), 1);
}

TEST(MeshTopology, SingleTriangleBoundary) {
    auto m = make_single_triangle();
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_TRUE(m.is_boundary_edge[e]);
    }
    EXPECT_EQ(m.num_boundary_loops(), 1);
}

TEST(MeshTopology, FindEdge) {
    auto m = make_tetrahedron();
    // Every pair of vertices in a tetrahedron is connected.
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_GE(m.find_edge(i, j), 0);
            EXPECT_EQ(m.find_edge(i, j), m.find_edge(j, i));
        }
    }
    // Non-existent edge (vertex out of range check).
    EXPECT_EQ(m.find_edge(0, 100), -1);
}

TEST(MeshTopology, EdgeFaceAdjacency) {
    auto m = make_tetrahedron();
    // Every edge in a closed tetrahedron has exactly 2 adjacent faces.
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_GE(m.edge_faces[e][0], 0);
        EXPECT_GE(m.edge_faces[e][1], 0);
    }
}

TEST(MeshTopology, VertexFaceAdjacency) {
    auto m = make_tetrahedron();
    // Every vertex in a tetrahedron is incident to 3 faces.
    for (int v = 0; v < m.num_vertices(); ++v) {
        EXPECT_EQ(static_cast<int>(m.VF[v].size()), 3);
    }
}

TEST(MeshTopology, FaceEdgeConsistency) {
    auto m = make_tetrahedron();
    // FE(f, k) is the edge opposite vertex F(f, k).
    for (int f = 0; f < m.num_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            int e = m.FE(f, k);
            int a = m.F(f, (k + 1) % 3);
            int b = m.F(f, (k + 2) % 3);
            // The edge should connect the two non-k vertices.
            int v0 = m.edges[e].v0;
            int v1 = m.edges[e].v1;
            EXPECT_TRUE((v0 == std::min(a, b) && v1 == std::max(a, b)));
        }
    }
}

// ===========================================================================
// Geometry tests
// ===========================================================================

TEST(MeshGeometry, TetrahedronTipAngles) {
    auto m = make_tetrahedron();
    // Regular tetrahedron with equilateral faces: all tip angles = pi/3.
    double expected = M_PI / 3.0;
    for (int f = 0; f < m.num_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(m.tip_angle(f, k), expected, 1e-10);
        }
    }
}

TEST(MeshGeometry, TetrahedronAngleSum) {
    auto m = make_tetrahedron();
    // Sum of angles in each face = pi.
    for (int f = 0; f < m.num_faces(); ++f) {
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) {
            sum += m.tip_angle(f, k);
        }
        EXPECT_NEAR(sum, M_PI, 1e-10);
    }
}

TEST(MeshGeometry, TetrahedronFaceArea) {
    auto m = make_tetrahedron();
    // Edge length = 2*sqrt(2), area = sqrt(3)/4 * (2*sqrt(2))^2 = 2*sqrt(3).
    double expected = 2.0 * std::sqrt(3.0);
    for (int f = 0; f < m.num_faces(); ++f) {
        EXPECT_NEAR(m.face_area(f), expected, 1e-10);
    }
}

TEST(MeshGeometry, TetrahedronEdgeLengths) {
    auto m = make_tetrahedron();
    double expected = 2.0 * std::sqrt(2.0);
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_NEAR(m.edge_length(e), expected, 1e-10);
    }
}

TEST(MeshGeometry, SingleTriangleArea) {
    auto m = make_single_triangle();
    EXPECT_NEAR(m.face_area(0), 0.5, 1e-10);
}

TEST(MeshGeometry, TotalArea) {
    auto m = make_tetrahedron();
    double expected = 4.0 * 2.0 * std::sqrt(3.0);
    EXPECT_NEAR(m.total_area(), expected, 1e-10);
}

TEST(MeshGeometry, FaceNormalsUnitLength) {
    auto m = make_tetrahedron();
    for (int f = 0; f < m.num_faces(); ++f) {
        EXPECT_NEAR(m.face_normal(f).norm(), 1.0, 1e-10);
    }
}

TEST(MeshGeometry, VertexNormalsUnitLength) {
    auto m = make_tetrahedron();
    for (int v = 0; v < m.num_vertices(); ++v) {
        EXPECT_NEAR(m.vertex_normal(v).norm(), 1.0, 1e-10);
    }
}

// ===========================================================================
// OBJ loading tests
// ===========================================================================

TEST(MeshLoading, LoadBunny) {
    std::string path = std::string(TEST_DATA_DIR) + "/bunny.obj";
    TriangleMesh m;
    bool ok = m.load_obj(path);
    ASSERT_TRUE(ok);
    EXPECT_GT(m.num_vertices(), 0);
    EXPECT_GT(m.num_faces(), 0);
    EXPECT_GT(m.num_edges(), 0);
    // Bunny should be a closed genus-0 surface.
    EXPECT_EQ(m.euler_characteristic(), 2);
}

TEST(MeshLoading, NonExistentFile) {
    TriangleMesh m;
    EXPECT_FALSE(m.load_obj("nonexistent_file.obj"));
}
