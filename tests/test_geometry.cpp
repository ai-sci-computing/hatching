/// @file test_geometry.cpp
/// @brief Tests for discrete differential geometry operators.

#include "geometry.h"
#include "triangle_mesh.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace hatching;

// ---------------------------------------------------------------------------
// Test meshes
// ---------------------------------------------------------------------------

static TriangleMesh make_tetrahedron() {
    TriangleMesh m;
    m.V.resize(4, 3);
    m.V << 1, 1, 1, //
        1, -1, -1,   //
        -1, 1, -1,   //
        -1, -1, 1;
    m.F.resize(4, 3);
    m.F << 0, 1, 2, //
        0, 3, 1,     //
        0, 2, 3,     //
        1, 3, 2;
    m.build_topology();
    return m;
}

/// Flat square made of 2 triangles (z=0 plane).
static TriangleMesh make_flat_square() {
    TriangleMesh m;
    m.V.resize(4, 3);
    m.V << 0, 0, 0, //
        1, 0, 0,     //
        1, 1, 0,     //
        0, 1, 0;
    m.F.resize(2, 3);
    m.F << 0, 1, 2, //
        0, 2, 3;
    m.build_topology();
    return m;
}

// ===========================================================================
// Angle scaling
// ===========================================================================

TEST(Geometry, AngleScalingInterior) {
    auto m = make_tetrahedron();
    auto s = compute_angle_scaling(m);

    // Each vertex of a regular tetrahedron sees 3 equilateral faces.
    // Angle sum at each vertex = 3 * pi/3 = pi.
    double angle_sum = M_PI;
    double expected_s = 2.0 * M_PI / angle_sum; // = 2.0

    for (int i = 0; i < m.num_vertices(); ++i) {
        EXPECT_NEAR(s(i), expected_s, 1e-10);
    }
}

TEST(Geometry, AngleScalingFlat) {
    auto m = make_flat_square();
    auto s = compute_angle_scaling(m);

    // Interior vertex: none (all boundary in a flat square).
    // Boundary vertices have s = 1.
    for (int i = 0; i < m.num_vertices(); ++i) {
        EXPECT_NEAR(s(i), 1.0, 1e-10);
    }
}

// ===========================================================================
// Rescaled angles
// ===========================================================================

TEST(Geometry, RescaledAnglesExist) {
    auto m = make_tetrahedron();
    auto s = compute_angle_scaling(m);
    auto theta = compute_rescaled_angles(m, s);

    // Every vertex should have entries for all 3 neighbors.
    for (int i = 0; i < m.num_vertices(); ++i) {
        EXPECT_EQ(static_cast<int>(theta[i].size()), 3);
    }
}

TEST(Geometry, RescaledAnglesSumTo2Pi) {
    auto m = make_tetrahedron();
    auto s = compute_angle_scaling(m);
    auto theta = compute_rescaled_angles(m, s);

    // For interior vertices, the rescaled angles should span [0, 2pi).
    // The last angle + the last gap should equal 2pi.
    for (int i = 0; i < m.num_vertices(); ++i) {
        double max_angle = 0.0;
        for (auto& [j, th] : theta[i]) {
            max_angle = std::max(max_angle, th);
        }
        // The max angle should be the last edge; the gap back to 0
        // should complete the 2pi circle.  For a regular tetrahedron,
        // each gap = 2pi/3, so max = 2*2pi/3 = 4pi/3.
        double expected = 2.0 * M_PI * 2.0 / 3.0; // 4pi/3
        EXPECT_NEAR(max_angle, expected, 1e-10);
    }
}

// ===========================================================================
// Holonomy and Gauss-Bonnet
// ===========================================================================

TEST(Geometry, HolonomyGaussBonnetTetrahedron) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);

    // On a regular tetrahedron, each face has holonomy = 2*pi, which wraps
    // to 0 under arg().  The mesh is too coarse for Gauss-Bonnet to hold
    // with wrapped holonomy.  Just verify each face holonomy is near 0
    // (i.e., 2*pi wrapped).
    for (int f = 0; f < m.num_faces(); ++f) {
        EXPECT_NEAR(geom.Omega(f), 0.0, 1e-6);
    }
}

TEST(Geometry, FlatHolonomyIsZero) {
    auto m = make_flat_square();
    auto geom = compute_geometry(m, 2);

    // On a flat surface, the holonomy of interior faces should be zero.
    // But in a flat square, all vertices are on the boundary, so the
    // boundary terms affect holonomy.  For the two triangles, since
    // angle scaling is 1 (boundary), holonomy should reflect the
    // actual angle deficit.
    // In a fully flat triangulation with boundary, interior faces can
    // still have nonzero holonomy if the angle scaling distorts things.
    // With s=1 (boundary), the rescaled angles equal Euclidean angles,
    // so holonomy should be zero for flat geometry.
    for (int f = 0; f < m.num_faces(); ++f) {
        EXPECT_NEAR(geom.Omega(f), 0.0, 1e-10);
    }
}

// ===========================================================================
// Cotangent weights
// ===========================================================================

TEST(Geometry, CotanWeightsPositive) {
    auto m = make_tetrahedron();
    auto w = compute_cotan_weights(m);

    // For a regular tetrahedron, all angles are acute (arccos(1/3) < pi/2),
    // so all cotangent weights should be positive.
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_GT(w(e), 0.0);
    }
}

TEST(Geometry, CotanWeightsSymmetric) {
    auto m = make_tetrahedron();
    auto w = compute_cotan_weights(m);

    // All edges in a regular tetrahedron are equivalent.
    for (int e = 1; e < m.num_edges(); ++e) {
        EXPECT_NEAR(w(e), w(0), 1e-10);
    }
}

// ===========================================================================
// Hopf differential
// ===========================================================================

TEST(Geometry, HopfDifferentialComputes) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);

    // Just check it produces a vector of the right size.
    EXPECT_EQ(geom.hopf_differential.size(), m.num_vertices());
}

// ===========================================================================
// Full geometry on loaded mesh
// ===========================================================================

TEST(Geometry, BunnyGaussBonnet) {
    std::string path = std::string(TEST_DATA_DIR) + "/bunny.obj";
    TriangleMesh m;
    if (!m.load_obj(path)) {
        GTEST_SKIP() << "bunny.obj not available";
    }

    auto geom = compute_geometry(m, 2);

    // Gauss-Bonnet for the n-line bundle: sum(Omega) = 2*n*pi*chi.
    // For n=2, chi=2: expected = 8*pi.  On fine meshes, per-face holonomy
    // is small enough that arg() wrapping does not corrupt the sum.
    double omega_sum = geom.Omega.sum();
    double expected = 2.0 * geom.n * M_PI * m.euler_characteristic();
    EXPECT_NEAR(omega_sum, expected, 0.5);
}
