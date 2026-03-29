/// @file test_stripe_pattern.cpp
/// @brief Tests for the stripe pattern computation.

#include "direction_field.h"
#include "geometry.h"
#include "stripe_pattern.h"
#include "triangle_mesh.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace hatching;

static TriangleMesh make_tetrahedron() {
    TriangleMesh m;
    m.V.resize(4, 3);
    m.V << 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1;
    m.F.resize(4, 3);
    m.F << 0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2;
    m.build_topology();
    return m;
}

// ===========================================================================
// lArg interpolant
// ===========================================================================

TEST(StripePattern, LArgZeroIndex) {
    // n=0 should return 0.
    EXPECT_DOUBLE_EQ(lArg(0, 0.5, 0.3, 0.2), 0.0);
}

TEST(StripePattern, LArgContinuousAtBarycenter) {
    // At the barycenter (1/3, 1/3, 1/3), lArg should be well-defined
    // but the formula has a removable singularity.  For n=1, the value
    // should be pi*n/3 * k for some sector.
    // Just check it doesn't crash.
    double val = lArg(1, 1.0 / 3.0 + 1e-10, 1.0 / 3.0, 1.0 / 3.0 - 1e-10);
    EXPECT_TRUE(std::isfinite(val));
}

TEST(StripePattern, LArgBarycentricCorners) {
    // At corners, verify the lArg values match the formula.
    // For n=1: lArg(1, 1, 0, 0): tk=0 is smallest =>
    //   pi/3 * (1 + (1-0)/(1-0)) = pi/3 * 2 = 2pi/3
    // lArg(1, 0, 1, 0): tk=0 is smallest (tie with ti=0, but tk<=ti holds)
    //   pi/3 * (1 + (0-1)/(1-0)) = pi/3 * 0 = 0
    // lArg(1, 0, 0, 1): ti=0 is smallest =>
    //   pi/3 * (3 + (0-1)/(1-0)) = pi/3 * 2 = 2pi/3
    EXPECT_NEAR(lArg(1, 1.0, 0.0, 0.0), 2.0 * M_PI / 3.0, 1e-10);
    EXPECT_NEAR(lArg(1, 0.0, 1.0, 0.0), 0.0, 1e-10);
    EXPECT_NEAR(lArg(1, 0.0, 0.0, 1.0), 2.0 * M_PI / 3.0, 1e-10);
}

// ===========================================================================
// Edge data
// ===========================================================================

TEST(StripePattern, EdgeDataComputes) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);
    auto field = compute_direction_field(m, 0.0, 0.0);

    Eigen::VectorXd freq =
        Eigen::VectorXd::Constant(m.num_vertices(), 20.0);
    auto [omega, s] = compute_edge_data(m, geom, field.u, freq);

    EXPECT_EQ(omega.size(), m.num_edges());
    EXPECT_EQ(s.size(), m.num_edges());

    // All s values should be +1 or -1.
    for (int e = 0; e < m.num_edges(); ++e) {
        EXPECT_TRUE(s(e) == 1 || s(e) == -1);
    }
}

// ===========================================================================
// Full pipeline
// ===========================================================================

TEST(StripePattern, FullPipelineTetrahedron) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);
    auto field = compute_direction_field(m, 0.0, 0.0);
    auto pattern = compute_stripe_pattern(m, field, geom, 20.0);

    // Should produce texture coordinates for every face corner.
    EXPECT_EQ(pattern.alpha.rows(), m.num_faces());
    EXPECT_EQ(pattern.alpha.cols(), 3);

    // All alpha values should be finite.
    for (int f = 0; f < m.num_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            EXPECT_TRUE(std::isfinite(pattern.alpha(f, k)));
        }
    }
}

TEST(StripePattern, FullPipelineBunny) {
    std::string path = std::string(TEST_DATA_DIR) + "/bunny.obj";
    TriangleMesh m;
    if (!m.load_obj(path)) {
        GTEST_SKIP() << "bunny.obj not available";
    }

    auto geom = compute_geometry(m, 2);
    auto field = compute_direction_field(m, 0.0, 0.0);
    auto pattern = compute_stripe_pattern(m, field, geom, 20.0);

    EXPECT_EQ(pattern.alpha.rows(), m.num_faces());
    EXPECT_EQ(pattern.alpha.cols(), 3);

    // Spot-check: at least some non-zero values.
    double alpha_range = pattern.alpha.maxCoeff() - pattern.alpha.minCoeff();
    EXPECT_GT(alpha_range, 1.0);
}

// ===========================================================================
// Stripe energy matrix properties
// ===========================================================================

TEST(StripePattern, EnergyMatrixSymmetric) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);
    auto field = compute_direction_field(m, 0.0, 0.0);

    Eigen::VectorXd freq =
        Eigen::VectorXd::Constant(m.num_vertices(), 20.0);
    auto [omega, s] = compute_edge_data(m, geom, field.u, freq);
    auto A = build_stripe_energy_matrix(m, omega, s);

    // A should be symmetric.
    Eigen::SparseMatrix<double> diff = A - Eigen::SparseMatrix<double>(A.transpose());
    EXPECT_NEAR(diff.norm(), 0.0, 1e-10);
}
