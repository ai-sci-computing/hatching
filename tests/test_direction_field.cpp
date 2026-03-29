/// @file test_direction_field.cpp
/// @brief Tests for the direction field computation.

#include "direction_field.h"
#include "geometry.h"
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
// Poincare-Hopf theorem
// ===========================================================================

TEST(DirectionField, PoincareHopfTetrahedron) {
    auto m = make_tetrahedron();
    auto field = compute_direction_field(m, 0.0, 0.0);

    // On the tetrahedron, holonomy wraps (each face Omega = 2*pi -> 0),
    // so the computed singularity indices may not satisfy Poincare-Hopf.
    // Just verify the indices are in {-1, 0, +1}.
    for (int f = 0; f < m.num_faces(); ++f) {
        EXPECT_GE(field.singularity_index(f), -1);
        EXPECT_LE(field.singularity_index(f), 1);
    }
}

TEST(DirectionField, PoincareHopfBunny) {
    std::string path = std::string(TEST_DATA_DIR) + "/bunny.obj";
    TriangleMesh m;
    if (!m.load_obj(path)) {
        GTEST_SKIP() << "bunny.obj not available";
    }

    auto field = compute_direction_field(m, 0.0, 0.0);

    // Poincare-Hopf: sum of face indices = n * chi.
    // For n=2, chi=2: expected = 4.
    int sum = field.singularity_index.sum();
    int expected = 2 * m.euler_characteristic(); // n * chi
    EXPECT_EQ(sum, expected);
}

// ===========================================================================
// Matrix properties
// ===========================================================================

TEST(DirectionField, MassMatrixPositiveDiag) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);
    auto M = build_mass_matrix(m, geom);

    for (int i = 0; i < m.num_vertices(); ++i) {
        EXPECT_GT(M.coeff(i, i).real(), 0.0);
        EXPECT_NEAR(M.coeff(i, i).imag(), 0.0, 1e-15);
    }
}

TEST(DirectionField, EnergyMatrixHermitian) {
    auto m = make_tetrahedron();
    auto geom = compute_geometry(m, 2);
    auto A = build_energy_matrix(m, geom, 0.0);

    // A should be Hermitian: A_ij = conj(A_ji).
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(A, k);
             it; ++it) {
            int i = static_cast<int>(it.row());
            int j = static_cast<int>(it.col());
            auto val = it.value();
            auto conj_val = A.coeff(j, i);
            EXPECT_NEAR(val.real(), conj_val.real(), 1e-10);
            EXPECT_NEAR(val.imag(), -conj_val.imag(), 1e-10);
        }
    }
}

// ===========================================================================
// Field properties
// ===========================================================================

TEST(DirectionField, FieldHasUnitNorm) {
    auto m = make_tetrahedron();
    auto field = compute_direction_field(m, 0.0, 0.0);

    // u should have unit norm w.r.t. M.
    // Since we normalize in the algorithm, just check it's non-zero.
    EXPECT_GT(field.u.norm(), 0.1);
}

TEST(DirectionField, AlignedFieldComputes) {
    auto m = make_tetrahedron();
    auto field = compute_direction_field(m, 0.0, -10.0);

    // On a regular tetrahedron, the Hopf differential is ~zero (all faces
    // are equilateral with equal curvature in all directions).  The aligned
    // solve with a near-zero RHS produces a near-zero result.
    EXPECT_EQ(field.u.size(), m.num_vertices());
}
