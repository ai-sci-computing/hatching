#pragma once

/// @file direction_field.h
/// @brief Globally optimal direction fields (Knoeppel et al. 2013).
///
/// Computes smooth n-direction fields on triangle meshes, optionally aligned
/// with principal curvature directions.  The smoothest field is found via a
/// sparse eigenvalue problem; curvature alignment is obtained by solving a
/// single sparse linear system.

#include "geometry.h"
#include "triangle_mesh.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace hatching {

/// @brief Result of the direction field computation.
struct DirectionField {
    /// @brief Complex coefficients u_i of the n-vector field at each vertex.
    /// The actual direction at vertex i (relative to the local reference
    /// frame) is recovered as arg(u_i) / n.
    Eigen::VectorXcd u;

    /// @brief Per-face singularity index (-1, 0, or +1).
    Eigen::VectorXi singularity_index;
};

/// @brief Build the Hermitian mass matrix M (Eq. 17, Knoeppel 2013).
///
/// M is |V| x |V| Hermitian (stored as complex-valued sparse matrix).
/// Diagonal: M_ii = |t_ijk| / 6  summed over incident triangles.
/// Off-diagonal: M_jk uses the closed-form expression involving holonomy.
Eigen::SparseMatrix<std::complex<double>>
build_mass_matrix(const TriangleMesh& mesh, const MeshGeometry& geom);

/// @brief Build the energy matrix A (Eq. 18, Knoeppel 2013).
///
/// A is |V| x |V| Hermitian. Encodes the smoothness energy E_s as a
/// quadratic form: E_s(psi) = 0.5 * u^H A u.
/// @param s Smoothness parameter in (-1, 1). s=0: Dirichlet, s=1:
///          holomorphic, s=-1: anti-holomorphic.
Eigen::SparseMatrix<std::complex<double>>
build_energy_matrix(const TriangleMesh& mesh, const MeshGeometry& geom,
                    double s);

/// @brief Compute the smoothest n-direction field (Algorithm 2).
///
/// Finds the smallest eigenvector of A u = lambda M u via inverse power
/// iteration with Cholesky factorization.
/// @param A Energy matrix.
/// @param M Mass matrix.
/// @param num_iterations Number of power iterations (default 20).
/// @return Complex coefficient vector u (unit norm w.r.t. M).
Eigen::VectorXcd compute_smoothest_field(
    const Eigen::SparseMatrix<std::complex<double>>& A,
    const Eigen::SparseMatrix<std::complex<double>>& M,
    int num_iterations = 20);

/// @brief Compute a curvature-aligned direction field (Algorithm 3).
///
/// Solves (A - lambda_t * M) u_tilde = M * q for the aligned field, then
/// normalizes.  lambda_t controls the alignment strength (more negative =
/// stronger alignment).
/// @param A Energy matrix.
/// @param M Mass matrix.
/// @param q Guidance field (Hopf differential), |V| complex vector.
/// @param lambda_t Alignment parameter. 0 is a good starting value;
///                 more negative values increase alignment.
/// @return Complex coefficient vector u (unit norm w.r.t. M).
Eigen::VectorXcd compute_aligned_field(
    const Eigen::SparseMatrix<std::complex<double>>& A,
    const Eigen::SparseMatrix<std::complex<double>>& M,
    const Eigen::VectorXcd& q, double lambda_t);

/// @brief Compute per-face singularity indices (Section 6.1.3).
///
/// For each face, the index is in {-1, 0, +1}. Nonzero indices indicate
/// singularities.  By the discrete Poincare-Hopf theorem, the sum of all
/// indices equals 2 * n * chi, where chi is the Euler characteristic.
Eigen::VectorXi compute_singularity_indices(
    const TriangleMesh& mesh, const MeshGeometry& geom,
    const Eigen::VectorXcd& u);

/// @brief Full pipeline: compute a curvature-aligned direction field.
/// @param mesh Input mesh.
/// @param s Smoothness type parameter (default 0 = Dirichlet).
/// @param lambda_t Alignment strength (default 0).
/// @return DirectionField with coefficients and singularity indices.
DirectionField compute_direction_field(const TriangleMesh& mesh,
                                        double s = 0.0,
                                        double lambda_t = 0.0);

} // namespace hatching
