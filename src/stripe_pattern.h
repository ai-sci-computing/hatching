#pragma once

/// @file stripe_pattern.h
/// @brief Stripe patterns on surfaces (Knoeppel et al. 2015).
///
/// Given a direction field and target line frequency on a triangle mesh,
/// computes per-triangle-corner texture coordinates that produce a globally
/// continuous stripe pattern.  Singularities (zeros of the wave function and
/// branch points of the double cover) are handled via the lArg interpolant
/// and barycentric subdivision.

#include "direction_field.h"
#include "geometry.h"
#include "triangle_mesh.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace hatching {

/// @brief Result of the stripe pattern computation.
struct StripePattern {
    /// @brief Texture coordinate alpha at each triangle corner.
    /// alpha(f, k) is the coordinate at corner k of face f.
    /// Plugging into cos(alpha) yields the stripe pattern.
    Eigen::MatrixXd alpha; // |F| x 3

    /// @brief Per-face index n_ijk (winding number around zeros).
    Eigen::VectorXi face_index;

    /// @brief Debug: raw holonomy/(2π) before rounding.
    Eigen::VectorXd face_index_raw;

    /// @brief Per-face flag: true if the face is a branch triangle
    ///        (needs barycentric subdivision for rendering).
    std::vector<bool> is_branch_triangle;

    /// @brief For branch triangles: texture coordinates at the midpoint m
    ///        and the duplicate vertex l. Indexed the same as alpha for
    ///        branch faces. beta_m(f) = midpoint coord, beta_l(f, k) =
    ///        subdivided corner coords.
    Eigen::MatrixXd alpha_subdiv; // |F| x 7 (3 corners + midpoint + 3 sub)
};

/// @brief Compute edge data for the stripe pattern algorithm (Algorithm 3).
///
/// Projects the direction field onto mesh edges to obtain angular
/// displacements omega_ij, and computes sign indicators s_ij for the
/// double cover.
///
/// @param mesh Input mesh.
/// @param geom Precomputed geometry.
/// @param u Direction field coefficients.
/// @param frequency Per-vertex target line frequency nu_i.
/// @return Tuple of (omega, s) vectors indexed by edge.
std::pair<Eigen::VectorXd, Eigen::VectorXi>
compute_edge_data(const TriangleMesh& mesh, const MeshGeometry& geom,
                  const Eigen::VectorXcd& u,
                  const Eigen::VectorXd& frequency);

/// @brief Build the energy matrix for the stripe pattern (Algorithm 4).
///
/// This is a real symmetric positive-definite matrix of size 2|V| x 2|V|,
/// encoding the conjugate-symmetric energy on the implicit double cover.
Eigen::SparseMatrix<double>
build_stripe_energy_matrix(const TriangleMesh& mesh,
                           const Eigen::VectorXd& omega,
                           const Eigen::VectorXi& s);

/// @brief Build the lumped mass matrix for the stripe pattern (Algorithm 5).
Eigen::SparseMatrix<double>
build_stripe_mass_matrix(const TriangleMesh& mesh);

/// @brief Find the principal eigenvector via inverse power iteration
///        (Algorithm 6).
/// @param A Energy matrix (2|V| x 2|V|).
/// @param B Mass matrix (2|V| x 2|V|).
/// @param num_iterations Number of power iterations.
/// @return Eigenvector x in R^{2|V|}, interpreted as |V| complex numbers.
Eigen::VectorXcd compute_stripe_field(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B,
    int num_iterations = 20);

/// @brief Extract texture coordinates from the stripe field (Algorithm 7).
///
/// Handles regular triangles (linear interpolation), zero-index triangles
/// (lArg interpolant, Eq. 11), and branch triangles (barycentric
/// subdivision, Sec. 4.3).
StripePattern compute_texture_coordinates(
    const TriangleMesh& mesh, const MeshGeometry& geom,
    const Eigen::VectorXcd& psi, const Eigen::VectorXd& omega,
    const Eigen::VectorXi& s);

/// @brief Full pipeline: compute stripe pattern from a direction field.
/// @param mesh Input mesh.
/// @param field Direction field (from compute_direction_field).
/// @param geom Precomputed geometry.
/// @param frequency Uniform target frequency (stripes per unit length).
/// @return StripePattern with texture coordinates.
StripePattern compute_stripe_pattern(const TriangleMesh& mesh,
                                      const DirectionField& field,
                                      const MeshGeometry& geom,
                                      double frequency = 20.0,
                                      bool use_psi_one = false);

/// @brief The lArg interpolant for singularity resolution (Eq. 11).
///
/// Maps barycentric coordinates (t_i, t_j, t_k) near a zero singularity
/// to a continuous angle, resolving the 2*pi*n winding.
/// @param n The winding number (face index).
/// @param ti, tj, tk Barycentric coordinates.
/// @return Angle correction to subtract from linearly interpolated alpha.
double lArg(int n, double ti, double tj, double tk);

} // namespace hatching
