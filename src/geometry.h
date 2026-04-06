#pragma once

/// @file geometry.h
/// @brief Discrete differential geometry operators on triangle meshes.
///
/// Implements the geometric computations needed by both the direction field
/// algorithm (Knoeppel et al. 2013) and the stripe pattern algorithm
/// (Knoeppel et al. 2015): angle rescaling, parallel transport, holonomy,
/// curvature, cotangent weights, and the Hopf differential.

#include "triangle_mesh.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace hatching {

/// @brief Precomputed geometric data for a triangle mesh.
///
/// All quantities follow the notation of Knoeppel et al. 2013 (Globally
/// Optimal Direction Fields), Section 6.
struct MeshGeometry {
    int n = 2; ///< Degree of the n-direction field (2 = line field).

    // --- Per-vertex ---

    /// @brief Angle scaling factor s_i = 2pi / sum of incident tip angles.
    /// Equation (11) of Knoeppel 2013.
    Eigen::VectorXd angle_scaling;

    /// @brief Rescaled polar angle of each outgoing halfedge at each vertex.
    /// theta_ij[vertex] is a map from neighbor vertex j to the rescaled angle
    /// of the halfedge from vertex to j, measured CCW from the reference
    /// direction at vertex.
    ///
    /// The reference direction at vertex i is the direction toward the first
    /// neighbor in the CCW ordering (determined by the incident faces).
    std::vector<std::unordered_map<int, double>> theta;

    // --- Per-edge (indexed by edge index, canonical orientation v0 < v1) ---

    /// @brief Parallel transport angle rho_ij for the n-line bundle.
    /// rho_ij = n * (-theta_{ij} + theta_{ji} + pi).
    /// The transport coefficient is r_ij = exp(i * rho_ij).
    Eigen::VectorXd rho;

    // --- Per-face ---

    /// @brief Holonomy (curvature 2-form) per face.
    /// Omega_ijk = arg(r_ij * r_jk * r_ki). Equation (13).
    Eigen::VectorXd Omega;

    /// @brief Gaussian curvature per face.
    /// K_ijk = Omega_ijk / (n * |t_ijk|). Equation (14).
    Eigen::VectorXd K_face;

    // --- Per-edge: cotangent weights ---

    /// @brief Cotangent weight w_ij = (cot beta_ij + cot beta_ji) / 2.
    Eigen::VectorXd cotan_weights;

    // --- Curvature guidance field ---

    /// @brief Hopf differential q_i (complex) at each vertex.
    /// This is the PL approximation of the trace-free shape operator,
    /// serving as the guidance field for curvature alignment.
    /// Section 6.1.2 and Appendix D.5 of Knoeppel 2013.
    Eigen::VectorXcd hopf_differential;
};

/// @brief Compute all geometric quantities needed for the direction field
///        and stripe pattern algorithms.
/// @param mesh Input triangle mesh (must have topology built).
/// @param n Degree of the direction field (2 for line field).
/// @return Populated MeshGeometry structure.
MeshGeometry compute_geometry(const TriangleMesh& mesh, int n = 2);

/// @brief Compute angle scaling factors s_i (Eq. 11, Knoeppel 2013).
Eigen::VectorXd compute_angle_scaling(const TriangleMesh& mesh);

/// @brief Compute rescaled polar angles of outgoing halfedges at each vertex.
///
/// At each vertex, the outgoing halfedges are ordered CCW (by walking around
/// incident faces). The angle of each halfedge is rescaled so that the total
/// angle sum equals 2*pi.
std::vector<std::unordered_map<int, double>>
compute_rescaled_angles(const TriangleMesh& mesh,
                        const Eigen::VectorXd& angle_scaling);

/// @brief Compute parallel transport angles rho_ij (Eq. 12, Knoeppel 2013).
Eigen::VectorXd compute_transport_angles(
    const TriangleMesh& mesh,
    const std::vector<std::unordered_map<int, double>>& theta, int n);

/// @brief Compute holonomy per face (Eq. 13, Knoeppel 2013).
Eigen::VectorXd compute_holonomy(const TriangleMesh& mesh,
                                  const Eigen::VectorXd& rho);

/// @brief Compute cotangent weights per edge.
Eigen::VectorXd compute_cotan_weights(const TriangleMesh& mesh);

/// @brief Compute the Hopf differential (curvature guidance field).
/// Section 6.1.2 and Appendix D.5 of Knoeppel 2013.
Eigen::VectorXcd compute_hopf_differential(
    const TriangleMesh& mesh,
    const std::vector<std::unordered_map<int, double>>& theta,
    const Eigen::VectorXd& angle_scaling);

} // namespace hatching
