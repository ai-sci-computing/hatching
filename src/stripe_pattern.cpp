/// @file stripe_pattern.cpp
/// @brief Implementation of stripe patterns on surfaces.
///
/// Follows the pseudocode in Appendix C of Knoeppel et al. 2015
/// ("Stripe Patterns on Surfaces") as closely as possible.

#include "stripe_pattern.h"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <complex>
#include <random>

namespace hatching {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Edge data (Algorithm 3, Knoeppel 2015)
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::VectorXi>
compute_edge_data(const TriangleMesh& mesh, const MeshGeometry& geom,
                  const Eigen::VectorXcd& u,
                  const Eigen::VectorXd& frequency) {
    const int ne = mesh.num_edges();
    Eigen::VectorXd omega(ne);
    Eigen::VectorXi s(ne);

    for (int e = 0; e < ne; ++e) {
        int i = mesh.edges[e].v0;
        int j = mesh.edges[e].v1;
        double len = mesh.edge_length(e);

        // Rescaled angles of the half-edges (from geometry, n-independent).
        double theta_ij = 0.0, theta_ji = 0.0;
        {
            auto it = geom.theta[i].find(j);
            if (it != geom.theta[i].end()) theta_ij = it->second;
        }
        {
            auto it = geom.theta[j].find(i);
            if (it != geom.theta[j].end()) theta_ji = it->second;
        }

        // Transport angle for 1-vector bundle: rho_ij = -theta_ij + theta_ji + pi.
        double rho_ij = -theta_ij + theta_ji + M_PI;

        // Direction field angle: phi = arg(u)/n.  For n=2 line field:
        double phi_i = std::arg(u(i)) / 2.0;
        double phi_j = std::arg(u(j)) / 2.0;

        // Sign s_ij (Eq. in Sec. 3.2): check if the direction field vectors
        // on both sides of the edge are consistently oriented.
        // s_ij = sgn(Re(e^{i*rho_ij} * X_i * conj(X_j)))
        // where X_i = e^{i*phi_i}.
        double transported_angle = phi_i + rho_ij;
        double diff = transported_angle - phi_j;
        // Normalize to [-pi, pi].
        diff = std::remainder(diff, 2.0 * M_PI);
        s(e) = (std::abs(diff) <= M_PI / 2.0) ? 1 : -1;

        // Adjusted phi_j: flip direction at j if s_ij = -1.
        double phi_j_eff = (s(e) == 1) ? phi_j : phi_j + M_PI;

        // Angular displacement omega_ij (Eq. 7): the integrated frequency
        // along the edge projected onto the direction field.
        // omega_ij = 0.5 * ell * (nu_i * cos(phi_i - theta_ij) +
        //                          nu_j * cos(phi_j_eff - theta_ji))
        omega(e) = 0.5 * len *
                   (frequency(i) * std::cos(phi_i - theta_ij) +
                    frequency(j) * std::cos(phi_j_eff - theta_ji));
    }

    return {omega, s};
}

// ---------------------------------------------------------------------------
// Energy matrix (Algorithm 4, Knoeppel 2015)
// ---------------------------------------------------------------------------

Eigen::SparseMatrix<double>
build_stripe_energy_matrix(const TriangleMesh& mesh,
                           const Eigen::VectorXd& omega,
                           const Eigen::VectorXi& s) {
    const int nv = mesh.num_vertices();
    const int n2 = 2 * nv;
    std::vector<Eigen::Triplet<double>> triplets;

    for (int e = 0; e < mesh.num_edges(); ++e) {
        int i = mesh.edges[e].v0;
        int j = mesh.edges[e].v1;

        // Cotangent weight: w = (cot beta_ij + cot beta_ji) / 2.
        double w = 0.0;
        for (int side = 0; side < 2; ++side) {
            int f = mesh.edge_faces[e][side];
            if (f < 0) continue;
            // Find the local index of the vertex opposite to this edge.
            for (int kk = 0; kk < 3; ++kk) {
                if (mesh.FE(f, kk) == e) {
                    double angle = mesh.tip_angle(f, kk);
                    w += 0.5 * std::cos(angle) / std::sin(angle);
                    break;
                }
            }
        }

        // The 2x2 block for the off-diagonal entry.
        // Paper's [·] notation: [z] = [Re z, Im z; -Im z, Re z].
        //
        // Algorithm 4:
        //   s_ij >= 0: A_ij = -w * [e^{i*omega}]
        //   s_ij <  0: A_ij = -w * [conj(e^{i*omega})]
        //
        // [e^{iw}]      = [ cos w,  sin w; -sin w, cos w]
        // [conj(e^{iw})]= [ cos w, -sin w;  sin w, cos w]

        double c = std::cos(omega(e));
        double sn = std::sin(omega(e));

        // Select the block: [e^{iw}] for s>=0, [conj(e^{iw})] for s<0.
        double block_sin = (s(e) >= 0) ? sn : -sn;

        // A_ij block: -w * [c, block_sin; -block_sin, c]
        triplets.push_back({i, j, -w * c});
        triplets.push_back({i, j + nv, -w * block_sin});
        triplets.push_back({i + nv, j, w * block_sin});
        triplets.push_back({i + nv, j + nv, -w * c});

        // A_ji = A_ij^T: -w * [c, -block_sin; block_sin, c]
        triplets.push_back({j, i, -w * c});
        triplets.push_back({j, i + nv, w * block_sin});
        triplets.push_back({j + nv, i, -w * block_sin});
        triplets.push_back({j + nv, i + nv, -w * c});

        // Diagonal: A_ii += w*I, A_jj += w*I.
        triplets.push_back({i, i, w});
        triplets.push_back({i + nv, i + nv, w});
        triplets.push_back({j, j, w});
        triplets.push_back({j + nv, j + nv, w});
    }

    Eigen::SparseMatrix<double> A(n2, n2);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// ---------------------------------------------------------------------------
// Mass matrix (Algorithm 5, Knoeppel 2015)
// ---------------------------------------------------------------------------

Eigen::SparseMatrix<double>
build_stripe_mass_matrix(const TriangleMesh& mesh) {
    const int nv = mesh.num_vertices();
    const int n2 = 2 * nv;
    std::vector<Eigen::Triplet<double>> triplets;

    for (int f = 0; f < mesh.num_faces(); ++f) {
        double area = mesh.face_area(f);
        for (int k = 0; k < 3; ++k) {
            int v = mesh.F(f, k);
            triplets.push_back({v, v, area / 3.0});
            triplets.push_back({v + nv, v + nv, area / 3.0});
        }
    }

    Eigen::SparseMatrix<double> B(n2, n2);
    B.setFromTriplets(triplets.begin(), triplets.end());
    return B;
}

// ---------------------------------------------------------------------------
// Principal eigenvector (Algorithm 6, Knoeppel 2015)
// ---------------------------------------------------------------------------

Eigen::VectorXcd compute_stripe_field(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::SparseMatrix<double>& B, int num_iterations) {
    const int n2 = static_cast<int>(A.rows());
    const int nv = n2 / 2;

    Eigen::SparseMatrix<double> A_reg = A;
    double eps = 1e-8;
    A_reg += eps * B;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_reg);
    if (solver.info() != Eigen::Success) {
        return Eigen::VectorXcd::Ones(nv);
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::VectorXd x(n2);
    for (int i = 0; i < n2; ++i) {
        x(i) = dist(rng);
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        Eigen::VectorXd Bx = B * x;
        x = solver.solve(Bx);
        double norm2 = x.dot(B * x);
        if (norm2 > 1e-30) {
            x /= std::sqrt(norm2);
        }
    }

    Eigen::VectorXcd psi(nv);
    for (int i = 0; i < nv; ++i) {
        psi(i) = Complex(x(i), x(i + nv));
    }
    return psi;
}

// ---------------------------------------------------------------------------
// lArg interpolant (Eq. 11, Knoeppel 2015)
// ---------------------------------------------------------------------------

double lArg(int n, double ti, double tj, double tk) {
    if (n == 0) return 0.0;

    double pi_n = M_PI * static_cast<double>(n);

    if (tk <= ti && tk <= tj) {
        return pi_n / 3.0 * (1.0 + (ti - tj) / (1.0 - 3.0 * tk));
    } else if (ti <= tj && ti <= tk) {
        return pi_n / 3.0 * (3.0 + (tj - tk) / (1.0 - 3.0 * ti));
    } else {
        return pi_n / 3.0 * (5.0 + (tk - ti) / (1.0 - 3.0 * tj));
    }
}

// ---------------------------------------------------------------------------
// Texture coordinates (Algorithm 7, Knoeppel 2015)
//
// This follows the paper's pseudocode closely.  For each face ijk:
// 1. Gather local copies of psi, omega, s.
// 2. Determine branch index S_ijk and adjust signs for the double cover.
// 3. Compute angles at triangle corners via the spinning form.
// 4. Compute face index n_ijk (winding number of psi around the face).
// 5. Adjust for zeros via lArg subtraction.
// ---------------------------------------------------------------------------

StripePattern compute_texture_coordinates(
    const TriangleMesh& mesh, const MeshGeometry& /*geom*/,
    const Eigen::VectorXcd& psi, const Eigen::VectorXd& omega,
    const Eigen::VectorXi& s) {
    const int nf = mesh.num_faces();
    StripePattern result;
    result.alpha.resize(nf, 3);
    result.face_index.resize(nf);
    result.is_branch_triangle.resize(nf, false);
    result.alpha_subdiv.resize(nf, 7);
    result.alpha_subdiv.setZero();

    for (int f = 0; f < nf; ++f) {
        int vi = mesh.F(f, 0);
        int vj = mesh.F(f, 1);
        int vk = mesh.F(f, 2);

        // Edge indices.
        int e_ij = mesh.find_edge(vi, vj);
        int e_jk = mesh.find_edge(vj, vk);
        int e_ki = mesh.find_edge(vk, vi);

        // Canonical orientation signs: c = +1 if edge stored as (a,b) with a<b
        // matches the face winding (a,b), -1 otherwise.
        int c_ij = (vi < vj) ? 1 : -1;
        int c_jk = (vj < vk) ? 1 : -1;
        int c_ki = (vk < vi) ? 1 : -1;

        // Local edge data (Algorithm 7, lines 6-7).
        Complex z_i = psi(vi), z_j = psi(vj), z_k = psi(vk);
        double v_ij = c_ij * omega(e_ij);
        double v_jk = c_jk * omega(e_jk);
        double v_ki = c_ki * omega(e_ki);

        // Branch index S_ijk (line 8).
        int S_ijk = s(e_ij) * s(e_jk) * s(e_ki);
        result.is_branch_triangle[f] = (S_ijk < 0);

        // Lines 9-11: branch triangle → flip v_ki.
        if (S_ijk < 0) {
            v_ki = -v_ki;
        }

        // Lines 12-16: if s_ij < 0, make values at j consistent w/ i.
        if (s(e_ij) < 0) {
            z_j = std::conj(z_j);
            v_ij = c_ij * v_ij;    // = omega(e_ij) since c_ij^2=1
            v_jk = -c_jk * v_jk;   // flip and un-orient
        }

        // Lines 17-21: if S_ijk*s_ki < 0, make values at k consistent w/ i.
        // S_ijk*s_ki = s_ij*s_jk*s_ki^2 = s_ij*s_jk.
        if (S_ijk * s(e_ki) < 0) {
            z_k = std::conj(z_k);
            v_ki = -c_ki * v_ki;    // flip and un-orient
            v_jk = c_jk * v_jk;    // un-orient (undo possible earlier flip)
        }

        // Lines 22-25: compute angles at triangle corners by accumulating
        // the spinning form around the face boundary.
        // delta_ab = arg(e^{iv_ab} z_a / z_b) is the angular "error"
        // between the transported psi and the actual psi at the target.
        // sigma_ab = v_ab - delta_ab is the spinning form.

        // Line 22: alpha_i = arg(z_i)
        double alpha_i = std::arg(z_i);

        // Line 23: alpha_j = alpha_i + v_ij - arg(e^{iv_ij} z_i / z_j)
        double alpha_j = alpha_i + v_ij -
            std::arg(std::exp(Complex(0, v_ij)) * z_i / z_j);

        // Line 24: alpha_k = alpha_j + v_jk - arg(e^{iv_jk} z_j / z_k)
        double alpha_k = alpha_j + v_jk -
            std::arg(std::exp(Complex(0, v_jk)) * z_j / z_k);

        // Line 25: alpha_i2 = alpha_k + v_ki - arg(e^{iv_ki} z_k / z_i)
        // (going all the way around back to i — the difference from alpha_i
        //  gives the total winding number)
        double alpha_i2 = alpha_k + v_ki -
            std::arg(std::exp(Complex(0, v_ki)) * z_k / z_i);

        // Line 26: midpoint coordinate
        double alpha_mid = alpha_i + (alpha_i2 - alpha_i) / 2.0;

        // Line 27: zero index n_ijk = winding / 2pi
        int n_ijk = static_cast<int>(
            std::round((alpha_i2 - alpha_i) / (2.0 * M_PI)));
        result.face_index(f) = n_ijk;

        // Lines 28-29: adjust corners for zeros.
        // The pseudocode subtracts 2*pi*n/3 and 4*pi*n/3 from j and k.
        double correction = 2.0 * M_PI * n_ijk / 3.0;
        alpha_j -= correction;
        alpha_k -= 2.0 * correction;

        result.alpha(f, 0) = alpha_i;
        result.alpha(f, 1) = alpha_j;
        result.alpha(f, 2) = alpha_k;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

StripePattern compute_stripe_pattern(const TriangleMesh& mesh,
                                      const DirectionField& field,
                                      const MeshGeometry& geom,
                                      double frequency) {
    Eigen::VectorXd freq =
        Eigen::VectorXd::Constant(mesh.num_vertices(), frequency);

    auto [omega, s] = compute_edge_data(mesh, geom, field.u, freq);

    auto A = build_stripe_energy_matrix(mesh, omega, s);
    auto B = build_stripe_mass_matrix(mesh);
    auto psi = compute_stripe_field(A, B);

    return compute_texture_coordinates(mesh, geom, psi, omega, s);
}

} // namespace hatching
