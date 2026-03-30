/// @file stripe_pattern.cpp
/// @brief Implementation of stripe patterns on surfaces.
///
/// Follows the pseudocode in Appendix C of Knoeppel et al. 2015
/// ("Stripe Patterns on Surfaces") as closely as possible.

#include "stripe_pattern.h"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <complex>
#include <cstdio>
#include <random>

namespace hatching {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Edge data (Algorithm 3, Knoeppel 2015) — per-edge local computation
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::VectorXi>
compute_edge_data(const TriangleMesh& mesh, const MeshGeometry& geom,
                  const Eigen::VectorXcd& u,
                  const Eigen::VectorXd& frequency) {
    const int nv = mesh.num_vertices();
    const int ne = mesh.num_edges();

    // Per-vertex canonical 1-direction: sigma_i = sqrt(u_i / |u_i|).
    // This picks one of the two branches of the half-angle purely from u_i
    // (no propagation, no BFS).  The branch cut of sqrt (at arg = ±pi)
    // is handled per-edge by the sign matching below.
    Eigen::VectorXcd sigma(nv);
    for (int v = 0; v < nv; ++v) {
        double mag = std::abs(u(v));
        if (mag < 1e-15) {
            sigma(v) = Complex(1, 0);
        } else {
            sigma(v) = std::sqrt(u(v) / mag);
        }
    }

    Eigen::VectorXd omega(ne);
    Eigen::VectorXi s(ne);

    for (int e = 0; e < ne; ++e) {
        int i = mesh.edges[e].v0;
        int j = mesh.edges[e].v1;
        double len = mesh.edge_length(e);

        // Rescaled angles of this edge in local frames at i and j.
        double theta_ij = 0.0, theta_ji = 0.0;
        {
            auto it = geom.theta[i].find(j);
            if (it != geom.theta[i].end()) theta_ij = it->second;
        }
        {
            auto it = geom.theta[j].find(i);
            if (it != geom.theta[j].end()) theta_ji = it->second;
        }

        // n=1 parallel transport from i to j (Eq. 12 with n=1).
        double rho1 = -theta_ij + theta_ji + M_PI;
        Complex r1 = std::exp(Complex(0, rho1));

        // Transport sigma_i to j's frame and match with sigma_j.
        Complex sigma_i_at_j = r1 * sigma(i);
        double alignment = (std::conj(sigma(j)) * sigma_i_at_j).real();

        Complex sigma_j_matched;
        if (alignment >= 0) {
            s(e) = 1;
            sigma_j_matched = sigma(j);
        } else {
            s(e) = -1;
            sigma_j_matched = -sigma(j);
        }

        // Angular displacement omega_ij (Eq. 6-7 of stripe paper).
        // Trapezoidal rule for the integral of the projected frequency
        // 1-form ν·cos(X - edge_dir) along the edge.
        // At i (outgoing): ν_i · ê_ij = freq_i · Re(conj(σ_i) · e^{iθ_ij})
        // At j (incoming): ν_j · ê_{j←i} = -freq_j · Re(conj(σ_j) · e^{iθ_ji})
        double proj_i = frequency(i) *
            (std::conj(sigma(i)) * std::exp(Complex(0, theta_ij))).real();
        double proj_j = frequency(j) *
            (std::conj(sigma_j_matched) * std::exp(Complex(0, theta_ji))).real();

        omega(e) = 0.5 * len * (proj_i - proj_j);
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

        double w = 0.0;
        for (int side = 0; side < 2; ++side) {
            int f = mesh.edge_faces[e][side];
            if (f < 0) continue;
            for (int kk = 0; kk < 3; ++kk) {
                if (mesh.FE(f, kk) == e) {
                    double angle = mesh.tip_angle(f, kk);
                    w += 0.5 * std::cos(angle) / std::sin(angle);
                    break;
                }
            }
        }

        double c = std::cos(omega(e));
        double sn = std::sin(omega(e));

        // Energy matrix (Algorithm 4):
        //   s ≥ 0: A_ij = -w · [e^{iω}]
        //   s < 0: A_ij = -w · [conj(e^{iω})] = -w · [e^{-iω}]
        // A_ji = A_ij^T.
        double se = (s(e) >= 0) ? 1.0 : -1.0;
        double sn_eff = se * sn;  // flip sin for s < 0 (conjugation)
        triplets.push_back({i, j, -w * c});
        triplets.push_back({i, j + nv, w * sn_eff});
        triplets.push_back({i + nv, j, -w * sn_eff});
        triplets.push_back({i + nv, j + nv, -w * c});

        triplets.push_back({j, i, -w * c});
        triplets.push_back({j, i + nv, -w * sn_eff});
        triplets.push_back({j + nv, i, w * sn_eff});
        triplets.push_back({j + nv, i + nv, -w * c});

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
// Mass matrix (Algorithm 5)
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
// Principal eigenvector (Algorithm 6)
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
// lArg interpolant (Eq. 11)
// ---------------------------------------------------------------------------

double lArg(int n, double ti, double tj, double tk) {
    if (n == 0) return 0.0;

    double pi_n = M_PI * static_cast<double>(n);

    // The function winds n times around the barycenter.
    // At corners: lArg(n,1,0,0)=0, lArg(n,0,1,0)=2πn/3, lArg(n,0,0,1)=4πn/3.
    if (tk <= ti && tk <= tj) {
        return pi_n / 3.0 * (1.0 + (tj - ti) / (1.0 - 3.0 * tk));
    } else if (ti <= tj && ti <= tk) {
        return pi_n / 3.0 * (3.0 + (tk - tj) / (1.0 - 3.0 * ti));
    } else {
        return pi_n / 3.0 * (5.0 + (ti - tk) / (1.0 - 3.0 * tj));
    }
}

// ---------------------------------------------------------------------------
// Texture coordinates (Algorithm 7)
// ---------------------------------------------------------------------------

StripePattern compute_texture_coordinates(
    const TriangleMesh& mesh, const MeshGeometry& /*geom*/,
    const Eigen::VectorXcd& psi, const Eigen::VectorXd& omega,
    const Eigen::VectorXi& s) {
    const int nf = mesh.num_faces();
    StripePattern result;
    result.alpha.resize(nf, 3);
    result.face_index.resize(nf);
    result.face_index_raw.resize(nf);
    result.is_branch_triangle.resize(nf, false);
    result.alpha_subdiv.resize(nf, 7);
    result.alpha_subdiv.setZero();
    double max_residual = 0.0;

    for (int f = 0; f < nf; ++f) {
        int vi = mesh.F(f, 0);
        int vj = mesh.F(f, 1);
        int vk = mesh.F(f, 2);

        int e_ij = mesh.find_edge(vi, vj);
        int e_jk = mesh.find_edge(vj, vk);
        int e_ki = mesh.find_edge(vk, vi);

        // Alg 7, lines 2-4: canonical edge signs.
        int c_ij = (vi < vj) ? 1 : -1;
        int c_jk = (vj < vk) ? 1 : -1;
        int c_ki = (vk < vi) ? 1 : -1;

        // Alg 7, lines 5-6: local copies.
        Complex z_i = psi(vi), z_j = psi(vj), z_k = psi(vk);
        double v_ij = c_ij * omega(e_ij);
        double v_jk = c_jk * omega(e_jk);
        double v_ki = c_ki * omega(e_ki);

        // Alg 7, line 7: branch index.
        int sij = s(e_ij), sjk = s(e_jk), ski = s(e_ki);
        int S_ijk = sij * sjk * ski;

        // Alg 7, lines 8-10: branch triangle adjustment.
        if (S_ijk < 0) {
            v_ki = -v_ki;
        }

        // Alg 7, lines 11-15: make values at j consistent with i.
        if (sij < 0) {
            z_j = std::conj(z_j);
            v_ij = c_ij * v_ij;
            v_jk = -c_jk * v_jk;
        }

        // Alg 7, lines 16-20: make values at k consistent with i.
        if (S_ijk * ski < 0) {
            z_k = std::conj(z_k);
            v_ki = -c_ki * v_ki;
            v_jk = c_jk * v_jk;
        }

        // Alg 7, lines 22-25: spinning form chain i→j→k→i.
        double alpha_i = std::arg(z_i);
        double alpha_j = alpha_i + v_ij -
            std::arg(std::exp(Complex(0, v_ij)) * z_i / z_j);
        double alpha_k = alpha_j + v_jk -
            std::arg(std::exp(Complex(0, v_jk)) * z_j / z_k);
        double alpha_i2 = alpha_k + v_ki -
            std::arg(std::exp(Complex(0, v_ki)) * z_k / z_i);

        // Alg 7, line 27: compute zero index.
        double raw = (alpha_i2 - alpha_i) / (2.0 * M_PI);
        int n_ijk = static_cast<int>(std::round(raw));
        double residual = std::abs(raw - n_ijk);
        if (residual > max_residual) max_residual = residual;
        result.face_index(f) = n_ijk;
        result.face_index_raw(f) = raw;

        // Alg 7, lines 28-29: adjust zeros.
        alpha_j -= 2.0 * M_PI * n_ijk / 3.0;
        alpha_k -= 4.0 * M_PI * n_ijk / 3.0;

        result.alpha(f, 0) = alpha_i;
        result.alpha(f, 1) = alpha_j;
        result.alpha(f, 2) = alpha_k;
    }

    // Count nonzero n_ijk values.
    int n_pos = 0, n_neg = 0, n_zero = 0;
    for (int f = 0; f < nf; ++f) {
        if (result.face_index(f) > 0) ++n_pos;
        else if (result.face_index(f) < 0) ++n_neg;
        else ++n_zero;
    }
    std::printf("n_ijk max residual: %.6f  (n>0: %d, n<0: %d, n=0: %d)\n",
                max_residual, n_pos, n_neg, n_zero);

    return result;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

StripePattern compute_stripe_pattern(const TriangleMesh& mesh,
                                      const DirectionField& field,
                                      const MeshGeometry& geom,
                                      double frequency,
                                      bool use_psi_one) {
    Eigen::VectorXd freq =
        Eigen::VectorXd::Constant(mesh.num_vertices(), frequency);

    auto [omega, s] = compute_edge_data(mesh, geom, field.u, freq);

    Eigen::VectorXcd psi;
    if (use_psi_one) {
        psi = Eigen::VectorXcd::Ones(mesh.num_vertices());
    } else {
        auto A = build_stripe_energy_matrix(mesh, omega, s);
        auto B = build_stripe_mass_matrix(mesh);
        psi = compute_stripe_field(A, B);
    }

    return compute_texture_coordinates(mesh, geom, psi, omega, s);
}

} // namespace hatching
