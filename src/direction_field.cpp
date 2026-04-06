/// @file direction_field.cpp
/// @brief Implementation of globally optimal direction fields.

#include "direction_field.h"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <complex>
#include <random>

namespace hatching {

using Complex = std::complex<double>;
using SpMat = Eigen::SparseMatrix<Complex>;
using Triplet = Eigen::Triplet<Complex>;

// ---------------------------------------------------------------------------
// Mass matrix M (Eq. 17, Knoeppel 2013)
// ---------------------------------------------------------------------------

SpMat build_mass_matrix(const TriangleMesh& mesh, const MeshGeometry& geom) {
    const int nv = mesh.num_vertices();
    std::vector<Triplet> triplets;
    triplets.reserve(nv + 6 * mesh.num_faces());

    for (int f = 0; f < mesh.num_faces(); ++f) {
        double area = mesh.face_area(f);
        double Omega_f = geom.Omega(f);
        int vi = mesh.F(f, 0);
        int vj = mesh.F(f, 1);
        int vk = mesh.F(f, 2);

        // Diagonal: M_ii += |t_ijk| / 6.
        double diag_val = area / 6.0;
        triplets.push_back({vi, vi, Complex(diag_val, 0)});
        triplets.push_back({vj, vj, Complex(diag_val, 0)});
        triplets.push_back({vk, vk, Complex(diag_val, 0)});

        // Off-diagonal (Eq. 17): M_jk from the face contribution.
        // M_jk = r_bar_jk * |t_ijk| * (6*exp(i*Omega) - 6 - 6i*Omega
        //         + 3*Omega^2 + i*Omega^3) / (3 * Omega^4)
        //
        // where r_bar_jk is the conjugate of the transport coefficient from
        // j to k through the face.
        //
        // For small Omega, this has a removable singularity with limit 1/12.
        // We use a Taylor expansion for |Omega| < 0.01.

        // The three off-diagonal pairs in face f are (j,k), (k,i), (i,j).
        auto mass_offdiag = [&](int a, int b, double rho_ab) {
            Complex r_bar = std::exp(Complex(0, -rho_ab));
            Complex M_ab;

            if (std::abs(Omega_f) < 1e-6) {
                // Taylor: 1/12 * (1 + i*Omega/4 + ...)
                M_ab = area * r_bar *
                       (1.0 / 12.0 +
                        Complex(0, 1) * Omega_f / 48.0);
            } else {
                double O = Omega_f;
                double O2 = O * O;
                double O3 = O2 * O;
                double O4 = O2 * O2;
                Complex eio = std::exp(Complex(0, O));
                Complex num = 6.0 * eio - 6.0 - Complex(0, 6) * O +
                              3.0 * O2 + Complex(0, 1) * O3;
                M_ab = area * r_bar * num / (3.0 * O4);
            }

            triplets.push_back({a, b, M_ab});
            triplets.push_back({b, a, std::conj(M_ab)});
        };

        // Compute per-edge transport within the face.
        // The transport from vertex a to vertex b through face f:
        // we need the "intra-face" transport coefficient.
        // For simplicity, we use the edge transport stored in geom.rho.
        auto face_rho = [&](int from, int to) -> double {
            int e = mesh.find_edge(from, to);
            if (mesh.edges[e].v0 == from) {
                return geom.rho(e);
            } else {
                return -geom.rho(e);
            }
        };

        mass_offdiag(vj, vk, face_rho(vj, vk));
        mass_offdiag(vk, vi, face_rho(vk, vi));
        mass_offdiag(vi, vj, face_rho(vi, vj));
    }

    SpMat M(nv, nv);
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

// ---------------------------------------------------------------------------
// Energy matrix A (Eq. 18, Knoeppel 2013)
// ---------------------------------------------------------------------------

SpMat build_energy_matrix(const TriangleMesh& mesh, const MeshGeometry& geom,
                          double s) {
    const int nv = mesh.num_vertices();
    std::vector<Triplet> triplets;
    triplets.reserve(nv + 6 * mesh.num_faces());

    // Build the connection Laplacian (Dirichlet energy of the n-line bundle).
    // For each edge: w_ij * |u_j - r_ij * u_i|^2 contributes:
    //   A_ii += w_ij,  A_jj += w_ij,
    //   A_ij += -w_ij * conj(r_ij),  A_ji += -w_ij * r_ij.
    //
    // The cotangent weight is assembled per face: each face contributes
    // 0.5 * cot(angle) to the edge opposite that angle.

    for (int f = 0; f < mesh.num_faces(); ++f) {
        for (int loc = 0; loc < 3; ++loc) {
            int a = mesh.F(f, (loc + 1) % 3);
            int b = mesh.F(f, (loc + 2) % 3);
            double angle = mesh.tip_angle(f, loc);
            double half_cot = 0.5 * std::cos(angle) / std::sin(angle);

            int e = mesh.find_edge(a, b);
            double rho_ab =
                (mesh.edges[e].v0 == a) ? geom.rho(e) : -geom.rho(e);
            Complex r_bar_ab = std::exp(Complex(0, -rho_ab));

            // Off-diagonal: A_ab = -w * conj(r_ab).
            Complex off_diag = -half_cot * r_bar_ab;
            triplets.push_back({a, b, off_diag});
            triplets.push_back({b, a, std::conj(off_diag)});

            // Diagonal: A_aa += w, A_bb += w.
            triplets.push_back({a, a, Complex(half_cot, 0)});
            triplets.push_back({b, b, Complex(half_cot, 0)});
        }
    }

    // Geometry-aware correction for s != 0 (Eq. 18).
    // For closed meshes, E_s = E_D - s*(E_A - E_H), and the difference
    // E_A - E_H = (1/2) * integral nK|psi|^2 dA.
    // This adds a curvature potential: A_ij -= s * (Omega/|t|) * M_ij
    // per face, where M_ij is the mass matrix contribution from that face.
    if (std::abs(s) > 1e-15) {
        SpMat M = build_mass_matrix(mesh, geom);
        for (int f = 0; f < mesh.num_faces(); ++f) {
            double area = mesh.face_area(f);
            double Omega_f = geom.Omega(f);
            if (area < 1e-15) continue;
            double factor = -s * Omega_f / area;

            int verts[3] = {mesh.F(f, 0), mesh.F(f, 1), mesh.F(f, 2)};
            // The per-face mass matrix contribution is already summed into M.
            // We approximate by using the global M entries scaled by the
            // fraction of area this face contributes to each vertex pair.
            // For the diagonal: M_ii_face ~ area/6.
            for (int k = 0; k < 3; ++k) {
                triplets.push_back(
                    {verts[k], verts[k], Complex(factor * area / 6.0, 0)});
            }
            // For off-diagonals: use the M coefficient times factor.
            // In the flat limit M_jk_face ~ area/12 * conj(r_jk).
            for (int k = 0; k < 3; ++k) {
                int a = verts[(k + 1) % 3];
                int b = verts[(k + 2) % 3];
                int e = mesh.find_edge(a, b);
                double rho_ab =
                    (mesh.edges[e].v0 == a) ? geom.rho(e) : -geom.rho(e);
                Complex r_bar = std::exp(Complex(0, -rho_ab));
                Complex M_ab_face = (area / 12.0) * r_bar;
                Complex correction = factor * M_ab_face;
                triplets.push_back({a, b, correction});
                triplets.push_back({b, a, std::conj(correction)});
            }
        }
    }

    SpMat A(nv, nv);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

// ---------------------------------------------------------------------------
// Smoothest field via inverse power iteration (Algorithm 2)
// ---------------------------------------------------------------------------

/// @brief Helper to convert a complex sparse matrix to a real 2n x 2n matrix
///        for use with real Cholesky solvers.
///
/// Maps the complex matrix A (n x n) to a real matrix R (2n x 2n) via the
/// isomorphism z = x + iy -> [x; y], so that R * [x; y] = [Re(A*z); Im(A*z)].
static Eigen::SparseMatrix<double> complex_to_real(const SpMat& C) {
    const int n = static_cast<int>(C.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(C.nonZeros() * 4);

    for (int k = 0; k < C.outerSize(); ++k) {
        for (SpMat::InnerIterator it(C, k); it; ++it) {
            int i = static_cast<int>(it.row());
            int j = static_cast<int>(it.col());
            double re = it.value().real();
            double im = it.value().imag();

            // [re -im] [x]   [re*x - im*y]
            // [im  re] [y] = [im*x + re*y]
            triplets.push_back({i, j, re});
            triplets.push_back({i, j + n, -im});
            triplets.push_back({i + n, j, im});
            triplets.push_back({i + n, j + n, re});
        }
    }

    Eigen::SparseMatrix<double> R(2 * n, 2 * n);
    R.setFromTriplets(triplets.begin(), triplets.end());
    return R;
}

/// @brief Convert a real 2n vector [x; y] back to complex n vector x + iy.
static Eigen::VectorXcd real_to_complex(const Eigen::VectorXd& r) {
    int n = static_cast<int>(r.size()) / 2;
    Eigen::VectorXcd c(n);
    for (int i = 0; i < n; ++i) {
        c(i) = Complex(r(i), r(i + n));
    }
    return c;
}

/// @brief Convert a complex n vector to a real 2n vector [Re; Im].
static Eigen::VectorXd complex_to_real_vec(const Eigen::VectorXcd& c) {
    int n = static_cast<int>(c.size());
    Eigen::VectorXd r(2 * n);
    for (int i = 0; i < n; ++i) {
        r(i) = c(i).real();
        r(i + n) = c(i).imag();
    }
    return r;
}

Eigen::VectorXcd compute_smoothest_field(const SpMat& A, const SpMat& M,
                                          int num_iterations) {
    const int n = static_cast<int>(A.rows());

    // Convert to real system for Cholesky.
    Eigen::SparseMatrix<double> A_real = complex_to_real(A);
    Eigen::SparseMatrix<double> M_real = complex_to_real(M);

    // Add small regularization to ensure positive definiteness.
    double eps = 1e-8;
    Eigen::SparseMatrix<double> I(2 * n, 2 * n);
    I.setIdentity();
    A_real += eps * M_real;

    // Cholesky factorization of A.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_real);
    if (solver.info() != Eigen::Success) {
        // Fallback: return random field.
        Eigen::VectorXcd u = Eigen::VectorXcd::Random(n);
        return u / u.norm();
    }

    // Inverse power iteration: find smallest eigenvector of A u = lambda M u.
    // x <- random, then repeat: x <- A^{-1} M x, x <- x / sqrt(x^T M x).
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Eigen::VectorXd x(2 * n);
    for (int i = 0; i < 2 * n; ++i) {
        x(i) = dist(rng);
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        Eigen::VectorXd Mx = M_real * x;
        x = solver.solve(Mx);

        // Normalize: x <- x / sqrt(x^T M x).
        double norm2 = x.dot(M_real * x);
        if (norm2 > 1e-30) {
            x /= std::sqrt(norm2);
        }
    }

    return real_to_complex(x);
}

// ---------------------------------------------------------------------------
// Aligned field (Algorithm 3)
// ---------------------------------------------------------------------------

Eigen::VectorXcd compute_aligned_field(const SpMat& A, const SpMat& M,
                                        const Eigen::VectorXcd& q_tilde,
                                        double lambda_t) {
    const int n = static_cast<int>(A.rows());

    // Solve (A - lambda_t * M) u_tilde = q_tilde  (Algorithm 3).
    // The RHS is q̃ = Mq directly — NOT M*q̃.
    SpMat lhs = A;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (SpMat::InnerIterator it(M, k); it; ++it) {
            lhs.coeffRef(it.row(), it.col()) -= lambda_t * it.value();
        }
    }

    Eigen::SparseMatrix<double> lhs_real = complex_to_real(lhs);
    Eigen::VectorXd rhs_real = complex_to_real_vec(q_tilde);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(lhs_real);
    if (solver.info() != Eigen::Success) {
        return Eigen::VectorXcd::Zero(n);
    }

    Eigen::VectorXd x = solver.solve(rhs_real);
    Eigen::VectorXcd u_tilde = real_to_complex(x);

    // Normalize: u = u_tilde / ||u_tilde||_M.
    Eigen::SparseMatrix<double> M_real = complex_to_real(M);
    double norm2 = x.dot(M_real * x);
    if (norm2 > 1e-30) {
        u_tilde /= std::sqrt(norm2);
    }

    return u_tilde;
}

// ---------------------------------------------------------------------------
// Singularity indices (Section 6.1.3, Eq. 19)
// ---------------------------------------------------------------------------

Eigen::VectorXi compute_singularity_indices(
    const TriangleMesh& mesh, const MeshGeometry& geom,
    const Eigen::VectorXcd& u) {
    const int nf = mesh.num_faces();
    Eigen::VectorXi idx(nf);

    for (int f = 0; f < nf; ++f) {
        int vi = mesh.F(f, 0);
        int vj = mesh.F(f, 1);
        int vk = mesh.F(f, 2);

        // For each edge of the face, compute the rotation angle omega_ij
        // such that u_j = exp(i * omega_ij) * r_ij * u_i.
        auto rotation_angle = [&](int a, int b) -> double {
            int e = mesh.find_edge(a, b);
            double rho_ab =
                (mesh.edges[e].v0 == a) ? geom.rho(e) : -geom.rho(e);
            Complex r = std::exp(Complex(0, rho_ab));
            Complex ratio = u(b) / (r * u(a));
            return std::arg(ratio);
        };

        double w_ij = rotation_angle(vi, vj);
        double w_jk = rotation_angle(vj, vk);
        double w_ki = rotation_angle(vk, vi);

        double Omega_f = geom.Omega(f);

        // Index = (1 / 2pi) * (w_ij + w_jk + w_ki + Omega_ijk).
        double raw_index =
            (w_ij + w_jk + w_ki + Omega_f) / (2.0 * M_PI);
        idx(f) = static_cast<int>(std::round(raw_index));
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Full pipeline
// ---------------------------------------------------------------------------

DirectionField compute_direction_field(const TriangleMesh& mesh, double s,
                                        double lambda_t) {
    MeshGeometry geom = compute_geometry(mesh, 2);

    SpMat M = build_mass_matrix(mesh, geom);
    SpMat A = build_energy_matrix(mesh, geom, s);

    DirectionField result;

    // Always use the curvature-aligned linear solve (Algorithm 3):
    //   (A - λ_t M) ũ = q̃
    // where q̃ is the Hopf differential (RHS).  λ_t = 0 gives maximum
    // smoothing; more negative λ_t gives stronger curvature alignment.
    // Use -q̃ to align with minimum curvature direction (Sec. 6.1.2).
    Eigen::VectorXcd q_tilde = geom.hopf_differential;
    result.u = compute_aligned_field(A, M, q_tilde, lambda_t);

    // Pointwise normalize: the field defines directions, not magnitudes.
    // u_i ← u_i / |u_i| at each vertex (skip near-zero vertices near
    // singularities where the field magnitude vanishes).
    for (int v = 0; v < mesh.num_vertices(); ++v) {
        double mag = std::abs(result.u(v));
        if (mag > 1e-15) {
            result.u(v) /= mag;
        }
    }

    result.singularity_index =
        compute_singularity_indices(mesh, geom, result.u);

    return result;
}

} // namespace hatching
