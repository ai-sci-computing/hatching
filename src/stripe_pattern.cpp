/// @file stripe_pattern.cpp
/// @brief Implementation of stripe patterns on surfaces.
///
/// Follows the pseudocode in Appendix C of Knoeppel et al. 2015
/// ("Stripe Patterns on Surfaces") as closely as possible.

#include "stripe_pattern.h"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <complex>
#include <queue>
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
    const int nv = mesh.num_vertices();
    const int ne = mesh.num_edges();

    // Extract consistent 1-vector directions from the 2-field u via BFS.
    // The 2-field u_i has arg(u_i) = 2*phi_i with a pi-ambiguity in phi.
    // BFS resolves this by transporting phi along edges and picking the
    // closest branch at each vertex.
    std::vector<double> phi(nv, 0.0);
    std::vector<bool> visited(nv, false);

    int seed = 0;
    for (int v = 1; v < nv; ++v) {
        if (std::abs(u(v)) > std::abs(u(seed))) seed = v;
    }
    phi[seed] = std::arg(u(seed)) / 2.0;
    visited[seed] = true;

    std::queue<int> bfs;
    bfs.push(seed);
    while (!bfs.empty()) {
        int v = bfs.front();
        bfs.pop();
        for (int ei : mesh.VE[v]) {
            int other = (mesh.edges[ei].v0 == v)
                            ? mesh.edges[ei].v1
                            : mesh.edges[ei].v0;
            if (visited[other]) continue;
            visited[other] = true;

            // n=1 transport from v to other.
            double th_vo = 0, th_ov = 0;
            {
                auto it = geom.theta[v].find(other);
                if (it != geom.theta[v].end()) th_vo = it->second;
            }
            {
                auto it = geom.theta[other].find(v);
                if (it != geom.theta[other].end()) th_ov = it->second;
            }
            double rho1 = -th_vo + th_ov + M_PI;
            double transported = phi[v] + rho1;

            double raw = std::arg(u(other)) / 2.0;
            double diff = std::remainder(raw - transported, M_PI);
            phi[other] = transported + diff;

            bfs.push(other);
        }
    }

    // Compute s_ij and omega_ij using the consistent phi values.
    Eigen::VectorXd omega(ne);
    Eigen::VectorXi s(ne);

    for (int e = 0; e < ne; ++e) {
        int i = mesh.edges[e].v0;
        int j = mesh.edges[e].v1;
        double len = mesh.edge_length(e);

        double theta_ij = 0.0, theta_ji = 0.0;
        {
            auto it = geom.theta[i].find(j);
            if (it != geom.theta[i].end()) theta_ij = it->second;
        }
        {
            auto it = geom.theta[j].find(i);
            if (it != geom.theta[j].end()) theta_ji = it->second;
        }

        double rho_ij = -theta_ij + theta_ji + M_PI;

        double transported = phi[i] + rho_ij;
        double diff = std::remainder(transported - phi[j], 2.0 * M_PI);
        s(e) = (std::abs(diff) <= M_PI / 2.0) ? 1 : -1;

        double phi_j_eff = (s(e) == 1) ? phi[j] : phi[j] + M_PI;

        // Angular displacement omega_ij (Eq. 6-7 of stripe paper).
        // Project Z = ν·X onto the edge direction (i→j) at both endpoints.
        // At i: edge direction is θ_ij (outgoing).
        // At j: edge direction is θ_ji + π (incoming = reversed outgoing).
        // cos(φ - (θ + π)) = -cos(φ - θ), so the second term is negated.
        omega(e) = 0.5 * len *
                   (frequency(i) * std::cos(phi[i] - theta_ij) -
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

        // Standard rotation block for all edges.  On the double cover,
        // the energy is always |ψ̃_q - e^{iω̃}ψ̃_p|² (standard transport).
        // The omega values from compute_edge_data already encode the
        // correct double cover transport for both s=+1 and s=-1.
        // A_ij = -w·R^T, A_ji = -w·R  where R = rotation by omega.
        triplets.push_back({i, j, -w * c});
        triplets.push_back({i, j + nv, -w * sn});
        triplets.push_back({i + nv, j, w * sn});
        triplets.push_back({i + nv, j + nv, -w * c});

        triplets.push_back({j, i, -w * c});
        triplets.push_back({j, i + nv, w * sn});
        triplets.push_back({j + nv, i, -w * sn});
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

    if (tk <= ti && tk <= tj) {
        return pi_n / 3.0 * (1.0 + (ti - tj) / (1.0 - 3.0 * tk));
    } else if (ti <= tj && ti <= tk) {
        return pi_n / 3.0 * (3.0 + (tj - tk) / (1.0 - 3.0 * ti));
    } else {
        return pi_n / 3.0 * (5.0 + (tk - ti) / (1.0 - 3.0 * tj));
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
    result.is_branch_triangle.resize(nf, false);
    result.alpha_subdiv.resize(nf, 7);
    result.alpha_subdiv.setZero();

    for (int f = 0; f < nf; ++f) {
        int vi = mesh.F(f, 0);
        int vj = mesh.F(f, 1);
        int vk = mesh.F(f, 2);

        int e_ij = mesh.find_edge(vi, vj);
        int e_jk = mesh.find_edge(vj, vk);
        int e_ki = mesh.find_edge(vk, vi);

        int c_ij = (vi < vj) ? 1 : -1;
        int c_jk = (vj < vk) ? 1 : -1;

        Complex z_i = psi(vi), z_j = psi(vj), z_k = psi(vk);
        double v_ij = c_ij * omega(e_ij);
        double v_jk = c_jk * omega(e_jk);

        // Spinning form: accumulate texture coordinates per face.
        // δ = arg(e^{iv} z_from / z_to) is the angular "error".
        // σ = v - δ is the actual angular displacement.
        // Jumps across edges are multiples of 2π (invisible under cos).
        double alpha_i = std::arg(z_i);

        double alpha_j = alpha_i + v_ij -
            std::arg(std::exp(Complex(0, v_ij)) * z_i / z_j);

        double alpha_k = alpha_j + v_jk -
            std::arg(std::exp(Complex(0, v_jk)) * z_j / z_k);

        result.alpha(f, 0) = alpha_i;
        result.alpha(f, 1) = alpha_j;
        result.alpha(f, 2) = alpha_k;
        result.face_index(f) = 0;
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
