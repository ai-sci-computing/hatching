/// @file geometry.cpp
/// @brief Implementation of discrete differential geometry operators.

#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace hatching {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Angle scaling (Eq. 11, Knoeppel 2013)
// ---------------------------------------------------------------------------

Eigen::VectorXd compute_angle_scaling(const TriangleMesh& mesh) {
    const int nv = mesh.num_vertices();
    Eigen::VectorXd s(nv);

    for (int i = 0; i < nv; ++i) {
        double angle_sum = 0.0;
        for (int f : mesh.VF[i]) {
            int k = mesh.local_index(f, i);
            angle_sum += mesh.tip_angle(f, k);
        }
        // On the boundary s_i := 1.0 (no rescaling).
        if (mesh.is_boundary_vertex[i] || angle_sum < 1e-15) {
            s(i) = 1.0;
        } else {
            s(i) = 2.0 * M_PI / angle_sum;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Rescaled halfedge angles
// ---------------------------------------------------------------------------

/// @brief Order the outgoing halfedges at vertex i in CCW order using
///        the incident face fan.
///
/// Returns a vector of neighbor vertex indices in CCW order.  For interior
/// vertices the fan is closed; for boundary vertices it is open.
static std::vector<int> ordered_neighbors(const TriangleMesh& mesh, int i) {
    const auto& faces = mesh.VF[i];
    if (faces.empty()) return {};

    // Build a map: for each face, find the two other vertices in CCW order
    // relative to i.  In a face (i, a, b) the CCW next after i is a, and
    // prev is b.  We record: face -> (next, prev).
    struct FanEntry {
        int next, prev;
        int face;
    };
    std::vector<FanEntry> fan;
    fan.reserve(faces.size());

    // Also build a map from "prev vertex" to the fan entry, so we can chain.
    std::unordered_map<int, int> prev_to_fan; // prev -> fan index

    for (int f : faces) {
        int k = mesh.local_index(f, i);
        int a = mesh.F(f, (k + 1) % 3); // CCW next
        int b = mesh.F(f, (k + 2) % 3); // CCW prev
        int idx = static_cast<int>(fan.size());
        fan.push_back({a, b, f});
        prev_to_fan[b] = idx;
    }

    // Chain the fan: the next vertex of one face should be the prev vertex
    // of the next face in CCW order.  I.e., fan entry with prev = X is
    // followed by fan entry with next = X... wait no.
    //
    // Around vertex i, faces are arranged so that for consecutive faces
    // f1 = (i, a, b) and f2 = (i, b, c), the shared edge is i-b.
    // So f1.prev = b and f2.next = b.  To chain CCW: start at some face,
    // follow: current.prev = next_face.next.
    std::unordered_map<int, int> next_to_fan;
    for (int idx = 0; idx < static_cast<int>(fan.size()); ++idx) {
        next_to_fan[fan[idx].next] = idx;
    }

    // Find the starting face.  For interior vertices any face works.
    // For boundary vertices, start at the face whose "next" edge is a
    // boundary (i.e., no face has prev = current.next).
    int start = 0;
    if (mesh.is_boundary_vertex[i]) {
        for (int idx = 0; idx < static_cast<int>(fan.size()); ++idx) {
            // This face starts the fan if there's no face whose prev ==
            // this face's next.
            if (prev_to_fan.find(fan[idx].next) == prev_to_fan.end()) {
                start = idx;
                break;
            }
        }
    }

    // Walk the fan.
    std::vector<int> result;
    result.reserve(fan.size() + 1);

    int cur = start;
    std::vector<bool> visited(fan.size(), false);
    while (!visited[cur]) {
        visited[cur] = true;
        result.push_back(fan[cur].next);
        // Find next face: it's the one whose "next" == current "prev".
        auto it = next_to_fan.find(fan[cur].prev);
        if (it == next_to_fan.end()) {
            // Boundary: add the last prev and stop.
            result.push_back(fan[cur].prev);
            break;
        }
        cur = it->second;
    }

    return result;
}

std::vector<std::unordered_map<int, double>>
compute_rescaled_angles(const TriangleMesh& mesh,
                        const Eigen::VectorXd& angle_scaling) {
    const int nv = mesh.num_vertices();
    std::vector<std::unordered_map<int, double>> theta(nv);

    for (int i = 0; i < nv; ++i) {
        auto neighbors = ordered_neighbors(mesh, i);
        if (neighbors.empty()) continue;

        double cumulative = 0.0;
        // The first neighbor defines angle 0 (the reference direction).
        theta[i][neighbors[0]] = 0.0;

        for (int p = 1; p < static_cast<int>(neighbors.size()); ++p) {
            int prev_nbr = neighbors[p - 1];
            int cur_nbr = neighbors[p];

            // Find the face containing i, prev_nbr, cur_nbr.
            // The tip angle at i between edges i->prev_nbr and i->cur_nbr.
            // This is the Euclidean tip angle at vertex i in the face that
            // has both prev_nbr and cur_nbr.
            double euclidean_angle = 0.0;
            for (int f : mesh.VF[i]) {
                int ki = mesh.local_index(f, i);
                int ka = mesh.local_index(f, prev_nbr);
                int kb = mesh.local_index(f, cur_nbr);
                if (ka >= 0 && kb >= 0) {
                    euclidean_angle = mesh.tip_angle(f, ki);
                    break;
                }
            }

            cumulative += euclidean_angle * angle_scaling(i);
            theta[i][cur_nbr] = cumulative;
        }
    }
    return theta;
}

// ---------------------------------------------------------------------------
// Parallel transport (Eq. 12)
// ---------------------------------------------------------------------------

Eigen::VectorXd compute_transport_angles(
    const TriangleMesh& mesh,
    const std::vector<std::unordered_map<int, double>>& theta, int n) {
    const int ne = mesh.num_edges();
    Eigen::VectorXd rho(ne);

    for (int e = 0; e < ne; ++e) {
        int i = mesh.edges[e].v0;
        int j = mesh.edges[e].v1;

        double theta_ij = 0.0; // rescaled angle of edge ij at vertex i
        double theta_ji = 0.0; // rescaled angle of edge ji at vertex j

        auto it_ij = theta[i].find(j);
        if (it_ij != theta[i].end()) theta_ij = it_ij->second;

        auto it_ji = theta[j].find(i);
        if (it_ji != theta[j].end()) theta_ji = it_ji->second;

        // Transport angle for the n-line bundle.
        // Knoeppel 2013 Eq. 12, matching stripe paper Eq. 5.
        rho(e) = static_cast<double>(n) * (-theta_ij + theta_ji + M_PI);
    }
    return rho;
}

// ---------------------------------------------------------------------------
// Holonomy (Eq. 13)
// ---------------------------------------------------------------------------

Eigen::VectorXd compute_holonomy(const TriangleMesh& mesh,
                                  const Eigen::VectorXd& rho) {
    const int nf = mesh.num_faces();
    Eigen::VectorXd Omega(nf);

    for (int f = 0; f < nf; ++f) {
        int i = mesh.F(f, 0);
        int j = mesh.F(f, 1);
        int k = mesh.F(f, 2);

        // Edge indices for edges ij, jk, ki.
        int e_ij = mesh.find_edge(i, j);
        int e_jk = mesh.find_edge(j, k);
        int e_ki = mesh.find_edge(k, i);

        // Transport coefficient r_ij = exp(i * rho_ij).
        // Sign: rho is stored for the canonical direction (v0 < v1).
        // If the face traversal direction matches canonical, use +rho;
        // otherwise use -rho (since r_ji = conj(r_ij) = exp(-i*rho_ij)).
        auto signed_rho = [&](int edge, int from, int to) -> double {
            if (mesh.edges[edge].v0 == from) {
                return rho(edge);
            } else {
                return -rho(edge);
            }
        };

        double r_ij = signed_rho(e_ij, i, j);
        double r_jk = signed_rho(e_jk, j, k);
        double r_ki = signed_rho(e_ki, k, i);

        // Holonomy = arg(exp(i*r_ij) * exp(i*r_jk) * exp(i*r_ki))
        //          = arg(exp(i*(r_ij + r_jk + r_ki)))
        // Normalized to (-pi, pi].
        double angle = r_ij + r_jk + r_ki;
        Omega(f) = std::remainder(angle, 2.0 * M_PI);
    }
    return Omega;
}

// ---------------------------------------------------------------------------
// Cotangent weights
// ---------------------------------------------------------------------------

Eigen::VectorXd compute_cotan_weights(const TriangleMesh& mesh) {
    const int ne = mesh.num_edges();
    Eigen::VectorXd w(ne);
    w.setZero();

    for (int f = 0; f < mesh.num_faces(); ++f) {
        for (int k = 0; k < 3; ++k) {
            // Edge opposite to vertex k.
            int ei = mesh.FE(f, k);
            double angle = mesh.tip_angle(f, k);
            double cot_val = std::cos(angle) / std::sin(angle);
            w(ei) += 0.5 * cot_val;
        }
    }
    return w;
}

// ---------------------------------------------------------------------------
// Hopf differential (Sec. 6.1.2, App. D.5, Knoeppel 2013)
// ---------------------------------------------------------------------------

Eigen::VectorXcd compute_hopf_differential(
    const TriangleMesh& mesh,
    const std::vector<std::unordered_map<int, double>>& theta,
    const Eigen::VectorXd& angle_scaling) {
    const int nv = mesh.num_vertices();
    Eigen::VectorXcd q_tilde(nv);
    q_tilde.setZero();

    // q_tilde_i = -1/4 * sum_{e incident to i} r_ie * beta_e * |p_e|
    //
    // where beta_e is the dihedral angle at edge e, |p_e| is the edge length,
    // and r_ie = exp(2i * theta_i(X_i, e)) is the transport coefficient from
    // vertex i to edge e for the 2-vector (line) bundle.
    //
    // The factor of 2 in the exponent corresponds to n=2 (line field).

    for (int e = 0; e < mesh.num_edges(); ++e) {
        int vi = mesh.edges[e].v0;
        int vj = mesh.edges[e].v1;
        double len = mesh.edge_length(e);

        // Dihedral angle at this edge.
        // The sign convention: β > 0 for convex edges.  To get a
        // consistent sign, we must order the two face normals so that
        // n_left is the face to the LEFT of the oriented edge (vi→vj)
        // and n_right is on the RIGHT.  The "left" face is the one where
        // the edge vi→vj appears in CCW order (i.e., vi is followed by
        // vj in the face winding).
        double beta = 0.0;
        if (mesh.edge_faces[e][0] >= 0 && mesh.edge_faces[e][1] >= 0) {
            int f0 = mesh.edge_faces[e][0];
            int f1 = mesh.edge_faces[e][1];

            // Determine which face has vi→vj in CCW order (= "left" face).
            int left_face = f0, right_face = f1;
            {
                // In face f0, check if vi is followed by vj in CCW order.
                int ki = mesh.local_index(f0, vi);
                int next = mesh.F(f0, (ki + 1) % 3);
                if (next != vj) {
                    // vi→vj is NOT CCW in f0, so f0 is the right face.
                    std::swap(left_face, right_face);
                }
            }

            Eigen::Vector3d n_left = mesh.face_normal(left_face);
            Eigen::Vector3d n_right = mesh.face_normal(right_face);
            Eigen::Vector3d edge_vec = mesh.V.row(vj) - mesh.V.row(vi);
            Eigen::Vector3d edge_dir = edge_vec.normalized();

            // Signed dihedral: positive when the surface bends "outward"
            // (convex) relative to the edge direction.
            double cos_beta = n_left.dot(n_right);
            cos_beta = std::max(-1.0, std::min(1.0, cos_beta));
            double sin_beta = n_left.cross(n_right).dot(edge_dir);
            beta = std::atan2(sin_beta, cos_beta);
        }

        // Transport coefficient r_ie = exp(2i * theta_{i,e}).
        // theta_{i,e} is the rescaled angle of the outgoing edge at vertex i.
        // Beta is a scalar property of the edge (same at both endpoints).
        auto add_contribution = [&](int v, int other) {
            auto it = theta[v].find(other);
            if (it == theta[v].end()) return;
            double th = it->second;
            Complex r = std::exp(Complex(0, 2.0 * th));
            q_tilde(v) += -0.25 * r * beta * len;
        };

        add_contribution(vi, vj);
        add_contribution(vj, vi);
    }

    return q_tilde;
}

// ---------------------------------------------------------------------------
// Full geometry computation
// ---------------------------------------------------------------------------

MeshGeometry compute_geometry(const TriangleMesh& mesh, int n) {
    MeshGeometry geom;
    geom.n = n;

    geom.angle_scaling = compute_angle_scaling(mesh);
    geom.theta = compute_rescaled_angles(mesh, geom.angle_scaling);
    geom.rho = compute_transport_angles(mesh, geom.theta, n);
    geom.Omega = compute_holonomy(mesh, geom.rho);
    geom.cotan_weights = compute_cotan_weights(mesh);

    // Gaussian curvature per face: K_ijk = Omega_ijk / (n * |t_ijk|).
    const int nf = mesh.num_faces();
    geom.K_face.resize(nf);
    for (int f = 0; f < nf; ++f) {
        double area = mesh.face_area(f);
        if (area > 1e-15) {
            geom.K_face(f) = geom.Omega(f) / (static_cast<double>(n) * area);
        } else {
            geom.K_face(f) = 0.0;
        }
    }

    geom.hopf_differential =
        compute_hopf_differential(mesh, geom.theta, geom.angle_scaling);

    return geom;
}

} // namespace hatching
