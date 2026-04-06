#pragma once

/// @file camera.h
/// @brief Orbit camera for 3D mesh viewing.

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace hatching {

/// @brief Simple orbit camera that rotates around a target point.
class Camera {
public:
    /// @brief Construct a camera looking at the origin.
    Camera();

    /// @brief Set the target (look-at) point and distance.
    void look_at(const Eigen::Vector3d& target, double distance);

    /// @brief Rotate the camera by (dx, dy) screen-space deltas.
    void rotate(double dx, double dy);

    /// @brief Zoom by a scroll delta.
    void zoom(double delta);

    /// @brief Pan by (dx, dy) screen-space deltas.
    void pan(double dx, double dy);

    /// @brief Get the 4x4 view matrix.
    Eigen::Matrix4f view_matrix() const;

    /// @brief Get the 4x4 projection matrix.
    /// @param aspect Viewport width / height.
    Eigen::Matrix4f projection_matrix(float aspect) const;

    /// @brief Camera position in world space.
    Eigen::Vector3d position() const;

    float fov = 45.0f;        ///< Vertical field of view in degrees.
    float near_plane = 0.01f; ///< Near clipping plane.
    float far_plane = 1000.0f; ///< Far clipping plane.

private:
    Eigen::Vector3d target_;
    double distance_;
    double theta_; ///< Azimuth angle (radians).
    double phi_;   ///< Elevation angle (radians).
};

} // namespace hatching
