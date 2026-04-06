/// @file camera.cpp
/// @brief Implementation of orbit camera.

#include "camera.h"

#include <cmath>

namespace hatching {

Camera::Camera()
    : target_(Eigen::Vector3d::Zero()),
      distance_(3.0),
      theta_(0.0),
      phi_(0.3) {}

void Camera::look_at(const Eigen::Vector3d& target, double distance) {
    target_ = target;
    distance_ = distance;
}

void Camera::rotate(double dx, double dy) {
    theta_ -= dx * 0.01;
    phi_ += dy * 0.01;
    // Clamp elevation to avoid gimbal lock.
    phi_ = std::max(-M_PI / 2.0 + 0.01, std::min(M_PI / 2.0 - 0.01, phi_));
}

void Camera::zoom(double delta) {
    distance_ *= std::exp(-delta * 0.1);
    distance_ = std::max(0.01, distance_);
}

void Camera::pan(double dx, double dy) {
    // Pan in the camera's local XY plane.
    double scale = distance_ * 0.002;
    Eigen::Vector3d right(std::cos(theta_), 0, -std::sin(theta_));
    Eigen::Vector3d up(0, 1, 0);
    target_ -= right * dx * scale;
    target_ += up * dy * scale;
}

Eigen::Vector3d Camera::position() const {
    double x = distance_ * std::cos(phi_) * std::sin(theta_);
    double y = distance_ * std::sin(phi_);
    double z = distance_ * std::cos(phi_) * std::cos(theta_);
    return target_ + Eigen::Vector3d(x, y, z);
}

Eigen::Matrix4f Camera::view_matrix() const {
    Eigen::Vector3d eye = position();
    Eigen::Vector3d forward = (target_ - eye).normalized();
    Eigen::Vector3d world_up(0, 1, 0);
    Eigen::Vector3d right = forward.cross(world_up).normalized();
    Eigen::Vector3d up = right.cross(forward).normalized();

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view(0, 0) = static_cast<float>(right.x());
    view(0, 1) = static_cast<float>(right.y());
    view(0, 2) = static_cast<float>(right.z());
    view(0, 3) = static_cast<float>(-right.dot(eye));
    view(1, 0) = static_cast<float>(up.x());
    view(1, 1) = static_cast<float>(up.y());
    view(1, 2) = static_cast<float>(up.z());
    view(1, 3) = static_cast<float>(-up.dot(eye));
    view(2, 0) = static_cast<float>(-forward.x());
    view(2, 1) = static_cast<float>(-forward.y());
    view(2, 2) = static_cast<float>(-forward.z());
    view(2, 3) = static_cast<float>(forward.dot(eye));
    return view;
}

Eigen::Matrix4f Camera::projection_matrix(float aspect) const {
    float f = 1.0f / std::tan(fov * static_cast<float>(M_PI) / 360.0f);
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0, 0) = f / aspect;
    proj(1, 1) = f;
    proj(2, 2) = (far_plane + near_plane) / (near_plane - far_plane);
    proj(2, 3) = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    proj(3, 2) = -1.0f;
    return proj;
}

} // namespace hatching
