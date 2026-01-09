// scan.hpp
#pragma once

#include <Eigen/Dense>

namespace ba {

class Scan {
public:
	virtual ~Scan() = default;

	// Identification
	inline int id() const { return id_; }

	// Pose accessors
	inline const Eigen::Matrix4d &pose() const { return pose_; }
	inline const Eigen::Matrix3d &pose2d() const { return pose2d_; }

	// Update pose (override if derived classes need custom behavior)
	virtual void update_pose(const Eigen::Matrix4d &new_pose) {
		pose_ = new_pose;
		pose2d_.setIdentity();
		pose2d_.block<2, 2>(0, 0) = new_pose.block<2, 2>(0, 0);
		pose2d_.block<2, 1>(0, 2) = new_pose.block<2, 1>(0, 3);
	}

	// Interpolate intensity value at a query point in world frame
    // No value will be provided if the requested point is out of bounds
    // Additionally provides optional Jacobian of intensity w.r.t. SE(2) pose (1x3)
	virtual std::optional<double> interpolate(double x, double y,
							   Eigen::Matrix<double, 1, 3> *jacobian = nullptr) const = 0;

	// Coverage check at a query point in world frame
	virtual bool check_coverage_at_point(double x, double y) const = 0;

protected:
	Scan(int scan_id, const Eigen::Matrix4d &pose)
		: id_(scan_id), pose_(pose) {
		pose2d_.setIdentity();
		pose2d_.block<2, 2>(0, 0) = pose.block<2, 2>(0, 0);
		pose2d_.block<2, 1>(0, 2) = pose.block<2, 1>(0, 3);
	}

	int id_;
	Eigen::Matrix4d pose_;
	Eigen::Matrix3d pose2d_;
};

} // namespace ba

