// scan.hpp
#pragma once

#include <Eigen/Dense>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se2/Transformation.hpp>

namespace ba {

class Scan {
public:
	virtual ~Scan() = default;

	// Identification
	int id() const { return id_; }

	// Pose accessors
	const lgmath::se3::Transformation &pose() const { return pose_; }
	const lgmath::se2::Transformation pose2d() const { return pose_.toSE2(); }
	

	// Update pose (override if derived classes need custom behavior)
	virtual void update_pose(const lgmath::se3::Transformation &new_pose) {
		pose_ = new_pose;
	}

	// Interpolate intensity value at a query point in world frame
    // No value will be provided if the requested point is out of bounds
    // Additionally provides optional Jacobian of intensity w.r.t. SE(2) pose (1x3)
	virtual std::optional<double> interpolate(double x, double y,
							   Eigen::Matrix<double, 1, 3> *jacobian = nullptr) const = 0;

	// Coverage check at a query point in world frame
	virtual bool check_coverage_at_point(double x, double y) const = 0;

protected:
	Scan(int scan_id, const lgmath::se3::Transformation &pose)
		: id_(scan_id), pose_(pose) {}

	int id_;
	lgmath::se3::Transformation pose_;
};

} // namespace ba

