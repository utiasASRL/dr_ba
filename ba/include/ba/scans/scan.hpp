// scan.hpp
#pragma once

#include <Eigen/Dense>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/se3/Operations.hpp>

namespace ba {

class Scan {
public:
	virtual ~Scan() = default;

	// Identification
	int id() const { return id_; }

	// Pose accessors
	const lgmath::se3::Transformation &pose() const { return pose_; }
	const lgmath::se2::Transformation pose2d() const { return pose_.toSE2(); }
	const lgmath::se3::Transformation &gt_pose() const { return gt_pose_; }
	const lgmath::se2::Transformation gt_pose2d() const { return gt_pose_.toSE2(); }
	

	// Update pose
	void update_pose(const Eigen::Matrix<double, 6, 1> &delta_xi) {
		// Update should be T_new = exp(-delta_xi) * T_old
		// but we don't have negative since lgmath flips the convention
		// internally
		lgmath::se3::Transformation T_update((delta_xi).eval());
		pose_ = T_update * pose_;
	}

	void update_pose(const Eigen::Matrix<double, 3, 1> &delta_xi) {
		Eigen::Matrix<double, 6, 1> delta_xi_se3;
		delta_xi_se3 << delta_xi(0), delta_xi(1), 0.0, 0.0, 0.0, delta_xi(2);
		update_pose(delta_xi_se3);
	}

	// Compute pose error (SE3)
	Eigen::Matrix<double, 6, 1> pose_error() const {
		lgmath::se3::Transformation T_err = pose_.inverse() * gt_pose_;
		return T_err.vec();
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
	Scan(int scan_id, const lgmath::se3::Transformation &pose,
		 const lgmath::se3::Transformation &gt_pose)
		: id_(scan_id), pose_(pose), gt_pose_(gt_pose) {}

	int id_;
	lgmath::se3::Transformation pose_;
	lgmath::se3::Transformation gt_pose_;
};

} // namespace ba

