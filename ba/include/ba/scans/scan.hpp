// scan.hpp
#pragma once

#include <Eigen/Dense>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/se3/Operations.hpp>
#include <ba/utils/ba_config.hpp>

namespace ba {

class Scan {
public:
	struct Measurement {
		double x;
		double y;
		double intensity;
		double covariance;
		Eigen::Matrix<double, 1, 3> jacobian; // Jacobian w.r.t. SE(2) pose
	};

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
		lgmath::se3::Transformation T_update((-delta_xi).eval());
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

	// Set fixed flag
	void set_fixed(bool fixed) { fixed_ = fixed; }
	bool is_fixed() const { return fixed_; }

	// Clone method for deep copying
	virtual std::shared_ptr<Scan> clone() const = 0;

	// Interpolate intensity value at a query point in world frame
    // No value will be provided if the requested point is out of bounds
    // Additionally provides optional Jacobian of intensity w.r.t. SE(2) pose (1x3)
	virtual std::optional<Measurement> interpolate(double x, double y) const = 0;

	// Coverage check at a query point in world frame
	virtual bool check_coverage_at_point(double x, double y) const = 0;

protected:
	Scan(int scan_id, const Options &opts, const lgmath::se3::Transformation &pose,
		 const lgmath::se3::Transformation &gt_pose)
		: id_(scan_id), meas_std_(opts.meas_std), pose_(pose), gt_pose_(gt_pose) {}

	int id_;
	double meas_std_;
	lgmath::se3::Transformation pose_;
	lgmath::se3::Transformation gt_pose_;
	bool fixed_ = false;
};

} // namespace ba

