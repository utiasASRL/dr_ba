#pragma once

#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ba/solver/solver.hpp>
#include <ba/problem/problem.hpp>
#include <ba/problem/loc_problem.hpp>
#include <ba/scans/scan.hpp>
#include <memory>

namespace ba {

class LocSolver : public Solver {
public:
    LocSolver(Problem& problem) : Solver(problem) {
        alpha_ = problem.opts().alpha;
    }

    void construct_problem(double downsample_factor = 1.0) override;
    void construct_problem(const std::shared_ptr<Scan>& scan);
    bool solve() override;
    void update_poses() override;
    void update_map() override;
    void optimize() override;

    // accesor
    double cost() const { return cost_; }

    ~LocSolver() = default;

private:
    // Variables to be passed around
    std::vector<ba::VoxelMap::Index> voxel_keys_;
    Eigen::Matrix3d lhs_;
    Eigen::Vector3d rhs_;
    double alpha_;
    int num_scans_;
    lgmath::se3::Transformation curr_pose_;
    Eigen::Matrix3d curr_cov_;
    Eigen::Vector3d avg_pose_error_;

    // local functions
    void compute_errors(LocProblem& loc_problem, const std::shared_ptr<Scan>& scan, int i);
    void odometry_step(LocProblem& loc_problem, int i);
};


}