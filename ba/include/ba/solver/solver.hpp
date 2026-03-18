#pragma once

#include <ba/scans/manager.hpp>
#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>
#include <ba/solver/result.hpp>
#include <ba/problem/problem.hpp>

namespace ba {

class Solver {
public:

    Solver(Problem& problem)
        : problem_(problem),
          opts_(problem.opts()),
          result_(problem.result()),
          scan_manager_(problem.scan_manager()),
          voxel_map_(problem.voxel_map()),
          pose_priors_(problem.pose_priors())
        {
            cost_ = std::numeric_limits<double>::max();
            prev_cost_ = std::numeric_limits<double>::max();
            alpha_ = opts_.alpha;
        }

    ~Solver() = default;

    // Functions to be fulfilled by derived classes
    virtual void construct_problem(double downsample_factor = 1.0) = 0;
    virtual bool solve() = 0;
    virtual void update_poses() = 0;
    virtual void update_map() = 0;
    virtual void optimize() = 0;

    // Accessors
    double cost() const { return cost_; }

    // Writers
    void set_alpha(double alpha) { alpha_ = alpha; }

protected:
    Problem& problem_;
    const Options& opts_;
    Result& result_;
    ScanManager& scan_manager_;
    VoxelMap& voxel_map_;
    const Problem::PriorMap& pose_priors_;
    
    // Variables to be passed around
    double cost_;
    double prev_cost_;
    double alpha_;
};


}