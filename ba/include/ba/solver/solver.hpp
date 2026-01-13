#pragma once

#include <ba/scans/manager.hpp>
#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>


namespace ba {

class Solver {
public:
    using PriorStruct = ankerl::unordered_dense::map<std::pair<int32_t, int32_t>, lgmath::se3::Transformation>;

    Solver(ba::Options opts, ScanManager& scan_manager, VoxelMap& vox_map, PriorStruct pose_priors)
        : opts_(opts), scan_manager_(scan_manager), vox_map_(vox_map), pose_priors_(pose_priors) {
            cost_ = std::numeric_limits<double>::max();
            prev_cost_ = std::numeric_limits<double>::max();
        }

    void construct_problem(ba::ScanManager &scan_manager, double downsample_factor = 1.0);
    bool solve();
    void update_poses(ba::ScanManager &scan_manager);
    void update_map();
    void optimize();

    // accesor
    double cost() const { return cost_; }

    ~Solver() = default;

private:
    ba::Options opts_;
    ScanManager& scan_manager_;
    VoxelMap& vox_map_;
    PriorStruct pose_priors_;
    
    // Variables to be passed around
    double cost_;
    double prev_cost_;
    double lambda_ = 1.0;
    std::vector<ba::VoxelMap::Index> voxel_keys_;
    Eigen::VectorXd del_x_;
    Eigen::MatrixXd H_TT_;
    Eigen::VectorXd H_MM_diag_;
    Eigen::MatrixXd H_TM_;
    Eigen::VectorXd J_M_B_;
    Eigen::VectorXd J_T_B_;

};


}