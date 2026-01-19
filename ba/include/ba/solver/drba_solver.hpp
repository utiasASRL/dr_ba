#pragma once

#include <ba/scans/manager.hpp>
#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>
#include <ba/solver/result.hpp>
#include <ba/problem/problem.hpp>
#include <ba/solver/solver.hpp>

namespace ba {

class DrBASolver : public Solver {
public:
    struct Tile {
        std::vector<VoxelMap::Index> voxel_indices;
        std::vector<int> scan_ids;
    };

    DrBASolver(Problem& problem) : Solver(problem) {}

    void tile_problem();
    void construct_problem(double downsample_factor = 1.0) override;
    bool solve() override;
    void update_poses() override;
    void update_map() override;
    void optimize() override;

    // accesor
    double cost() const { return cost_; }

    ~DrBASolver() = default;

private:
    // Problem tiling
    std::vector<Tile> tiles_;
    
    // Variables to be passed around
    std::vector<ba::VoxelMap::Index> voxel_keys_;
    Eigen::VectorXd del_x_;
    Eigen::MatrixXd lhs_;
    Eigen::VectorXd rhs_;
};


}