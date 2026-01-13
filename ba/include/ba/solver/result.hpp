#pragma once

#include <ba/scans/manager.hpp>
#include <ba/map/voxel_map.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace ba {

class Result {
public:
    Result(VoxelMap &voxel_map, ScanManager &scan_manager, const fs::path& output_dir)
        : voxel_map_(std::move(voxel_map)),
          scan_manager_(std::move(scan_manager)),
          output_dir_(std::move(output_dir)) {
    }
        
    // Immutable accessors
    const std::vector<double>& cost_history() const { return cost_history_; }
    const std::vector<Eigen::Vector3d>& rmse_history() const { return rmse_history_; }

    // Mutable accessors
    VoxelMap& voxel_map() { return voxel_map_; }
    ScanManager& scan_manager() { return scan_manager_; }

    // Adding info
    void add_cost(double cost) {
        cost_history_.push_back(cost);
    }
    void add_rmse(const Eigen::Vector3d& rmse) {
        rmse_history_.push_back(rmse);
    }
    
    // Result output
    void save_rmse_cost_to_csv(const fs::path& output_dir = {}) const;
    void save_voxel_map() const;
    void save_full_result() const {
        save_voxel_map();
        save_rmse_cost_to_csv();
    }
    void visualize_all_results();

private:
    VoxelMap voxel_map_;
    ScanManager scan_manager_;
    const fs::path& output_dir_;

    std::vector<double> cost_history_;
    std::vector<Eigen::Vector3d> rmse_history_;
    const fs::path csv_path_ = output_dir_ / "rmse_cost_history.csv";
    const fs::path voxel_path_ = output_dir_ / "voxel_map.bin";

};

} // namespace ba