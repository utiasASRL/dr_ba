// problem.hpp
#pragma once

#include <ba/utils/ba_config.hpp>
#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include <ba/solver/result.hpp>

#include <iostream>
#include <ankerl/unordered_dense.h>
#include <lgmath/se3/Transformation.hpp>
#include <filesystem>
namespace fs = std::filesystem;

namespace ba {

class Problem {
public:
    using PriorMap = ankerl::unordered_dense::map<std::pair<int32_t, int32_t>, lgmath::se3::Transformation>;
    void initialize() {
        init_scans_and_map();
        initialized_ = true;
    }

    virtual void finalize() = 0;

    // Automatically clean up temporary directory if it exists
    virtual ~Problem() {
        cleanup_temp_dir();
    }

    // Accessors
    Options& opts() { return opts_; }
    VoxelMap& voxel_map() { return voxel_map_; }
    ScanManager& scan_manager() { return scan_manager_; }
    Result& result() { return result_; }
    PriorMap& pose_priors() { return pose_priors_; }
    bool is_initialized() const { return initialized_; }


protected:
    Problem(Options& opts)
        : opts_(opts),
          voxel_map_(opts_.voxel_res),
          scan_manager_(opts_.max_loaded_scans),
          result_(voxel_map_, scan_manager_, opts_.output_path),
          initialized_(false) {}

    virtual void init_scans_and_map() = 0;

    void cleanup_temp_dir() noexcept {
        if (!temp_dir_.empty() && fs::exists(temp_dir_)) {
            std::error_code ec;
            fs::remove_all(temp_dir_, ec);
            // intentionally ignore errors in destructor
        }
    }

    Options& opts_;
    VoxelMap voxel_map_;
    ScanManager scan_manager_;
    Result result_;
    // TODO: Deal with pose priors better
    PriorMap pose_priors_;
    bool initialized_;
    fs::path temp_dir_;
};


}   // namespace ba