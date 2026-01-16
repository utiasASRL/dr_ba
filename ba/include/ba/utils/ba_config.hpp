// ba_config.hpp
#pragma once

#include <string>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace ba {

struct Options {
    // Data paths
    fs::path data_path;
    fs::path meas_path;
    fs::path input_path;
    std::vector<std::string> seq_ids;

    // Map parameters
    double voxel_res = 0.2;        // meters

    // Output parameters
    fs::path output_path;
    bool visualize_result = true;
    bool save_result = true;

    // Input parameters
    double max_dist = 80.0;       // meters
    double gauss_blur_sigma = 2.0; // pixels
    std::string init_poses = "gt";    // 'gt' or 'pogo'
    double init_translation_std = 0.5; // meters
    double init_rotation_std = 5.0;    // degrees
    std::string input_type = "local_maps"; // 'scans' or 'local_maps'
    double local_map_res = 0.1; // meters/pixel

    // Keyframing parameters
    int num_frames = 5;
    double max_kf_dist = 2.0;    // meters
    double max_kf_rot = 10.0;    // degrees

    // Optimization parameters
    int max_iterations = 20;
    double convergence_tol = 1e-3;
    double alpha = 0.5;        // step size
    bool adaptive_alpha = true; // decrease alpha if cost does not decrease
    double prior_map_std = 1e-3; // intensity units
    double meas_std = 1.0;       // intensity units
    bool use_rel_pose_prior = true;
    double rel_pose_prior_translation_std = 0.1; // meters
    double rel_pose_prior_rotation_std = 5.0;    // degrees
    double range_factor = 0.0;   // factor to scale range uncertainty to intensity uncertainty
    double cumul_thresh = 1.0;   // threshold to ignore measurements with too high a cumulative return
    int num_coarse_iterations = 5; // number of initial iterations with higher downsampling
    double coarse_downsample = 0.2; // downsampling factor for coarse iterations
    double refine_downsample = 1.0; // downsampling factor for refinement iterations
};

Options load_options(const YAML::Node& config);

} // namespace ba