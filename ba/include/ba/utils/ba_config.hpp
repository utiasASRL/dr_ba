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
    std::string seq_id;

    // Map parameters
    double voxel_res = 0.2;        // meters

    // Output parameters
    fs::path output_path;
    bool visualize_result = true;
    bool save_result = true;

    // Input parameters
    double max_dist = 80.0;       // meters
    bool dist_field_preproc = true;
    double gauss_blur_sigma = 2.0; // pixels
    std::string init_poses = "gt";    // 'gt' or 'pogo'
    double init_translation_std = 0.5; // meters
    double init_rotation_std = 5.0;    // degrees
    std::string input_type = "local_maps"; // 'scans' or 'local_maps'
    double local_map_res = 0.1; // meters/pixel

    // Keyframing parameters
    double max_kf_dist = 2.0;    // meters
    double max_kf_rot = 10.0;    // degrees
    bool fix_first_scan = true;

    // Optimization parameters
    int num_threads = 1;
    int max_iterations = 20;
    double convergence_tol = 1e-3;
    double alpha = 0.5;        // step size
    bool adaptive_alpha = true; // decrease alpha if cost does not decrease
    double meas_std = 1.0;       // intensity units
    bool use_rel_pose_prior = true;
    double rel_pose_prior_translation_std = 0.1; // meters
    double rel_pose_prior_rotation_std = 5.0;    // degrees
    double range_factor = 0.0;   // factor to scale range uncertainty to intensity uncertainty
    bool use_cumul_thresh = true;
    double cumul_thresh = 1.0;   // threshold to ignore measurements with too high a cumulative return
    double zero_thresh = 1.0;    // max cumul return threshold to consider a measurement as zero return
    int num_coarse_iterations = 5; // number of initial iterations with higher downsampling
    double coarse_downsample = 0.2; // downsampling factor for coarse iterations
    double refine_downsample = 1.0; // downsampling factor for refinement iterations
    double tile_size = 0.0;     // meters, size of tiles to process separately
    int max_loaded_scans = 0;   // max number of scans to keep loaded in memory at once (>0 all)
    bool coarse_to_fine = false; // whether to switch from coarse to fine after num_coarse_iterations

    // Mapping parameters
    std::string pose_source = "gt"; // 'estimate', 'gt', 'pogo', 'dro'
    fs::path estimate_location;
    std::string map_seq;
    double map_max_dist = 80.0; // meters
    bool map_dist_field_preproc = true;
    double map_gauss_blur_sigma = 3.0; // pixels
    std::vector<std::pair<int, int>> frame_ranges; // pairs of start and end frame indices

    // Localization parameters
    bool use_odometry_prior = false;
    double odom_translation_std = 0.2; // meters
    double odom_rotation_std = 5.0;    // degrees
    fs::path map_location;
    int start_frame = 0;
    int end_frame = -1; // -1 for last frame
};

Options load_options(const YAML::Node& config);

} // namespace ba