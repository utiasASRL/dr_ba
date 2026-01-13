#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include <ba/scans/local_map_scan.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/utils/io_utils.hpp"
#include "ba/solver/solver.hpp"

#include <iostream>
#include <random>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/so3/Operations.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Sparse>

namespace fs = std::filesystem;

void save_results_to_csv(const std::vector<Eigen::Vector3d>& rmse_history,
                        const std::vector<double>& cost_history,
                        const std::string& path) {
    std::ofstream file(path);
    file << "cost, rmse_x,rmse_y,rmse_yaw\n";

    for (std::size_t i = 0; i < rmse_history.size(); ++i) {
        double cost = (i - 1 >= 0) ? cost_history[i - 1] : 0.0;
        file << cost_history[i] << "," << rmse_history[i](0) << "," << rmse_history[i](1) << "," << rmse_history[i](2) << "\n";
    }
}


int main() {
    // Load in config from ba/config/dr_ba_config.yaml
    fs::path config_path = fs::path(__FILE__).parent_path().parent_path() / "config" / "dr_ba_config.yaml";
    YAML::Node config = YAML::LoadFile(config_path.string());
    ba::Options opts = ba::load_options(config);
    std::string seq_id = opts.seq_ids[0];

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts.data_path / seq_id, all_gt_poses, all_gt_times);
    // Initialize uniform distribution for noise
    std::uniform_real_distribution<double> translation_dist(-opts.init_translation_std, opts.init_translation_std);
    double rotation_std_rad = opts.init_rotation_std * M_PI / 180.0;
    std::uniform_real_distribution<double> rotation_dist(-rotation_std_rad, rotation_std_rad);
    std::mt19937 rng(99); // Fixed seed for reproducibility

    // Load pogo poses
    std::vector<lgmath::se3::Transformation> all_pogo_poses;
    std::vector<double> all_pogo_times;
    ba::load_pogo_poses_and_times(opts.meas_path / seq_id, all_pogo_poses, all_pogo_times);

    // Initialize map and loader
    ba::VoxelMap vox_map(opts.voxel_res);
    ba::ScanManager scan_manager;

    // Load in image
    // TODO: Generalize to other input types
    fs::path all_img_dir = opts.meas_path / seq_id / opts.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Load in cumulative return images
    fs::path cumul_img_dir = opts.meas_path / seq_id / "cumulated_returns";
    std::vector<fs::path> cumul_files;
    for (const auto& entry : fs::directory_iterator(cumul_img_dir)) {
        if (entry.is_regular_file()) {
            cumul_files.push_back(entry.path());
        }
    }
    std::sort(cumul_files.begin(), cumul_files.end());

    // Initialize looping through trajectory
    lgmath::se3::Transformation T_gt_abs_0(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    lgmath::se3::Transformation T_est_abs_0(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    lgmath::se3::Transformation T_kf_prev(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));  // Previous keyframe pose
    int kf_prev_id = 0;
    ankerl::unordered_dense::map<std::pair<int32_t, int32_t>, lgmath::se3::Transformation> pose_priors;

    // Loop through all images
    std::cout << "Loading images from: " << all_img_dir << std::endl;
    double num_scans = files.size();
    double num_checked = -1;
    double num_loaded = 0;
    std::vector<Eigen::Vector3d> rmse_history;
    for (const auto& path : files) {
        if (num_loaded >= opts.num_frames) break;
        num_checked++;

        // Load in scan pose
        double timestamp_scan = std::stod(path.stem().string()) / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_scan);

        // Load in initial guess pose
        lgmath::se3::Transformation T_est_rel(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
        lgmath::se3::Transformation T_est_abs;
        if (opts.init_poses == "pogo") {
            T_est_abs = ba::get_interpolated_pose(all_pogo_poses, all_pogo_times, timestamp_scan);
            T_est_rel = T_est_abs_0.inverse() * T_est_abs;
        } else if (opts.init_poses == "gt") {
            T_est_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_scan);
            T_est_rel = T_est_abs_0.inverse() * T_est_abs;
            // Add noise to gt pose sampled from uniform distribution
            Eigen::Vector3d noise;
            noise << translation_dist(rng), translation_dist(rng), rotation_dist(rng);
            lgmath::se3::Transformation T_noise = lgmath::se2::Transformation(noise).toSE3();
            T_est_rel = T_est_rel * T_noise;
        } else {
            throw std::invalid_argument("Invalid init_poses option: " + opts.init_poses);
        }

        if (num_loaded != 0) {
            // Temp, only load scans close to frame 0 in translation
            // double translation_from_0 = (T_est_abs.r_ab_inb() - T_est_abs_0.r_ab_inb()).norm();
            // if (translation_from_0 > 5.0) {
            //     continue;
            // }
        
            // Check if this pose is a keyframe
            lgmath::se3::Transformation T_kf_rel = T_est_abs.inverse() * T_kf_prev;
            double del_x = T_kf_rel.r_ab_inb()(0);
            double del_y = T_kf_rel.r_ab_inb()(1);
            double del_theta = T_kf_rel.vec()(5); // Yaw angle
            double translation_mag = std::sqrt(std::pow(del_x, 2) + std::pow(del_y, 2));
            double rotation_mag = std::abs(del_theta) * 180.0 / M_PI; // convert to degrees
            if (translation_mag < opts.max_kf_dist && rotation_mag < opts.max_kf_rot) {
                // Not a keyframe, skip
                continue;
            }
            // Set up prior from prev keyframe radar frame to this keyframe radar frame
            pose_priors[{kf_prev_id, num_checked}] = T_kf_rel;
        } else {
            T_gt_abs_0 = T_gt_abs;
            T_est_abs_0 = T_est_abs;
            T_kf_prev = T_est_abs;
            T_est_rel = lgmath::se3::Transformation(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
        }

        // We've decided this is a keyframe!
        std::cout << "Processing frame " << num_checked << " / " << num_scans << std::endl;
        kf_prev_id = num_checked;
        num_loaded++;
        T_kf_prev = T_est_abs;

        // Get relative gt transform
        lgmath::se3::Transformation T_gt_rel = T_gt_abs_0.inverse() * T_gt_abs;

        // TODO: Add support for more than just local_maps
        if (opts.input_type != "local_maps") {
            throw std::invalid_argument("Input type " + opts.input_type + " not supported yet.");
        }

        // Load in image as Eigen matrix
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        // Apply Gaussian blur
        if (opts.gauss_blur_sigma > 0.0) {
            int ksize = static_cast<int>(std::ceil(opts.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
            cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts.gauss_blur_sigma);
        }
        // Convert to CV_64F and normalize to [0, 1]
        img.convertTo(img, CV_64F, 1.0 / 255.0);
        Eigen::MatrixXd img_mat;
        cv::cv2eigen(img, img_mat);

        // Load in cumulative return image
        fs::path cumul_path = cumul_files[num_checked];
        cv::Mat cumul_img = cv::imread(cumul_path.string(), cv::IMREAD_GRAYSCALE);
        // Convert to CV_64F and normalize to [0, 1]
        cumul_img.convertTo(cumul_img, CV_64F, 1.0 / 255.0);
        Eigen::MatrixXd cumul_img_mat;
        cv::cv2eigen(cumul_img, cumul_img_mat);

        // Project relative matrices to SE2
        T_est_rel = T_est_rel.toSE2().toSE3();
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        // Create scan object
        auto scan = std::make_shared<ba::LocalMapScan>(num_checked, opts, T_est_rel, T_gt_rel, img_mat, cumul_img_mat);
        scan_manager.add_scan(scan);
        vox_map.init_map(T_est_rel, opts.max_dist);
    }

    std::cout << "Scan manager has " << scan_manager.num_scans() << " scans." << std::endl;
    std::cout << "Full voxel map has " << vox_map.size() << " voxels." << std::endl;
    std::cout << "Initial pose RMSE (x, y, yaw): " << scan_manager.compute_pose_rmse().transpose() << std::endl;

    rmse_history.push_back(scan_manager.compute_pose_rmse());

    std::cout << "Starting optimization..." << std::endl;
    ba::Solver solver(opts, scan_manager, vox_map, pose_priors);
    std::vector<double> cost_history = solver.optimize(rmse_history);

    // Save RMSE history to CSV
    fs::path rmse_path = opts.meas_path / seq_id / "dr_ba_rmse_history.csv";
    save_results_to_csv(rmse_history, cost_history, rmse_path.string());
    std::cout << "Saved RMSE history to: " << rmse_path << std::endl;

    // Call Python plot
    std::string cmd = "python3 /home/dl/Documents/phd/dev/dr_ba/ba/app/plot_errors.py " + rmse_path.string();

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("Python script failed");
    }

    fs::path voxel_output_path = opts.meas_path / seq_id / "voxels.bin";
    vox_map.save_to_file(voxel_output_path.string());
    std::cout << "Saved voxel map to: " << voxel_output_path << std::endl;

    // Visualize final map
    vox_map.visualize();

    return 0;
}
