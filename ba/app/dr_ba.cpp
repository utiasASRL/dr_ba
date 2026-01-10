#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include <ba/scans/local_map_scan.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/utils/io_utils.hpp"

#include <iostream>
#include <random>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <lgmath/se3/Transformation.hpp>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/so3/Operations.hpp>
#include <opencv2/core/eigen.hpp>

namespace fs = std::filesystem;
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Vec = Eigen::VectorXd;
using Vec3d = Eigen::Vector3dd;
using Mat = Eigen::MatrixXd;

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

    // Initialize looping through trajectory
    lgmath::se3::Transformation T_gt_0;
    lgmath::se3::Transformation T_est_0;
    lgmath::se3::Transformation T_kf_prev;  // Previous keyframe pose

    // Loop through all images
    std::cout << "Loading images from: " << all_img_dir << std::endl;
    double num_scans = files.size();
    double num_checked = -1;
    double num_loaded = 0;
    for (const auto& path : files) {
        if (num_loaded >= opts.num_frames) break;
        num_checked++;

        // Load in scan pose
        double timestamp_scan = std::stod(path.stem().string()) / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_scan);

        // Load in initial guess pose
        lgmath::se3::Transformation T_est;
        if (opts.init_poses == "pogo") {
            T_est = ba::get_interpolated_pose(all_pogo_poses, all_pogo_times, timestamp_scan);
        } else if (opts.init_poses == "gt") {
            T_est = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_scan);
            // Add noise to gt pose sampled from uniform distribution
            Vec3d noise;
            noise << translation_dist(rng), translation_dist(rng), rotation_dist(rng);
            lgmath::se3::Transformation T_noise = lgmath::se2::Transformation(noise).toSE3();
            T_est = T_est * T_noise;
        } else {
            throw std::invalid_argument("Invalid init_poses option: " + opts.init_poses);
        }

        if (num_loaded != 0) {
            // Check if this pose is a keyframe
            lgmath::se3::Transformation T_kf_rel = T_est.inverse() * T_kf_prev;
            double del_x = T_kf_rel.r_ab_inb()(0);
            double del_y = T_kf_rel.r_ab_inb()(1);
            double del_theta = T_kf_rel.vec()(5); // Yaw angle
            double translation_mag = std::sqrt(std::pow(del_x, 2) + std::pow(del_y, 2));
            double rotation_mag = std::abs(del_theta) * 180.0 / M_PI; // convert to degrees
            if (translation_mag < opts.max_kf_dist && rotation_mag < opts.max_kf_rot) {
                // Not a keyframe, skip
                continue;
            }
        } else {
            T_gt_0 = T_gt;
            T_est_0 = T_est;
            T_kf_prev = T_est;
        }

        // We've decided this is a keyframe!
        std::cout << "Processing frame " << num_checked << " / " << num_scans << std::endl;
        num_loaded++;
        T_kf_prev = T_est; 

        // Get relative gt transform
        lgmath::se3::Transformation T_gt_rel = T_gt_0.inverse() * T_gt;

        // Get relative est transform
        lgmath::se3::Transformation T_est_rel = T_est_0.inverse() * T_est;

        // TODO: Add support for more than just local_maps
        if (opts.input_type != "local_maps") {
            throw std::invalid_argument("Input type " + opts.input_type + " not supported yet.");
        }

        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        img.convertTo(img, CV_64F, 1.0 / 255.0);
        Mat img_mat;
        cv::cv2eigen(img, img_mat);

        // Create scan object
        auto scan = std::make_shared<ba::LocalMapScan>(num_checked, T_est_rel, T_gt_rel, opts.local_map_res, img_mat);
        scan_manager.add_scan(scan);
        vox_map.init_map(T_est_rel, opts.max_dist);
    }

    std::cout << "Scan manager has " << scan_manager.num_scans() << " scans." << std::endl;
    std::cout << "Full voxel map has " << vox_map.size() << " voxels." << std::endl;
    std::cout << "Initial pose RMSE (x, y, yaw): " << scan_manager.compute_pose_rmse().transpose() << std::endl;

    // Visualize final map
    vox_map.randomize(42); // For visualization purposes
    vox_map.visualize();

    return 0;
}
