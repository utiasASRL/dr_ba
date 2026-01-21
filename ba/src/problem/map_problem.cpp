#include <ba/problem/map_problem.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>

namespace ba {

void MapProblem::init_scans_and_map() {
    if (opts_.pose_source == "estimate") {
        init_scans_and_map_from_estimates();
    } else {
        init_scans_and_map_from_data();
    }
}

void MapProblem::init_scans_and_map_from_estimates() {
    std::string seq_id = opts_.seq_ids[0];
    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = fs::temp_directory_path() / "dr_map_temp_gauss_blur" / seq_id;
    fs::create_directories(temp_dir);

    // Load in voxel map from estimates
    std::string estimate_location = opts_.estimate_location.string() + "/voxel_map.bin";
    std::cout << "Loading voxel map from estimates: " << estimate_location << std::endl;
    voxel_map_.load_poses_from_file(estimate_location);
    const std::vector<int> pose_ids = voxel_map_.pose_ids();
    const std::vector<lgmath::se2::Transformation> poses_se2 = voxel_map_.poses();
    // Re-initialize a new voxel_map
    voxel_map_ = VoxelMap(opts_.voxel_res);

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id, all_gt_poses, all_gt_times);

    // Load in scans based on pose IDs
    // Load in images
    fs::path all_img_dir = opts_.meas_path / seq_id / opts_.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Load in cumulative return images
    fs::path cumul_img_dir = opts_.meas_path / seq_id / "cumulated_returns";
    std::vector<fs::path> cumul_files;
    for (const auto& entry : fs::directory_iterator(cumul_img_dir)) {
        if (entry.is_regular_file()) {
            cumul_files.push_back(entry.path());
        }
    }

    // Check validity of frame ranges
    int num_scans = files.size();
    for (auto& range : opts_.frame_ranges) {
        if (range.second == -1) {
            range.second = num_scans - 1;
        }
        if (range.second < range.first) {
            throw std::invalid_argument("Invalid frame range: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "]");
        }
        if (range.first < 0 || range.second >= num_scans) {
            throw std::out_of_range("Frame range out of bounds: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "]");
        }
    }

    std::sort(cumul_files.begin(), cumul_files.end());
    lgmath::se3::Transformation T_gt_abs_0;
    int num_checked = -1;
    for (size_t i=0; i<files.size(); i++) {
        num_checked++;

        // Check if frame in desired ranges
        bool in_range = false;
        for (const auto& range : opts_.frame_ranges) {
            if (num_checked >= range.first && num_checked <= range.second) {
                in_range = true;
                break;
            }
        }

        if (i == 0) {
            T_gt_abs_0 = ba::get_interpolated_pose(all_gt_poses, all_gt_times, std::stod(files[0].stem().string()) / 1e6);
        }
        // Check if current index is in pose_ids
        if (std::find(pose_ids.begin(), pose_ids.end(), i) == pose_ids.end()) {
            continue;
        }
        if (!in_range) {
            continue;
        }

        const auto& path = files[i];

        // Load in scan pose
        int64_t timestamp = std::stoll(path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        lgmath::se3::Transformation T_gt_rel = T_gt_abs_0.inverse() * T_gt_abs;
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        // Get SE2 pose from voxel map
        int pose_idx = std::distance(pose_ids.begin(), std::find(pose_ids.begin(), pose_ids.end(), i));
        lgmath::se2::Transformation T_est_se2 = poses_se2[pose_idx];

        // Load in image as Eigen matrix
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        // Apply Gaussian blur
        if (opts_.gauss_blur_sigma > 0.0) {
            int ksize = static_cast<int>(std::ceil(opts_.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
            cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts_.gauss_blur_sigma);
        }

        // Load in image as Eigen matrix
        fs::path temp_img_path = temp_dir / (std::to_string(num_checked) + "_" + std::to_string(opts_.gauss_blur_sigma) + ".png");
        if (!fs::exists(temp_img_path)) {
            // Only process if the temp image does not already exist
            cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
            // Apply Gaussian blur
            if (opts_.gauss_blur_sigma > 0.0) {
                int ksize = static_cast<int>(std::ceil(opts_.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
                cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts_.gauss_blur_sigma);
            }
            // Convert to CV_32F and normalize to [0, 1]
            img.convertTo(img, CV_32F, 1.0 / 255.0);
            // Save blurred image to temp directory for easy loading
            ba::save_img_bin(temp_img_path, img);
        }

        // Load in cumulative return image
        std::optional<fs::path> temp_cumul_img_path = temp_dir / (std::to_string(num_checked) + "_" + std::to_string(opts_.gauss_blur_sigma) + "_cumul.png");
        if (!fs::exists(temp_cumul_img_path.value())) {
            fs::path cumul_path = cumul_files[num_checked];
            cv::Mat cumul_img = cv::imread(cumul_path.string(), cv::IMREAD_GRAYSCALE);
            // Convert to CV_32F and normalize to [0, 1]
            cumul_img.convertTo(cumul_img, CV_32F, 1.0 / 255.0);
            // Save cumulative image to temp directory for easy loading
            ba::save_img_bin(temp_cumul_img_path.value(), cumul_img);
        }

        // Create LocalMapScan and add to scan manager
        auto scan = std::make_shared<LocalMapScan>(
            timestamp,
            i,
            opts_,
            T_est_se2.toSE3(),
            T_gt_rel,
            temp_img_path,
            temp_cumul_img_path);
        scan_manager_.add_scan(scan);
        voxel_map_.init_map(T_est_se2, opts_.max_dist, i);
    }
}

void MapProblem::init_scans_and_map_from_data() {
    std::string seq_id = opts_.seq_ids[0];
    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = fs::temp_directory_path() / "dr_ba_temp_gauss_blur";
    fs::create_directories(temp_dir);

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id, all_gt_poses, all_gt_times);

    // Load pogo poses
    std::vector<lgmath::se3::Transformation> all_pogo_poses;
    std::vector<double> all_pogo_times;
    ba::load_pogo_poses_and_times(opts_.meas_path / seq_id, all_pogo_poses, all_pogo_times);

    // Load DRO poses
    std::vector<lgmath::se3::Transformation> all_dro_poses;
    std::vector<double> all_dro_times;
    ba::load_dro_poses_and_times(opts_.meas_path / seq_id, all_dro_poses, all_dro_times);

    // Initialize uniform distribution for noise
    std::uniform_real_distribution<double> translation_dist(-opts_.init_translation_std, opts_.init_translation_std);
    double rotation_std_rad = opts_.init_rotation_std * M_PI / 180.0;
    std::uniform_real_distribution<double> rotation_dist(-rotation_std_rad, rotation_std_rad);
    std::mt19937 rng(99); // Fixed seed for reproducibility

    // Load in images
    fs::path all_img_dir = opts_.meas_path / seq_id / opts_.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Load in cumulative return images
    fs::path cumul_img_dir = opts_.meas_path / seq_id / "cumulated_returns";
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

    // Loop through all images
    std::cout << "Loading images from: " << all_img_dir << std::endl;
    int num_scans = files.size();
    int num_checked = -1;
    int num_loaded = 0;

    // Check validity of frame ranges
    int max_frame = 0;
    for (auto& range : opts_.frame_ranges) {
        if (range.second == -1) {
            range.second = num_scans - 1;
        }
        if (range.second < range.first) {
            throw std::invalid_argument("Invalid frame range: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "]");
        }
        if (range.first < 0 || range.second >= num_scans) {
            throw std::out_of_range("Frame range out of bounds: [" + std::to_string(range.first) + ", " + std::to_string(range.second) + "]");
        }
        max_frame = std::max(max_frame, range.second);
    }

    for (const auto& path : files) {
        // Only consider files ending with .png
        if (path.extension() != ".png") {
            continue;
        }
        num_checked++;
        if (num_checked > max_frame) {
            break;
        }

        // Check if frame in desired ranges
        bool in_range = false;
        for (const auto& range : opts_.frame_ranges) {
            if (num_checked >= range.first && num_checked <= range.second) {
                in_range = true;
                break;
            }
        }

        // Load in scan pose
        int64_t timestamp = std::stoll(path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);

        // Load in initial guess pose
        lgmath::se3::Transformation T_est_rel(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
        lgmath::se3::Transformation T_est_abs;
        if (opts_.pose_source == "pogo") {
            T_est_abs = ba::get_interpolated_pose(all_pogo_poses, all_pogo_times, timestamp_seconds);
        } else if (opts_.pose_source == "gt") {
            T_est_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        } else if (opts_.pose_source == "dro") {
            T_est_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamp_seconds);
        } else {
            throw std::invalid_argument("Invalid pose_source option: " + opts_.init_poses);
        }

        T_est_rel = T_est_abs_0.inverse() * T_est_abs;
        // Add noise to gt pose sampled from uniform distribution
        Eigen::Vector3d noise;
        noise << translation_dist(rng), translation_dist(rng), rotation_dist(rng);
        lgmath::se3::Transformation T_noise = lgmath::se2::Transformation(noise).toSE3();
        T_est_rel = T_est_rel * T_noise;

        if (num_checked != 0) {
            // Check if this pose is a keyframe
            lgmath::se3::Transformation T_kf_rel = T_est_abs.inverse() * T_kf_prev;
            double del_x = T_kf_rel.r_ab_inb()(0);
            double del_y = T_kf_rel.r_ab_inb()(1);
            double del_theta = T_kf_rel.vec()(5); // Yaw angle
            double translation_mag = std::sqrt(std::pow(del_x, 2) + std::pow(del_y, 2));
            double rotation_mag = std::abs(del_theta) * 180.0 / M_PI; // convert to degrees
            if (translation_mag < opts_.max_kf_dist && rotation_mag < opts_.max_kf_rot) {
                // Not a keyframe, skip
                continue;
            }
            // Set up prior from prev keyframe radar frame to this keyframe radar frame
            pose_priors_[{kf_prev_id, num_checked}] = T_kf_rel;
        } else {
            T_gt_abs_0 = T_gt_abs;
            T_est_abs_0 = T_est_abs;
            T_kf_prev = T_est_abs;
            T_est_rel = lgmath::se3::Transformation(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
        }

        // We've decided this is a keyframe!
        // std::cout << "Processing frame " << num_checked << " / " << num_scans << std::endl;
        kf_prev_id = num_checked;
        T_kf_prev = T_est_abs;

        if (!in_range) {
            // We want to do keyframing in the same way for all frames, but dont
            // want to load out of range
            continue;
        }

        // Get relative gt transform
        lgmath::se3::Transformation T_gt_rel = T_gt_abs_0.inverse() * T_gt_abs;

        // TODO: Add support for more than just local_maps
        if (opts_.input_type != "scans" && opts_.input_type != "local_maps") {
            throw std::invalid_argument("Input type " + opts_.input_type + " not supported yet.");
        }

        // Load in image as Eigen matrix
        fs::path temp_img_path = temp_dir / (std::to_string(num_checked) + "_" + std::to_string(opts_.gauss_blur_sigma) + ".png");
        if (!fs::exists(temp_img_path)) {
            // Only process if the temp image does not already exist
            cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
            // Apply Gaussian blur
            if (opts_.gauss_blur_sigma > 0.0) {
                int ksize = static_cast<int>(std::ceil(opts_.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
                cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts_.gauss_blur_sigma);
            }
            // Convert to CV_32F and normalize to [0, 1]
            img.convertTo(img, CV_32F, 1.0 / 255.0);
            // Save blurred image to temp directory for easy loading
            ba::save_img_bin(temp_img_path, img);
        }

        // Load in cumulative return image
        std::optional<fs::path> temp_cumul_img_path = temp_dir / (std::to_string(num_checked) + "_" + std::to_string(opts_.gauss_blur_sigma) + "_cumul.png");
        if (!fs::exists(temp_cumul_img_path.value())) {
            fs::path cumul_path = cumul_files[num_checked];
            cv::Mat cumul_img = cv::imread(cumul_path.string(), cv::IMREAD_GRAYSCALE);
            // Convert to CV_32F and normalize to [0, 1]
            cumul_img.convertTo(cumul_img, CV_32F, 1.0 / 255.0);
            // Save cumulative image to temp directory for easy loading
            ba::save_img_bin(temp_cumul_img_path.value(), cumul_img);
        }

        // Create scan object
        if (!opts_.use_cumul_thresh) {
            // We won't be using cumulative return, so don't provide a path
            temp_cumul_img_path = std::nullopt;
        }

        // Project relative matrices to SE2
        T_est_rel = T_est_rel.toSE2().toSE3();
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        auto scan = std::make_shared<ba::LocalMapScan>(timestamp, num_checked, opts_, T_est_rel, T_gt_rel, temp_img_path, temp_cumul_img_path);
        scan_manager_.add_scan(scan);
        num_loaded++;
    }

    std::cout << "Loaded " << scan_manager_.num_scans() << "/" << num_scans << " scans." << std::endl;
    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;

    // Initialize voxel map around all scans
    std::cout << "Initializing voxel map..." << std::endl;
    for (int scan_id : scan_manager_.get_all_scan_ids()) {
        auto scan = scan_manager_.get_scan(scan_id);
        voxel_map_.init_map(scan->pose(), opts_.max_dist, scan_id);
    }

    std::pair<double, double> x_bounds = voxel_map_.x_bounds();
    std::pair<double, double> y_bounds = voxel_map_.y_bounds();

    std::cout << "Map bounds:" << std::endl;
    std::cout << "X: [" << x_bounds.first << ", " << x_bounds.second << "] meters" << std::endl;
    std::cout << "Y: [" << y_bounds.first << ", " << y_bounds.second << "] meters" << std::endl;
    std::cout << "Initialized voxel map with " << voxel_map_.size() << " voxels." << std::endl;

}

void MapProblem::finalize() {
    // Save results
    if (opts_.save_result) {
        result_.save_full_result();
    }
        

    // Visualize results
    if (opts_.visualize_result)
        result_.visualize_map();
}

}   // namespace ba