#include <ba/problem/ba_problem.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>

namespace ba {

void BAProblem::init_scans() {
    std::string seq_id = opts_.seq_ids[0];
    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = fs::temp_directory_path() / "dr_ba_temp_gauss_blur";
    fs::create_directories(temp_dir);
    // Clean up temp folder if it already exists
    for (const auto& entry : fs::directory_iterator(temp_dir)) {
        fs::remove_all(entry.path());
    }

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
    int max_number_frames = opts_.num_frames > 0 ? opts_.num_frames : files.size();
    for (const auto& path : files) {
        if (num_loaded >= max_number_frames) break;
        num_checked++;

        // Load in scan pose
        int64_t timestamp = std::stoll(path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds

        // Load in gt pose
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);

        // Load in initial guess pose
        lgmath::se3::Transformation T_est_rel(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
        lgmath::se3::Transformation T_est_abs;
        if (opts_.init_poses == "pogo") {
            T_est_abs = ba::get_interpolated_pose(all_pogo_poses, all_pogo_times, timestamp_seconds);
        } else if (opts_.init_poses == "gt") {
            T_est_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        } else if (opts_.init_poses == "dro") {
            T_est_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamp_seconds);
        } else {
            throw std::invalid_argument("Invalid init_poses option: " + opts_.init_poses);
        }
        T_est_rel = T_est_abs_0.inverse() * T_est_abs;
        // Add noise to gt pose sampled from uniform distribution
        Eigen::Vector3d noise;
        noise << translation_dist(rng), translation_dist(rng), rotation_dist(rng);
        lgmath::se3::Transformation T_noise = lgmath::se2::Transformation(noise).toSE3();
        T_est_rel = T_est_rel * T_noise;

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

        // Get relative gt transform
        lgmath::se3::Transformation T_gt_rel = T_gt_abs_0.inverse() * T_gt_abs;

        // TODO: Add support for more than just local_maps
        if (opts_.input_type != "local_maps") {
            throw std::invalid_argument("Input type " + opts_.input_type + " not supported yet.");
        }

        // Load in image as Eigen matrix
        cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        // Apply Gaussian blur
        if (opts_.gauss_blur_sigma > 0.0) {
            int ksize = static_cast<int>(std::ceil(opts_.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
            cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts_.gauss_blur_sigma);
        }
        // Convert to CV_32F and normalize to [0, 1]
        img.convertTo(img, CV_32F, 1.0 / 255.0);
        // Save blurred image to temp directory for easy loading
        fs::path temp_img_path = temp_dir / (std::to_string(num_checked) + ".png");
        ba::save_img_bin(temp_img_path, img);

        // Load in cumulative return image
        fs::path cumul_path = cumul_files[num_checked];
        cv::Mat cumul_img = cv::imread(cumul_path.string(), cv::IMREAD_GRAYSCALE);
        // Convert to CV_32F and normalize to [0, 1]
        cumul_img.convertTo(cumul_img, CV_32F, 1.0 / 255.0);
        // Save cumulative image to temp directory for easy loading
        std::optional<fs::path> temp_cumul_img_path = temp_dir / (std::to_string(num_checked) + "_cumul.png");
        ba::save_img_bin(temp_cumul_img_path.value(), cumul_img);

        // Project relative matrices to SE2
        T_est_rel = T_est_rel.toSE2().toSE3();
        T_gt_rel = T_gt_rel.toSE2().toSE3();

        // Create scan object
        if (opts_.cumul_thresh > 1.0 || opts_.cumul_thresh < 0.0) {
            // We won't be using cumulative return, so don't provide a path
            temp_cumul_img_path = std::nullopt;
        }
        auto scan = std::make_shared<ba::LocalMapScan>(timestamp, num_checked, opts_, T_est_rel, T_gt_rel, temp_img_path, temp_cumul_img_path);
        if (opts_.fix_first_scan && num_loaded == 0) {
            scan->set_fixed(true); // Fix the first scan's pose
        }
        scan_manager_.add_scan(scan);
        num_loaded++;
    }

    std::cout << "Loaded " << scan_manager_.num_scans() << "/" << num_scans << " scans." << std::endl;
    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;
}


void BAProblem::init_map() {
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

void BAProblem::finalize() {
    // Save results
    if (opts_.save_result)
        result_.save_full_result();

    // Visualize results
    if (opts_.visualize_result)
        result_.visualize_all_results();
}

}   // namespace ba