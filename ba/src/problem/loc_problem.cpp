#include <ba/problem/loc_problem.hpp>
#include <ba/map/voxel_map.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ba/scans/scan.hpp>

namespace ba {

void LocProblem::init_scans_and_map() {
    // Load in map from estimates
    load_map_from_estimate();
    // Load in all scan poses to localize
    load_scans();
}

void LocProblem::load_map_from_estimate() {
    std::string map_seq_id = opts_.map_seq;
    // Load in voxel map from estimates
    std::string map_location = opts_.map_location.string() + "/voxel_map.bin";
    std::cout << "Loading voxel map from estimates: " << map_location << std::endl;
    voxel_map_.load_from_file(map_location);

    // Load groundtruth map poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;

    // TODO: Make seq_id configurable and make everything not hardcoded...
    // seq id for map should be encoded in the map object
    ba::load_groundtruth_poses_and_times(opts_.data_path / map_seq_id, all_gt_poses, all_gt_times);

    // Store gt poses corresponding to map poses
    fs::path all_map_img_dir = opts_.meas_path / map_seq_id / opts_.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_map_img_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    const std::vector<int> pose_ids = voxel_map_.pose_ids();
    const std::vector<lgmath::se2::Transformation> map_poses = voxel_map_.poses();
    for (size_t i = 0; i < pose_ids.size(); i++) {
        int pose_id = pose_ids[i];
        // Get timestamp from pose_id by looking at the filename for boreas-2024-12-03-12-54
        fs::path img_path = files[pose_id];
        int64_t timestamp = std::stoll(img_path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        gt_map_poses_.push_back(T_gt_abs);
    }

    std::cout << gt_map_poses_.size() << " groundtruth map poses loaded." << std::endl;
    std::cout << voxel_map_.poses().size() << " estimated map poses loaded." << std::endl;
}

void LocProblem::load_scans() {
    std::cout << "Loading scans for localization..." << std::endl;
    std::string seq_id = opts_.seq_ids[0];
    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = fs::temp_directory_path() / "dr_ba_temp_gauss_blur" / seq_id;
    fs::create_directories(temp_dir);

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id, all_gt_poses, all_gt_times);

    // Load DRO poses
    std::vector<lgmath::se3::Transformation> all_dro_poses;
    std::vector<double> all_dro_times;
    ba::load_dro_poses_and_times(opts_.meas_path / seq_id, all_dro_poses, all_dro_times);

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

    // Loop through all images
    std::cout << "Loading images from: " << all_img_dir << std::endl;
    int num_scans = files.size();
    opts_.end_frame = (opts_.end_frame == -1) ? (num_scans - 1) : opts_.end_frame;
    int num_checked = -1;
    int num_loaded = 0;
    for (const auto& path : files) {
        // Only consider files ending with .png
        if (path.extension() != ".png") {
            continue;
        }
        num_checked++;
        if (num_checked < opts_.start_frame || num_checked > opts_.end_frame) {
            continue;
        }
        
        // Load in scan pose
        int64_t timestamp = std::stoll(path.stem().string()); // in microseconds
        double timestamp_seconds = timestamp / 1e6; // convert to seconds

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

        // Load in all relevant poses
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamp_seconds);
        gt_poses_.push_back(T_gt_abs);
        lgmath::se3::Transformation T_dro_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamp_seconds);
        dro_poses_.push_back(T_dro_abs);

        // Just initialize all poses with identity matrix. We'll handle initialization later.
        lgmath::se3::Transformation T_init;
        auto scan = std::make_shared<ba::LocalMapScan>(timestamp, num_checked, opts_, T_init, T_init, temp_img_path, temp_cumul_img_path);
        scan_manager_.add_scan(scan);
        num_loaded++;
    }

    std::cout << "Loaded " << scan_manager_.num_scans() << "/" << num_scans << " scans." << std::endl;
    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;
}

void LocProblem::finalize() {
    // Implementation for finalizing the localization problem
    // This could involve saving results or cleaning up resources

}


}  // namespace ba