#include <ba/problem/loc_problem.hpp>
#include <ba/map/voxel_map.hpp>
#include <ba/utils/io_utils.hpp>
#include <random>
#include <opencv2/opencv.hpp>
#include <ba/scans/local_map_scan.hpp>
#include <ba/scans/scan.hpp>

namespace ba {

void LocProblem::get_scan_indeces() {
    // For localization load all scans
    std::string seq_id = opts_.seq_ids[0];
    fs::path all_img_dir = opts_.meas_path / seq_id / opts_.input_type;
    int count = -1;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        // Only consider files ending with .png
        if (entry.path().extension() != ".png") {
            continue;
        }
        if (entry.is_regular_file()) {
            count++;
            if (count < opts_.start_frame) {
                continue;
            }
            scan_indices_.push_back(count);
            if (opts_.end_frame >= 0 && count >= opts_.end_frame) {
                break;
            }
        }
    }
    // Sort scan indices
    std::sort(scan_indices_.begin(), scan_indices_.end());
}
    
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

    // Load groundtruth poses
    std::vector<lgmath::se3::Transformation> all_gt_poses;
    std::vector<double> all_gt_times;
    ba::load_groundtruth_poses_and_times(opts_.data_path / seq_id, all_gt_poses, all_gt_times);

    // Load DRO poses
    std::vector<lgmath::se3::Transformation> all_dro_poses;
    std::vector<double> all_dro_times;
    ba::load_dro_poses_and_times(opts_.meas_path / seq_id, all_dro_poses, all_dro_times);

    // Loop through indeces
    for (size_t i=0; i < scan_indices_.size(); i++) {
        int idx = scan_indices_[i];

        // Load in all relevant poses
        lgmath::se3::Transformation T_gt_abs = ba::get_interpolated_pose(all_gt_poses, all_gt_times, timestamps_[i]/1e6);
        gt_poses_.push_back(T_gt_abs);
        lgmath::se3::Transformation T_dro_abs = ba::get_interpolated_pose(all_dro_poses, all_dro_times, timestamps_[i]/1e6);
        dro_poses_.push_back(T_dro_abs);

        // Load in image paths
        fs::path img_path = img_paths_[i];
        std::optional<fs::path> cumul_img_path = std::nullopt;
        if (opts_.use_cumul_thresh) {
            if (cumul_paths_.empty()) {
                throw std::runtime_error("Cumulative image paths are empty but use_cumul_thresh is true.");
            }
            cumul_img_path = cumul_paths_[i];
        }

        // Just initialize all poses with identity matrix. We'll handle initialization later.
        lgmath::se3::Transformation T_init;
        auto scan = std::make_shared<ba::LocalMapScan>(timestamps_[i], idx, opts_, T_init, T_init, img_path, cumul_img_path);
        scan_manager_.add_scan(scan);
    }

    std::cout << "Scan manager has " << scan_manager_.num_scans() << " scans." << std::endl;
}

void LocProblem::finalize() {
    // Implementation for finalizing the localization problem
    // This could involve saving results or cleaning up resources

}


}  // namespace ba