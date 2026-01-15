#include <ba/solver/result.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <lgmath/se2/Transformation.hpp>

namespace ba {

void Result::save_rmse_cost_to_csv(const fs::path& optional_output_dir) const {
    fs::path dir = optional_output_dir.empty() ? csv_path_: optional_output_dir;

    std::ofstream file(dir);
    file << "cost,ate,rmse_x,rmse_y,rmse_yaw\n";
    for (std::size_t i = 0; i < rmse_history_.size(); ++i) {
        file << cost_history_[i] << "," << ate_history_[i] << "," << rmse_history_[i](0) << "," << rmse_history_[i](1) << "," << rmse_history_[i](2) << "\n";
    }
}

void Result::save_poses_to_csv(const fs::path& optional_output_dir) const {
    fs::path dir = optional_output_dir.empty() ? poses_path_ : optional_output_dir;

    std::ofstream file(dir);
    file << "scan_id, x, y, yaw, x_gt, y_gt, yaw_gt\n";
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();
    for (int i = 0; i < scan_manager_.num_scans(); ++i) {
        auto scan = scan_manager_.get_scan(scan_id_list[i]);
        lgmath::se2::Transformation T_est = scan->pose2d();
        Eigen::Vector2d t_est = T_est.r_ab_inb();
        double yaw_est = T_est.vec()(2);
        lgmath::se2::Transformation T_gt = scan->gt_pose2d();
        Eigen::Vector2d t_gt = T_gt.r_ab_inb();
        double yaw_gt = T_gt.vec()(2);

        file << scan->id() << "," << t_est(0) << "," << t_est(1) << "," << yaw_est << ","
             << t_gt(0) << "," << t_gt(1) << "," << yaw_gt << "\n";
    }
}

void Result::save_voxel_map() const {
    voxel_map_.save_to_file(voxel_path_.string());
}

void Result::visualize_all_results() {
    // Visualize voxel map
    voxel_map_.visualize();

    // Check if output_dir is empty
    std::string cmd;
    fs::path temp_dir;
    if (output_dir_.empty()) {
        temp_dir = fs::temp_directory_path() / "dr_ba_temp_visualization";
        fs::create_directories(temp_dir);
        std::cout << "Output directory not set. Using temporary directory: " << temp_dir.string() << std::endl;
        fs::path temp_csv_path = temp_dir / "rmse_cost_history.csv";
        save_rmse_cost_to_csv(temp_csv_path);
        cmd = "python3 /home/dl/Documents/phd/dev/dr_ba/ba/app/plot_errors.py " + temp_csv_path.string();
    } else {
        cmd = "python3 /home/dl/Documents/phd/dev/dr_ba/ba/app/plot_errors.py " + csv_path_.string() + " " + output_dir_.string();
    }

    // Visualize RMSE and cost history
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("Python script failed");
    }
    
    // Clean up temporary directory
    if (output_dir_.empty()) {
        std::cout << "Removing temporary directory: " << temp_dir.string() << std::endl;
        fs::remove_all(temp_dir);
    }

}

}   // namespace ba