#include <ba/solver/result.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace ba {

void Result::save_rmse_cost_to_csv(const fs::path& optional_output_dir) const {

    std::cout << "Number of entries: " << cost_history_.size() << std::endl;
    std::cout << "Number of RMSE entries: " << rmse_history_.size() << std::endl;

    fs::path dir = optional_output_dir.empty() ? csv_path_: optional_output_dir;

    std::ofstream file(dir);
    file << "cost, rmse_x,rmse_y,rmse_yaw\n";
    for (std::size_t i = 0; i < rmse_history_.size(); ++i) {
        file << cost_history_[i] << "," << rmse_history_[i](0) << "," << rmse_history_[i](1) << "," << rmse_history_[i](2) << "\n";
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