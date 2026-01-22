#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/solver/result.hpp"
#include <ba/solver/ba_solver.hpp>
#include <ba/problem/map_problem.hpp>

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Load in config from ba/config/map_config.yaml
    fs::path config_path = fs::path(__FILE__).parent_path().parent_path() / "config" / "map_config.yaml";
    YAML::Node config = YAML::LoadFile(config_path.string());
    ba::Options opts = ba::load_options(config);

    // Set up timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Set up output folder
    fs::path output_run_dir;
    if (opts.save_result) {
        if (!fs::exists(opts.output_path)) {
            fs::create_directories(opts.output_path);
        }
        // Get count of existing runs
        int num_runs = 0;
        for (const auto& entry : fs::directory_iterator(opts.output_path)) {
            if (entry.is_directory() && entry.path().filename().string().find("map_") == 0) {
                num_runs++;
            }
        }
        output_run_dir = opts.output_path / ("map_" + std::to_string(num_runs + 1));
        fs::create_directories(output_run_dir);
        std::cout << "Outputing results to: " << output_run_dir << std::endl;

        // Copy config to output folder
        fs::path output_config_path = output_run_dir / "map_config.yaml";
        fs::copy_file(config_path, output_config_path);
    }

    // Overwrite opts output path to run-specific folder
    opts.output_path = output_run_dir;

    ba::MapProblem problem(opts);
    problem.initialize();
    
    std::cout << "Solving for map..." << std::endl;
    ba::DrBASolver solver(problem);
    solver.update_map();
    

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total optimization time: " << total_time << " s" << std::endl;

    problem.finalize();

    return 0;
}
