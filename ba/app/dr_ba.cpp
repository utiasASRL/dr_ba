#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/solver/result.hpp"
#include <ba/problem/ba_problem.hpp>
#include <ba/solver/drba_solver.hpp>
#include <ba/solver/loc_solver.hpp>

#include <iostream>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

int main() {
    // Load in config from ba/config/dr_ba_config.yaml
    fs::path config_path = fs::path(__FILE__).parent_path().parent_path() / "config" / "dr_ba_config.yaml";
    YAML::Node config = YAML::LoadFile(config_path.string());
    ba::Options opts = ba::load_options(config);

    omp_set_num_threads(opts.num_threads);
    Eigen::setNbThreads(1);

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
            if (entry.is_directory() && entry.path().filename().string().find("run_") == 0) {
                num_runs++;
            }
        }
        output_run_dir = opts.output_path / ("run_" + std::to_string(num_runs + 1));
        fs::create_directories(output_run_dir);
        std::cout << "Outputing results to: " << output_run_dir << std::endl;

        // Copy config to output folder
        fs::path output_config_path = output_run_dir / "dr_ba_config.yaml";
        fs::copy_file(config_path, output_config_path);
    }

    // Overwrite opts output path to run-specific folder
    opts.output_path = output_run_dir;

    ba::BAProblem problem(opts);

    std::cout << "Starting optimization..." << std::endl;
    ba::DrBASolver solver(problem);
    solver.optimize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total optimization time: " << total_time << " s" << std::endl;

    problem.finalize();

    return 0;
}
