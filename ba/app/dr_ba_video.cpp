#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/solver/result.hpp"
#include <ba/problem/ba_problem.hpp>
#include <ba/solver/ba_solver.hpp>
#include <ba/solver/loc_solver.hpp>

#include <iostream>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    fs::path config_path;

    if (argc > 1) {
        // Use config path provided at runtime
        config_path = fs::path(argv[1]);
    } else {
        // Default config: ba/config/ba_config.yaml
        config_path = fs::path(__FILE__).parent_path().parent_path()
                    / "config" / "ba_vid_config.yaml";
    }

    if (!fs::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path.string());
    }

    std::cout << "Using config file: " << config_path.string() << std::endl;

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
        fs::path output_config_path = output_run_dir / "ba_vid_config.yaml";
        fs::copy_file(config_path, output_config_path);
    }

    // Overwrite opts output path to run-specific folder
    opts.output_path = output_run_dir;
    // opts.max_iterations = 1; // Only one at a time so we can visualize in between
    ba::BAProblem problem(opts);
    problem.initialize();

    int num_loaded_scans = problem.scan_manager().num_scans();

    std::cout << "Pre-optimization: " << num_loaded_scans << " scans loaded." << std::endl;

    // We want to re-initialize
    int temp_loaded_scans = 0;
    int max_frame_range = opts.frame_ranges.front().first + 1;
    int num_voxel_maps = 0;
    while (temp_loaded_scans < num_loaded_scans) {
        std::cout << "Initializing problem with frame range [0, " << max_frame_range << "]..." << std::endl;
        ba::Options opts_temp = ba::load_options(config);
        // Change frame range
        opts_temp.frame_ranges.clear();
        int start_frame = opts.frame_ranges.front().first;
        int end_frame = max_frame_range;
        opts_temp.frame_ranges.push_back({start_frame, end_frame});

        ba::BAProblem temp_problem(opts_temp);
        temp_problem.initialize();
        int new_loaded_scans = temp_problem.scan_manager().num_scans();
        if (new_loaded_scans == temp_loaded_scans) {
            max_frame_range++;
            continue;
        }
        temp_loaded_scans = new_loaded_scans;

        // Load in temporary solver and update map to visualize progress
        ba::DrBASolver solver_temp(temp_problem);
        solver_temp.update_map();

        // Save map
        std::string temp_filename = "voxel_map_" + std::to_string(num_voxel_maps) + ".bin";
        fs::path temp_voxel_path = output_run_dir / temp_filename;
        temp_problem.result().save_voxel_map(temp_voxel_path);

        max_frame_range++;
        num_voxel_maps++;
    }

    std::cout << "Starting optimization..." << std::endl;
    ba::DrBASolver solver(problem);
    double curr_cost = std::numeric_limits<double>::max() - 1000.0;
    double prev_cost = std::numeric_limits<double>::max();
    int num_iter = 0;
    double alpha = opts.alpha;
    while (num_iter < 12) {
        prev_cost = curr_cost;
        solver.optimize();
        curr_cost = solver.cost();
        
        // Update alpha manually
        alpha *= 0.8;
        solver.set_alpha(alpha);

        // Update map
        solver.update_map();

        // Save map
        std::string temp_filename = "voxel_map_" + std::to_string(num_voxel_maps) + ".bin";
        fs::path temp_voxel_path = output_run_dir / temp_filename;
        problem.result().save_voxel_map(temp_voxel_path);
        num_voxel_maps++;
        num_iter++;
    }

    // solver.optimize();
    // solver.update_map();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total optimization time: " << total_time << " s" << std::endl;

    problem.finalize();

    if (opts.save_result) {
        std::cout << "Results saved to: " << output_run_dir << std::endl;
    }

    return 0;
}
