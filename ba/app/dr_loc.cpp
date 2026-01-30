#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include "ba/utils/ba_config.hpp"

#include <ba/problem/loc_problem.hpp>
#include <ba/solver/loc_solver.hpp>
#include <ba/solver/ba_solver.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    fs::path config_path;

    if (argc > 1) {
        // Use config path provided at runtime
        config_path = fs::path(argv[1]);
    } else {
        // Default config: ba/config/ba_config.yaml
        config_path = fs::path(__FILE__).parent_path().parent_path()
                    / "config" / "loc_config.yaml";
    }

    if (!fs::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path.string());
    }

    std::cout << "Using config file: " << config_path.string() << std::endl;

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
            if (entry.is_directory() && entry.path().filename().string().find("loc_") == 0) {
                num_runs++;
            }
        }
        output_run_dir = opts.output_path / ("loc_" + std::to_string(num_runs + 1));
        fs::create_directories(output_run_dir);
        std::cout << "Outputing results to: " << output_run_dir << std::endl;

        // Copy config to output folder
        fs::path output_config_path = output_run_dir / "loc_config.yaml";
        fs::copy_file(config_path, output_config_path);

        // Copy voxel map to output folder
        fs::path voxel_map_src = opts.map_location.string() + "/voxel_map.bin";
        fs::path voxel_map_dst = output_run_dir / "voxel_map.bin";
        fs::copy_file(voxel_map_src, voxel_map_dst);
    }

    // Overwrite opts output path to run-specific folder
    opts.output_path = output_run_dir;

    ba::LocProblem problem(opts);
    auto &voxel_map = problem.voxel_map();
    problem.initialize();

    std::cout << "Loaded voxel map with " << voxel_map.size() << " voxels." << std::endl;
    
    ba::LocSolver solver(problem);
    solver.optimize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total optimization time: " << total_time << " s" << std::endl;

    if (opts.save_result) {
        std::cout << "Results saved to: " << output_run_dir << std::endl;
    }

    problem.finalize();

    return 0;
}
