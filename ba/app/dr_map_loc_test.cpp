#include <ba/map/voxel_map.hpp>
#include <ba/scans/manager.hpp>
#include "ba/utils/ba_config.hpp"
#include "ba/solver/result.hpp"
#include <ba/solver/ba_solver.hpp>
#include <ba/problem/map_problem.hpp>
#include <ba/problem/loc_problem.hpp>
#include <ba/solver/loc_solver.hpp>

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
                    / "config" / "map_loc_config.yaml";
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
            if (entry.is_directory() && entry.path().filename().string().find("map_loc_") == 0) {
                num_runs++;
            }
        }
        output_run_dir = opts.output_path / ("map_loc_" + std::to_string(num_runs + 1));
        fs::create_directories(output_run_dir);
        std::cout << "Outputing results to: " << output_run_dir << std::endl;

        // Copy config to output folder
        fs::path output_config_path = output_run_dir / "map_loc_config.yaml";
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

    // Now do the loc part
    std::cout << "Starting localization..." << std::endl;
    // Reload opts since map overwrites some options... need to clean this up
    opts = ba::load_options(config);
    // Overwrite opts output path to run-specific folder
    opts.output_path = output_run_dir;
    start_time = std::chrono::high_resolution_clock::now();

    opts.map_location = output_run_dir;

    ba::LocProblem loc_problem(opts);
    loc_problem.initialize();
    std::cout << "Loaded voxel map with " << loc_problem.voxel_map().size() << " voxels." << std::endl;
    ba::LocSolver loc_solver(loc_problem);
    loc_solver.optimize();
    end_time = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Total localization time: " << total_time << " s" << std::endl;
    loc_problem.finalize();

    if (opts.save_result) {
        std::cout << "Results saved to: " << output_run_dir << std::endl;
    }

    return 0;
}
