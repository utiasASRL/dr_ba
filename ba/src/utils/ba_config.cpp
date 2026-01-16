#include "ba/utils/ba_config.hpp"
#include <stdexcept>

namespace ba {

Options load_options(const YAML::Node& config) {
    Options opts;

    if (config["map"]) {
        if (config["map"]["voxel_res"])
            opts.voxel_res = config["map"]["voxel_res"].as<double>();
        else
            throw std::runtime_error("Voxel resolution not found in config file.");
    } else {
        throw std::runtime_error("Map configuration not found in config file.");
    }

    if (config["output"]) {
        if (config["output"]["save_result"])
            opts.save_result = config["output"]["save_result"].as<bool>();
        else
            opts.save_result = true;
        if (config["output"]["output_path"])
            opts.output_path = std::filesystem::path(config["output"]["output_path"].as<std::string>());
        else {
            if (opts.save_result)
                throw std::runtime_error("Output path not found in config file.");
        }
        if (config["output"]["visualize"])
            opts.visualize_result = config["output"]["visualize"].as<bool>();
        else
            opts.visualize_result = true;
    } else {
        throw std::runtime_error("Output configuration not found in config file.");
    }

    if (config["input"]) {
        if (config["input"]["data_path"])
            opts.data_path = std::filesystem::path(config["input"]["data_path"].as<std::string>());
        else
            throw std::runtime_error("Data path not found in config file.");
        if (config["input"]["meas_path"])
            opts.meas_path = std::filesystem::path(config["input"]["meas_path"].as<std::string>());
        else
            throw std::runtime_error("Measurement path not found in config file.");
        if (config["input"]["seq_ids"]) {
            opts.seq_ids.clear();
            for (const auto& id_node : config["input"]["seq_ids"]) {
                opts.seq_ids.push_back(id_node.as<std::string>());
            }
        } else
            throw std::runtime_error("Sequence IDs not found in config file.");
        if (config["input"]["max_dist"])
            opts.max_dist = config["input"]["max_dist"].as<double>();
        else
            throw std::runtime_error("Max distance not found in config file.");
        if (config["input"]["gauss_blur_sigma"])
            opts.gauss_blur_sigma = config["input"]["gauss_blur_sigma"].as<double>();
        else
            throw std::runtime_error("Gaussian blur sigma not found in config file.");
        if (config["input"]["init_poses"])
            opts.init_poses = config["input"]["init_poses"].as<std::string>();
        else
            throw std::runtime_error("Initial poses type not found in config file.");
        if (config["input"]["init_translation_std"])
            opts.init_translation_std = config["input"]["init_translation_std"].as<double>();
        else
            throw std::runtime_error("Initial translation std not found in config file.");
        if (config["input"]["init_rotation_std"])
            opts.init_rotation_std = config["input"]["init_rotation_std"].as<double>();
        else
            throw std::runtime_error("Initial rotation std not found in config file.");
        if (config["input"]["input_type"])
            opts.input_type = config["input"]["input_type"].as<std::string>();
        else
            throw std::runtime_error("Input type not found in config file.");
        if (config["input"]["local_map_res"])
            opts.local_map_res = config["input"]["local_map_res"].as<double>();
        else
            throw std::runtime_error("Local map resolution not found in config file.");
    } else {
        throw std::runtime_error("Input configuration not found in config file.");
    }

    if (config["keyframing"]) {
        if (config["keyframing"]["num_frames"])
            opts.num_frames = config["keyframing"]["num_frames"].as<int>();
        else
            throw std::runtime_error("Number of frames not found in config file.");
        if (config["keyframing"]["max_kf_dist"])
            opts.max_kf_dist = config["keyframing"]["max_kf_dist"].as<double>();
        else
            throw std::runtime_error("Max keyframe distance not found in config file.");
        if (config["keyframing"]["max_kf_rot"])
            opts.max_kf_rot = config["keyframing"]["max_kf_rot"].as<double>();
        else
            throw std::runtime_error("Max keyframe rotation not found in config file.");
    } else {
        throw std::runtime_error("Keyframing configuration not found in config file.");
    }

    if (config["optimization"]) {
        if (config["optimization"]["max_iterations"])
            opts.max_iterations = config["optimization"]["max_iterations"].as<int>();
        else
            throw std::runtime_error("Max iterations not found in config file.");
        if (config["optimization"]["convergence_tol"])
            opts.convergence_tol = config["optimization"]["convergence_tol"].as<double>();
        else
            throw std::runtime_error("Convergence tolerance not found in config file.");
        if (config["optimization"]["alpha"])
            opts.alpha = config["optimization"]["alpha"].as<double>();
        if (config["optimization"]["adaptive_alpha"])
            opts.adaptive_alpha = config["optimization"]["adaptive_alpha"].as<bool>();
        if (config["optimization"]["prior_map_std"])
            opts.prior_map_std = config["optimization"]["prior_map_std"].as<double>();
        else
            throw std::runtime_error("Prior map standard deviation not found in config file.");
        if (config["optimization"]["meas_std"])
            opts.meas_std = config["optimization"]["meas_std"].as<double>();
        else
            throw std::runtime_error("Measurement standard deviation not found in config file.");
        if (config["optimization"]["rel_pose_prior"]) {
            if (config["optimization"]["rel_pose_prior"]["use_pose_prior"])
                opts.use_rel_pose_prior = config["optimization"]["rel_pose_prior"]["use_pose_prior"].as<bool>();
            else
                throw std::runtime_error("Use relative pose prior flag not found in config file.");
            if (config["optimization"]["rel_pose_prior"]["translation_std"])
                opts.rel_pose_prior_translation_std = config["optimization"]["rel_pose_prior"]["translation_std"].as<double>();
            else
                throw std::runtime_error("Relative pose prior translation std not found in config file.");
            if (config["optimization"]["rel_pose_prior"]["rotation_std"])
                opts.rel_pose_prior_rotation_std = config["optimization"]["rel_pose_prior"]["rotation_std"].as<double>();
        } else {
            throw std::runtime_error("Relative pose prior configuration not found in config file.");
        }
        if (config["optimization"]["range_factor"])
            opts.range_factor = config["optimization"]["range_factor"].as<double>();
        else
            throw std::runtime_error("Range factor not found in config file.");
        if (config["optimization"]["cumul_thresh"])
            opts.cumul_thresh = config["optimization"]["cumul_thresh"].as<double>();
        else
            throw std::runtime_error("Cumulative threshold not found in config file.");
        if (config["optimization"]["num_coarse_iterations"])
            opts.num_coarse_iterations = config["optimization"]["num_coarse_iterations"].as<int>();
        else
            throw std::runtime_error("Number of coarse iterations not found in config file.");
        if (config["optimization"]["coarse_downsample"])
            opts.coarse_downsample = config["optimization"]["coarse_downsample"].as<double>();
        else
            throw std::runtime_error("Coarse downsample factor not found in config file.");
        if (config["optimization"]["refine_downsample"])
            opts.refine_downsample = config["optimization"]["refine_downsample"].as<double>();
        else
            throw std::runtime_error("Refine downsample factor not found in config file.");
        if (config["optimization"]["tile_size"])
            opts.tile_size = config["optimization"]["tile_size"].as<double>();
    }

    return opts;
}

} // namespace ba