#include <ba/solver/loc_solver.hpp>
#include <ba/problem/loc_problem.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <lgmath/se2/Operations.hpp>

namespace ba {

void LocSolver::construct_problem(const std::shared_ptr<Scan>& scan) {
    // // Reset cost
    // cost_ = 0.0;

    // // Initialize matrices
    // lhs_.setZero(3, 3);
    // rhs_.setZero(3);

    // // Loop through all voxels in the scan's coverage
    // for (const auto& voxel_idx : voxel_keys_) {
    //     double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
    //     double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
    //     double vox_intensity = voxel_map_.at(voxel_idx);

    //     // Interpolate intensity and Jacobian
    //     std::optional<Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
    //     // If scan is outside coverage, no intensity will be provided
    //     if (!interp_meas.has_value()) {
    //         continue;
    //     }

    //     // Weight everything by square root of measurement covariance
    //     double I_meas = interp_meas->intensity;
    //     double meas_cov = interp_meas->covariance;
    //     Eigen::Matrix<double, 1, 3> d_beta_d_T = - interp_meas->jacobian;// / std::sqrt(meas_cov);

    //     double err = vox_intensity - I_meas;

    //     if (err > 0.2) {
    //         continue;
    //     }

    //     lhs_ += d_beta_d_T.transpose() * d_beta_d_T;
    //     rhs_ += d_beta_d_T.transpose() * err;

    //     // Compute cost
    //     cost_ += 0.5 * std::pow(err, 2) / meas_cov;
    // }
}

void LocSolver::construct_problem(double downsample_factor) {
    // Implementation as above
}

bool LocSolver::solve() {
    return true;
}

void LocSolver::update_poses() {
}

void LocSolver::update_map() {
    // No map updating needed for localization solver
}

void LocSolver::optimize() {
    // Ensure problem is initialized
    if (!problem_.is_initialized()) {
        problem_.initialize();
    }

    Eigen::Vector3d avg_pose_error(0.0, 0.0, 0.0);
    int max_id = scan_manager_.get_all_scan_ids().back();
    std::vector<int> scan_id_list = scan_manager_.get_all_scan_ids();

    // Initialize pose using gt
    // Cast to LocProblem to access derived class methods
    auto& loc_problem = static_cast<LocProblem&>(problem_);
    // These are re-used, hence the names. They may not be the actual nearest node if we start
    // not at the start of the loop. loc_init_pose should still position us correctly.
    lgmath::se3::Transformation nearest_map_gt_pose = loc_problem.gt_map_poses().at(0);
    lgmath::se3::Transformation nearest_map_est_pose = loc_problem.voxel_map().poses().at(0).toSE3();
    lgmath::se3::Transformation loc_init_pose = loc_problem.gt_poses().at(0);

    // Get the initial pose within the map frame
    lgmath::se3::Transformation curr_pose = nearest_map_est_pose * nearest_map_gt_pose.inverse() * loc_init_pose;

    // Project to SE2 to get rid of any gt rounding in 3D dimensions
    curr_pose = curr_pose.toSE2().toSE3();
    double avg_runtime = 0.0;
    int64_t start_timestamp = scan_manager_.ref_timestamp();
    Eigen::Matrix3d curr_cov = 0.001 * Eigen::Matrix3d::Identity();
    for (size_t i = 0; i < scan_id_list.size(); i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int scan_id = scan_id_list.at(i);
        auto scan = scan_manager_.get_scan(scan_id);
        scan->set_pose(curr_pose);
        std::cout << "----------------------------------------" << std::endl;
        int64_t scan_timestamp = scan->timestamp();
        double time_from_start = static_cast<double>(scan_timestamp - start_timestamp) / 1e6;
        // Get voxels in range of initial pose
        voxel_keys_ = voxel_map_.get_voxels_in_range(scan->pose2d(), opts_.max_dist);
        std::cout << "Optimizing scan ID: " << scan_id << "/" << max_id
                  << " (timestamp: " << scan->timestamp() << ", " << time_from_start << " s from start)" << std::endl;
        scan->load_data();
        for (int iter = 0; iter < opts_.max_iterations; iter++) {
            // Reset cost
            cost_ = 0.0;

            // Construct problem
            // Initialize matrices
            lhs_.setZero(3, 3);
            rhs_.setZero(3);

            // Add prior
            if (opts_.use_odometry_prior) {
                // Form error between pose prior and current estimate
                lgmath::se3::Transformation T_prior_err = scan->pose().inverse() * curr_pose;
                Eigen::Matrix<double, 3, 1> prior_err = T_prior_err.toSE2().vec();
                Eigen::Matrix<double, 3, 3> prior_info = curr_cov.inverse();
                // lhs_ += prior_info;
                // rhs_ += prior_info * prior_err;

                Eigen::Matrix3d dro_process_noise = Eigen::Matrix3d::Zero();
                dro_process_noise(0, 0) = std::pow(opts_.odom_translation_std, 2);
                dro_process_noise(1, 1) = std::pow(opts_.odom_translation_std, 2);
                dro_process_noise(2, 2) = std::pow(opts_.odom_rotation_std * M_PI / 180.0, 2); // convert to radians

                lhs_ += dro_process_noise.inverse();
                rhs_ += dro_process_noise.inverse() * prior_err;
            }

            // Loop through all voxels in the scan's coverage
            int num_voxels_used = 0;
            std::vector<double> voxel_errors;
            for (const auto& voxel_idx : voxel_keys_) {
                double voxel_x = static_cast<double>(voxel_idx.first) * voxel_map_.res();
                double voxel_y = static_cast<double>(voxel_idx.second) * voxel_map_.res();
                double vox_intensity = voxel_map_.at(voxel_idx);

                // Interpolate intensity and Jacobian
                std::optional<Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
                // If scan is outside coverage, no intensity will be provided
                if (!interp_meas.has_value()) {
                    continue;
                }

                // Weight everything by square root of measurement covariance
                double I_meas = interp_meas->intensity;
                double meas_cov = interp_meas->covariance;
                double err_weight = 1.0 / meas_cov;

                // Compute unweighted error
                double err = (vox_intensity - I_meas);

                // Compute robust cost
                // double huber_thresh = 0.2;
                // double huber_weight = 1.0;
                // if (std::abs(err) > huber_thresh) {
                //     huber_weight = huber_thresh / std::abs(err);
                // }
                // err_weight *= huber_weight;
                double err_weight_sqrt = std::sqrt(err_weight);

                voxel_errors.push_back(std::abs(err));

                Eigen::Matrix<double, 1, 3> d_beta_d_T = - interp_meas->jacobian * err_weight_sqrt;
                err *= err_weight_sqrt;

                lhs_ += d_beta_d_T.transpose() * d_beta_d_T;
                rhs_ += d_beta_d_T.transpose() * err;

                // Compute cost
                cost_ += 0.5 * std::pow(err, 2) ;
                num_voxels_used++;
            }
            
            // std::cout << "Error min | max | mean : ";
            // if (!voxel_errors.empty()) {
            //     double err_min = *std::min_element(voxel_errors.begin(), voxel_errors.end());
            //     double err_max = *std::max_element(voxel_errors.begin(), voxel_errors.end());
            //     double err_mean = std::accumulate(voxel_errors.begin(), voxel_errors.end(), 0.0) / static_cast<double>(voxel_errors.size());
            //     std::cout << err_min << " | " << err_max << " | " << err_mean << std::endl;
            // } else {
            //     std::cout << "N/A (no voxels used)" << std::endl;
            // }
            
            if (num_voxels_used == 0) {
                throw std::runtime_error("Error: No voxels used in localization optimization! Check if your map and loc entries overlap?");
            }

            // Solve problem
            Eigen::Vector3d delta_xi = - alpha_ * lhs_.inverse() * rhs_;

            // Update poses
            scan->update_pose(delta_xi);

            if (iter != 0 && (delta_xi.norm() < opts_.convergence_tol || std::abs(prev_cost_ - cost_) < opts_.convergence_tol)) {
                std::cout << "Converged from: " << ((delta_xi.norm() < opts_.convergence_tol ) ? "small pose update." : "small cost change.") << std::endl;
                break;
            }

            prev_cost_ = cost_;
        }
        scan->unload_data();

        // Compute difference between prior and estimate
        lgmath::se3::Transformation pose_diff = scan->pose().inverse() * curr_pose;
        double pos_diff = pose_diff.r_ab_inb().head<2>().norm();
        std::cout << "Position difference from prior: " << pos_diff << " m." << std::endl;
        // if (pos_diff > 0.2) {
        //     std::cout << "Pose change from prior: " << pos_diff << " m. Ignoring estimate and using prior." << std::endl;
        // } else {
        //     curr_pose = scan->pose();
        // }
        curr_pose = scan->pose();
        

        // Compute errors
        // First, find nearest map pose to the scan's estimated pose
        std::cout << "Finding nearest map pose to scan..." << std::endl;
        double min_dist = std::numeric_limits<double>::max();
        int best_map_idx = -1;
        for (size_t j = 0; j < loc_problem.gt_map_poses().size(); j++) {
            lgmath::se3::Transformation map_est_pose = loc_problem.voxel_map().poses().at(j).toSE3();
            double dist = (map_est_pose.inverse() * scan->pose()).r_ab_inb().norm();

            // Prefer selecting nodes with similar orientation
            if (std::abs((map_est_pose.vec()(5) - scan->pose().vec()(5))) > M_PI / 2.0) {
                dist += 1000.0;
            }

            if (dist < min_dist) {
                min_dist = dist;
                nearest_map_gt_pose = loc_problem.gt_map_poses().at(j);
                // Also get the estimated map pose
                nearest_map_est_pose = map_est_pose;
                best_map_idx = static_cast<int>(j);
            }
        }
        if (best_map_idx == -1) {
            throw std::runtime_error("Error: Could not find nearest map pose! Check if your map and loc entries overlap?");
        }
        std::cout << "Nearest map pose index: " << best_map_idx << ", distance: " << min_dist << " m." << std::endl;

        // Compute estimated pose within local map
        lgmath::se3::Transformation loc_est_pose = nearest_map_est_pose.inverse() * curr_pose;
        scan->set_pose(loc_est_pose);

        // Compute gt pose within local map
        lgmath::se3::Transformation loc_gt_pose = nearest_map_gt_pose.inverse() * loc_problem.gt_poses().at(i);
        // Discard 3D info from the relative transform
        loc_gt_pose = loc_gt_pose.toSE2().toSE3();
        scan->set_gt_pose(loc_gt_pose);

        // Store result
        curr_cov = lhs_.inverse();
        Eigen::Matrix<double, 3, 1> loc_est_pose_xy = scan->pose().r_ab_inb();
        Eigen::Matrix<double, 3, 1> loc_gt_pose_xy = scan->gt_pose().r_ab_inb();
        double loc_est_yaw = scan->pose().vec()(5);
        double loc_gt_yaw = scan->gt_pose().vec()(5);
        LocProblem::LocResultEntry result_entry;
        result_entry.map_id = loc_problem.voxel_map().pose_ids().at(best_map_idx);
        result_entry.scan_id = scan->id();
        result_entry.est_x = loc_est_pose_xy(0);
        result_entry.est_y = loc_est_pose_xy(1);
        result_entry.est_yaw = loc_est_yaw;
        result_entry.gt_x = loc_gt_pose_xy(0);
        result_entry.gt_y = loc_gt_pose_xy(1);
        result_entry.gt_yaw = loc_gt_yaw;
        result_entry.std_x = std::sqrt(curr_cov(0,0));
        result_entry.std_y = std::sqrt(curr_cov(1,1));
        result_entry.std_yaw = std::sqrt(curr_cov(2,2));
        loc_problem.add_loc_result(result_entry);

        // Periodically save results to memory
        if (opts_.save_result && (i % 10 == 0 || i == scan_id_list.size() - 1)) {
            loc_problem.save_loc_results(opts_.output_path);
        }

        Eigen::Matrix<double, 6, 1> pose_error = scan->pose_error();
        std::cout << "Final pose error (m, m, deg): " << pose_error(0) << ", " << pose_error(1) << ", " << pose_error(5) * 180.0 / M_PI << std::endl;
        avg_pose_error(0) += pose_error(0) * pose_error(0);
        avg_pose_error(1) += pose_error(1) * pose_error(1);
        avg_pose_error(2) += pose_error(5) * pose_error(5);

        // If pose error is larger than max_dist, localization has failed and will not recover
        if (std::sqrt(pose_error(0) * pose_error(0) + pose_error(1) * pose_error(1)) > 15.0) {
            throw std::runtime_error("Error: Localization has diverged! Pose error exceeded maximum map range.");
        }

        // Propagate curr_pose using DRO estimates
        if (i == scan_id_list.size() - 1) {
            break;
        }

        // Set curr_pose for next iteration
        lgmath::se3::Transformation dro_rel_pose;
        lgmath::se3::Transformation curr_dro_pose = loc_problem.dro_poses().at(i);
        lgmath::se3::Transformation next_dro_pose = loc_problem.dro_poses().at(i + 1);
        dro_rel_pose = curr_dro_pose.inverse() * next_dro_pose;
        Eigen::Vector3d dro_rel_pose_xi = dro_rel_pose.toSE2().vec();
        curr_pose = curr_pose * dro_rel_pose;

        // Compute Jacobians
        Eigen::Matrix3d dro_noise_jacobian = lgmath::se2::vec2jac(dro_rel_pose_xi);
        Eigen::Matrix3d dro_pose_jacobian = - lgmath::se2::tranAd(dro_rel_pose.toSE2().inverse().matrix());

        // Load in process noise
        Eigen::Matrix3d dro_process_noise = Eigen::Matrix3d::Zero();
        dro_process_noise(0, 0) = std::pow(opts_.odom_translation_std, 2);
        dro_process_noise(1, 1) = std::pow(opts_.odom_translation_std, 2);
        dro_process_noise(2, 2) = std::pow(opts_.odom_rotation_std * M_PI / 180.0, 2); // convert to radians

        // Propagate covariance
        curr_cov = dro_pose_jacobian * curr_cov * dro_pose_jacobian.transpose() + dro_noise_jacobian * dro_process_noise * dro_noise_jacobian.transpose();

        // For debug, set current pose to groundtruth
        // curr_pose = nearest_map_est_pose * nearest_map_gt_pose.inverse() * loc_problem.gt_poses().at(i + 1);
        // curr_pose = curr_pose.toSE2().toSE3();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        avg_runtime += static_cast<double>(duration);
        std::cout << "Average scan runtime: " << avg_runtime / static_cast<double>(i + 1) << " ms." << std::endl;
    }
    
    
    std::cout << "Localization of " << scan_id_list.size() << " scans took " << avg_runtime / static_cast<double>(scan_id_list.size()) << " ms on average." <<  std::endl;

    avg_pose_error /= static_cast<double>(scan_id_list.size());
    avg_pose_error = avg_pose_error.cwiseSqrt();
    avg_pose_error(2) = avg_pose_error(2) * 180.0 / M_PI;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "RMSE over all scans (m, m, deg):\n" << avg_pose_error.transpose() << std::endl;
}


} // namespace ba