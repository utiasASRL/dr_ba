#include <ba/solver/solver.hpp>
#include <iostream>

namespace ba {

void Solver::construct_problem(ba::ScanManager &scan_manager, double downsample_factor) {
    // Load in constants
    std::vector<int> scan_id_list = scan_manager.get_all_scan_ids();
    int states_size = (scan_manager.num_scans() - 1) * 3; // SE2 poses with first pose fixed

    // Reset cost
    cost_ = 0.0;

    // Downsample desired voxels
    voxel_keys_ = vox_map_.get_sorted_keys_downsampled(downsample_factor);
    int voxels_size = static_cast<int>(voxel_keys_.size());

    // Initialize matrices
    H_TT_.setZero(states_size, states_size);
    J_T_B_.setZero(states_size);
    H_TM_.setZero(states_size, voxels_size);
    J_M_B_.setZero(voxels_size);
    H_MM_diag_.setZero(voxels_size);

    // Load in relative SE2 pose priors
    if (opts_.use_rel_pose_prior) {
        for (const auto& prior : pose_priors_) {
            int scan_id_a = prior.first.first;
            int scan_id_b = prior.first.second;
            lgmath::se2::Transformation T_prior = prior.second.toSE2();
            // T_prior is the transform from scan A to scan B
            // Load in poses
            lgmath::se2::Transformation T_a = scan_manager.get_scan(scan_id_a)->pose2d();
            lgmath::se2::Transformation T_b = scan_manager.get_scan(scan_id_b)->pose2d();
            // Compute error
            lgmath::se2::Transformation T_err = T_prior * (T_b.inverse() * T_a).inverse();
            Eigen::Matrix<double, 3, 1> err_vec = T_err.vec();
            // Compute Jacobians
            Eigen::Matrix<double, 3, 3> Ad_T_b_inv = T_b.inverse().adjoint();
            Eigen::Matrix<double, 3, 3> d_e_d_Ta = Ad_T_b_inv;
            Eigen::Matrix<double, 3, 3> d_e_d_Tb = - Ad_T_b_inv;
            // Weight by prior covariance
            Eigen::Vector3d prior_std_diag;
            prior_std_diag << opts_.rel_pose_prior_translation_std, opts_.rel_pose_prior_translation_std,
                                opts_.rel_pose_prior_rotation_std * M_PI / 180.0;
            int num_steps = scan_id_b - scan_id_a;
            prior_std_diag *= static_cast<double>(num_steps); // Scale covariance with number of steps
            Eigen::Matrix3d prior_cov_sqrt_inv = prior_std_diag.cwiseInverse().asDiagonal();
            // Weight Jacobians and error
            d_e_d_Ta = prior_cov_sqrt_inv * d_e_d_Ta;
            d_e_d_Tb = prior_cov_sqrt_inv * d_e_d_Tb;
            Eigen::Matrix<double, 3, 1> err_vec_weighted = prior_cov_sqrt_inv * err_vec;
            // Assemble into H and J
            // Get scan index from scan id
            int state_idx_a = -1;
            int state_idx_b = -1;
            for (std::size_t idx = 0; idx < scan_id_list.size(); ++idx) {
                if (scan_id_list[idx] == scan_id_a) {
                    state_idx_a = (idx - 1) * 3;
                }
                if (scan_id_list[idx] == scan_id_b) {
                    state_idx_b = (idx - 1) * 3;
                }
            }

            if (state_idx_a < 0 && state_idx_b < 0) {
                // Something is wrong!
                throw std::runtime_error("Both scans in prior are fixed poses.");
            }

            if (state_idx_a < 0) {
                // First pose is fixed, only assemble for b
                H_TT_.block<3,3>(state_idx_b, state_idx_b) += d_e_d_Tb.transpose() * d_e_d_Tb;
                J_T_B_.segment<3>(state_idx_b) += d_e_d_Tb.transpose() * err_vec_weighted;
            } else {
                H_TT_.block<3,3>(state_idx_a, state_idx_a) += d_e_d_Ta.transpose() * d_e_d_Ta;
                H_TT_.block<3,3>(state_idx_a, state_idx_b) += d_e_d_Ta.transpose() * d_e_d_Tb;
                H_TT_.block<3,3>(state_idx_b, state_idx_a) += d_e_d_Tb.transpose() * d_e_d_Ta;
                J_T_B_.segment<3>(state_idx_a) += d_e_d_Ta.transpose() * err_vec_weighted;
            }

            cost_ += 0.5 * err_vec.transpose() * err_vec;
        }
    }

    // Loop through all voxels
    for (int v_idx = 0; v_idx < voxels_size; v_idx++) {
        const auto& voxel_idx = voxel_keys_[v_idx];
        double voxel_x = static_cast<double>(voxel_idx.first) * vox_map_.res();
        double voxel_y = static_cast<double>(voxel_idx.second) * vox_map_.res();
        double vox_intensity = vox_map_.at(voxel_idx);
        bool voxel_covered = false;
        for (int scan_idx = 0; scan_idx < scan_manager.num_scans(); scan_idx++) {
            int scan_id = scan_id_list[scan_idx];
            auto scan = scan_manager.get_scan(scan_id);

            // Interpolate intensity and Jacobian
            
            std::optional<ba::Scan::Measurement> interp_meas = scan->interpolate(voxel_x, voxel_y);
            // If scan is outside coverage, no intensity will be provided
            if (!interp_meas.has_value()) {
                continue;
            }
            voxel_covered = true;
            double I_meas = interp_meas->intensity;
            Eigen::Matrix<double, 1, 3> d_I_d_T = interp_meas->jacobian;
            double meas_cov = interp_meas->covariance;
            // Weight everything by square root of measurement covariance
            double I_meas_weighted = I_meas / std::sqrt(meas_cov);
            Eigen::Matrix<double, 1, 3> d_e_d_T = - d_I_d_T / std::sqrt(meas_cov); // e = I_vox - I_meas
            double d_e_d_M = 1.0 / std::sqrt(meas_cov);

            // Assemble Jacobians and Hessians
            if (scan_idx != 0) {
                // First pose is fixed
                int state_idx = (scan_idx - 1) * 3;
                // H_TT
                H_TT_.block<3,3>(state_idx, state_idx) += d_e_d_T.transpose() * d_e_d_T;
                // H_TM
                H_TM_.block<3,1>(state_idx, v_idx) += d_e_d_T.transpose() * d_e_d_M;
                // J_T_B
                J_T_B_.segment<3>(state_idx) += d_e_d_T.transpose() * I_meas_weighted;
            }

            // H_MM
            H_MM_diag_(v_idx) += d_e_d_M * d_e_d_M;
            // J_M_B
            J_M_B_(v_idx) += d_e_d_M * I_meas_weighted;

            // Update cost (purely for monitoring convergence)
            cost_ += 0.5 * std::pow((vox_intensity - I_meas), 2);
        }
        if (!voxel_covered) {
            // Add zero prior to this voxel
            H_MM_diag_(v_idx) += 1.0 / (opts_.prior_map_std * opts_.prior_map_std);
            cost_ += 0.5 * std::pow(vox_intensity, 2);
        }
    }
}

bool Solver::solve() {
    // H_MM_diag_ is the vector of diagonal elements
    Eigen::VectorXd H_MM_inv_diag = H_MM_diag_.cwiseInverse();

    // Compute lhs directly
    Eigen::MatrixXd lhs = H_TT_;

    // Subtract H_TM * H_MM^-1 * H_TM^T efficiently
    for (int j = 0; j < H_TM_.cols(); ++j) {
        lhs.noalias() -= H_TM_.col(j) * (H_TM_.col(j).array() * H_MM_inv_diag(j)).matrix().transpose();
    }

    // Compute rhs
    Eigen::VectorXd rhs = - H_TM_ * (J_M_B_.array() / H_MM_diag_.array()).matrix() + J_T_B_;

    // Regularization
    // Eigen::VectorXd diag = lhs.diagonal();
    // lhs.diagonal().array() += lambda_ * diag.array().max(1e-8);

    lhs += 1e-8 * Eigen::MatrixXd::Identity(lhs.rows(), lhs.cols());

    // Solve
    del_x_ = lhs.selfadjointView<Eigen::Upper>().ldlt().solve(rhs);

    // // Recompute cost after applying del_x_ on a copy of the scan manager
    // ba::ScanManager scan_manager_copy = scan_manager_.deep_copy();
    // update_poses(scan_manager_copy);
    // construct_problem(scan_manager_copy, 1.0);
    // update_map();

    // std::cout << "lambda: " << lambda_
    //         << " |delta|: " << del_x_.norm()
    //         << " cost: " << cost_ << std::endl;

    // // Update lambda_
    // if (cost_ < prev_cost_ || del_x_.norm() < opts_.convergence_tol) {
    //     // Decrease lambda
    //     lambda_ = std::max(lambda_ * 0.8, 1e-8);
    //     return true;
    // } else {
    //     // Increase lambda
    //     lambda_ *= 2.0;
    //     return false;
    // }

    return true;
}

void Solver::update_poses(ba::ScanManager &scan_manager) {
    // Update poses
    if (del_x_.size() == 0) {
        throw std::runtime_error("No pose updates available. Have you run solve()?");
    }
    if (del_x_.size() != (scan_manager.num_scans() - 1) * 3) {
        throw std::runtime_error("Size of pose updates does not match number of scans.");
    }
    std::vector<int> scan_id_list = scan_manager.get_all_scan_ids();
    for (int scan_idx = 1; scan_idx < scan_manager.num_scans(); scan_idx++) {
        // Extract delta for this scan
        int state_idx = (scan_idx - 1) * 3;
        Eigen::Matrix<double, 3, 1> delta_xi = del_x_.segment<3>(state_idx);
        // Load in scan and update pose
        auto scan = scan_manager.get_scan(scan_id_list[scan_idx]);
        scan->update_pose(delta_xi);
    }
}

void Solver::update_map() {
    // Update map voxels
    int voxels_size = voxel_keys_.size();
    for (int v = 0; v < voxels_size; ++v) {
        double new_intensity = J_M_B_(v) / H_MM_diag_(v);
        const auto& voxel_idx = voxel_keys_[v];
        vox_map_.at(voxel_idx) = new_intensity;
    }
}

std::vector<double> Solver::optimize(std::vector<Eigen::Vector3d>& rmse_history) {
    std::vector<double> cost_history;
    for (int iter = 0; iter < opts_.max_iterations; iter++) {
        std::cout << "Iteration " << iter + 1 << " / " << opts_.max_iterations << std::endl;
        double downsample_factor = (iter < opts_.num_coarse_iterations) ? opts_.coarse_downsample : opts_.refine_downsample;

        // Construct problem
        construct_problem(scan_manager_, downsample_factor);

        // Solve problem, skip updating if solve failed
        bool success = solve();
        if (!success) continue;

        // Save cost
        cost_history.push_back(cost_);

        // Update poses
        update_poses(scan_manager_);

        // Update map
        update_map();

        std::cout << "Cost: " << cost_ << std::endl;
        std::cout << "Pose RMSE (x, y, yaw): " << scan_manager_.compute_pose_rmse().transpose() << std::endl;
        rmse_history.push_back(scan_manager_.compute_pose_rmse());
        if (iter != 0 && (del_x_.norm() < opts_.convergence_tol || std::abs(prev_cost_ - cost_) < opts_.convergence_tol)) {
            std::cout << "Converged!" << std::endl;
            break;
        }
        prev_cost_ = cost_;
    }
    return cost_history;
}


} // namespace ba