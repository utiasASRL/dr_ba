// loader.hpp
#pragma once

#include <string>
#include <memory>
#include <ba/scans/scan.hpp>
#include <ankerl/unordered_dense.h>
#include <iostream>

namespace ba {

class ScanManager {
public:
    ScanManager() = default;
    void add_scan(std::shared_ptr<Scan> scan) {
        scans_.emplace(scan->id(), scan);
        scan_id_list_.push_back(scan->id());
    }

    std::shared_ptr<Scan> get_scan(int scan_id) {
        return scans_.at(scan_id);
    }

    std::shared_ptr<const Scan> get_scan(int scan_id) const {
        return scans_.at(scan_id);
    }

    int num_scans() const {
        return static_cast<int>(scans_.size());
    }

    int num_active_scans() const {
        int count = 0;
        for (const auto& kv : scans_) {
            if (!kv.second->is_fixed()) {
                count++;
            }
        }
        return count;
    }

    std::vector<int> get_all_scan_ids() const {
        return scan_id_list_;
    }

    // Compute RMSE of all scan poses compared to groundtruth (SE2: x, y, yaw)
    Eigen::Matrix<double, 3, 1> compute_pose_rmse() const {
        if (scans_.empty()) {
            return Eigen::Matrix<double, 3, 1>::Zero();
        }
        Eigen::Matrix<double, 3, 1> rmse = Eigen::Matrix<double, 3, 1>::Zero();
        for (const auto& kv : scans_) {
            if (kv.second->is_fixed()) {
                continue;
            }
            Eigen::Matrix<double, 6, 1> err = kv.second->pose_error();
            rmse(0) += err(0) * err(0); // x
            rmse(1) += err(1) * err(1); // y
            rmse(2) += err(5) * err(5); // yaw
        }
        rmse /= static_cast<double>(num_active_scans());
        rmse = rmse.cwiseSqrt();
        rmse(2) *= (180.0 / M_PI); // convert yaw to degrees
        return rmse;
    }

    double compute_ate() const {
        if (scans_.empty()) {
            return 0.0;
        }
        // Load in array of xy positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_positions(2, num_scans());
        Eigen::Matrix<double, 2, Eigen::Dynamic> gt_positions(2, num_scans());
        int idx = 0;
        for (const auto& kv : scans_) {
            lgmath::se2::Transformation T_est = kv.second->pose2d();
            lgmath::se2::Transformation T_gt = kv.second->gt_pose2d();
            est_positions(0, idx) = T_est.r_ab_inb()(0);
            est_positions(1, idx) = T_est.r_ab_inb()(1);
            gt_positions(0, idx) = T_gt.r_ab_inb()(0);
            gt_positions(1, idx) = T_gt.r_ab_inb()(1);
            idx++;
        }

        // Get centroids
        Eigen::Vector2d est_centroid = est_positions.rowwise().mean();
        Eigen::Vector2d gt_centroid = gt_positions.rowwise().mean();

        // Center the positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_centered = est_positions.colwise() - est_centroid;
        Eigen::Matrix<double, 2, Eigen::Dynamic> gt_centered = gt_positions.colwise() - gt_centroid;

        // Compute covariance matrix
        Eigen::Matrix2d H = gt_centered * est_centered.transpose();

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();

        // Correct for reflection
        if (R.determinant() < 0) {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1;
            R = V * svd.matrixU().transpose();
        }

        // Compute translation
        Eigen::Vector2d t = est_centroid - R * gt_centroid;

        // Apply transformation to estimated positions
        Eigen::Matrix<double, 2, Eigen::Dynamic> est_aligned = (R * gt_positions).colwise() + t;

        // Compute ATE
        double ate = 0.0;
        for (int i = 0; i < num_scans(); ++i) {
            ate += (est_aligned.col(i) - est_positions.col(i)).squaredNorm();
        }
        ate = std::sqrt(ate / static_cast<double>(num_scans()));

        return ate;
    }

    ScanManager deep_copy() const {
        ScanManager copy;
        for (const auto& scan_id : scan_id_list_) {
            const auto& scan = scans_.at(scan_id);
            auto scan_clone = scan->clone();
            copy.add_scan(scan_clone);
        }
        return copy;
    }

private:
    ankerl::unordered_dense::map<int, std::shared_ptr<Scan>> scans_;
    std::vector<int> scan_id_list_;
};


} // namespace ba