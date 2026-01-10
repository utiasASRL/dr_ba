// loader.hpp
#pragma once

#include <string>
#include <memory>
#include <ba/scans/scan.hpp>

namespace ba {

class ScanManager {
public:
    ScanManager() = default;
    void add_scan(std::shared_ptr<Scan> scan) {
        scans_.emplace(scan->id(), scan);
    }

    std::shared_ptr<Scan> get_scan(int scan_id) {
        return scans_.at(scan_id);
    }

    int num_scans() const {
        return static_cast<int>(scans_.size());
    }

    virtual Eigen::Matrix<double, 3, 1> compute_pose_rmse() const {
        if (scans_.empty()) {
            return Eigen::Matrix<double, 3, 1>::Zero();
        }
        Eigen::Matrix<double, 3, 1> rmse = Eigen::Matrix<double, 3, 1>::Zero();
        for (const auto& kv : scans_) {
            Eigen::Matrix<double, 6, 1> err = kv.second->pose_error();
            rmse(0) += err(0) * err(0); // x
            rmse(1) += err(1) * err(1); // y
            rmse(2) += err(5) * err(5); // yaw
        }
        rmse /= static_cast<double>(scans_.size());
        rmse = rmse.cwiseSqrt();
        return rmse;
    }

private:
    ankerl::unordered_dense::map<int, std::shared_ptr<Scan>> scans_;
};


} // namespace ba