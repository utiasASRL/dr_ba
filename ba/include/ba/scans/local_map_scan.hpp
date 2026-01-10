// local_map_scan.hpp
#pragma once

#include <ba/scans/scan.hpp>
#include <Eigen/Dense>
#include <lgmath/se3/Transformation.hpp>

namespace ba {

class LocalMapScan : public Scan {
public:
    using PixelCoords = std::pair<double, double>;
    using Index = std::pair<int32_t, int32_t>;

    LocalMapScan(int scan_id, const lgmath::se3::Transformation &pose, double res, const Eigen::MatrixXd &local_map)
        : Scan(scan_id, pose) {
        res_ = res;
        local_map_ = local_map;
        img_width_ = local_map.cols();
        img_height_ = local_map.rows();
    }
    LocalMapScan(int scan_id, const lgmath::se3::Transformation &pose, const lgmath::se3::Transformation &gt_pose, 
                double res, const Eigen::MatrixXd &local_map)
        : Scan(scan_id, pose, gt_pose) {
        res_ = res;
        local_map_ = local_map;
        img_width_ = local_map.cols();
        img_height_ = local_map.rows();
    }

    std::optional<double> interpolate(double x, double y, Eigen::Matrix<double, 1, 3> *jacobian = nullptr) const override;
    bool check_coverage_at_point(double x, double y) const override;
    double res() const { return res_; }
    int img_width() const { return img_width_; }
    int img_height() const { return img_height_; }

    // Convert world frame coordinates to root pixel index
    Index get_root_pixel_coords(double x, double y) const;

    // Convert world frame coordinates to pixel coordinates
    // Note that image coordinates have x up and y right, with (0,0) in the middle of the image
    // Pixel coordinates have u right and v down, with (0,0) at the top-left corner of the image
    PixelCoords coord_to_pixel(double x, double y, Eigen::Matrix<double, 2, 3> *jacobian = nullptr) const;

private:
    double res_;
    int img_width_;
    int img_height_;
    Eigen::MatrixXd local_map_;
};


}