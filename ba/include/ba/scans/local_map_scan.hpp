// local_map_scan.hpp
#pragma once

#include <ba/scans/scan.hpp>
#include <Eigen/Dense>
#include <lgmath/se3/Transformation.hpp>
#include <opencv2/opencv.hpp>
#include <ba/utils/io_utils.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

namespace ba {

class LocalMapScan : public Scan {
public:
    using PixelCoords = std::pair<double, double>;
    using Index = std::pair<int32_t, int32_t>;

    LocalMapScan(
        int64_t timestamp,
        int scan_id,
        double local_map_res,
        const OptimizationOptions& opts,
        const lgmath::se3::Transformation& pose,
        const lgmath::se3::Transformation& gt_pose,
        const fs::path& img_path,
        std::optional<fs::path> cumul_img_path = std::nullopt)
        : Scan(timestamp, scan_id, opts.meas_std, pose, gt_pose),
        res_(local_map_res),
        range_factor_(opts.range_factor),
        cumul_thresh_(opts.cumul_thresh),
        zero_thresh_(opts.zero_thresh),
        img_width_(0),
        img_height_(0),
        img_path_(img_path),
        cumul_img_path_(std::move(cumul_img_path)) {
        
        // Check that image file exists
        if (!fs::exists(img_path_)) {
            throw std::invalid_argument(
                "Local map image file does not exist: " + img_path_.string());
        }

        // Load image to get dimensions only
        // TODO: This is very slow so I am hardcoding it for now...
        // load_data();
        // img_width_ = local_map_.cols();
        // img_height_ = local_map_.rows();
        // unload_data();
        img_width_= 2401; // hardcoded for boreas local maps
        img_height_ = 2401;
    }

    // Add initializer that takes in Eigen matrix directly (mostly for testing)
    LocalMapScan(
        int64_t timestamp,
        int scan_id,
        double local_map_res,
        const OptimizationOptions& opts,
        const lgmath::se3::Transformation& pose,
        const lgmath::se3::Transformation& gt_pose,
        const Eigen::MatrixXf& local_map)
        : Scan(timestamp, scan_id, opts.meas_std, pose, gt_pose),
        res_(local_map_res),
        range_factor_(opts.range_factor),
        cumul_thresh_(opts.cumul_thresh),
        zero_thresh_(opts.zero_thresh),
        img_width_(local_map.cols()),
        img_height_(local_map.rows()),
        local_map_(local_map) {}

    std::optional<Measurement> interpolate(double x, double y) const override;
    bool check_coverage_at_point(double x, double y) const override;
    double res() const { return res_; }
    int img_width() const { return img_width_; }
    int img_height() const { return img_height_; }

    // Convert world frame coordinates to local image coordinates with origin at center of image
    PixelCoords coord_to_image_coord(double x, double y) const;

    // Convert world frame coordinates to root pixel index
    Index get_root_pixel_coords(double x, double y) const;

    // Convert world frame coordinates to pixel coordinates
    // Note that image coordinates have x up and y right, with (0,0) in the middle of the image
    // Pixel coordinates have u right and v down, with (0,0) at the top-left corner of the image
    PixelCoords coord_to_pixel(double x, double y, Eigen::Matrix<double, 2, 3> *jacobian = nullptr) const;

    // Clone method for deep copying
    std::shared_ptr<Scan> clone() const override {
        return std::make_shared<LocalMapScan>(*this);
    }

    // Load/unload heavy data
    void load_data() override;
    void unload_data() override {
        local_map_.resize(0,0);
        if (cumul_img_path_.has_value()) {
            cumul_img_.resize(0,0);
        }
    }
    bool data_loaded() const override {
        return (local_map_.size() != 0);
    }

private:
    double res_;
    double range_factor_;
    double cumul_thresh_;
    double zero_thresh_;
    int img_width_;
    int img_height_;
    Eigen::MatrixXf local_map_;
    Eigen::MatrixXf cumul_img_;
    fs::path img_path_;
    std::optional<fs::path> cumul_img_path_;
};


}