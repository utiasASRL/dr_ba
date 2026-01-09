#include "ba/utils/io_utils.hpp"
#include <fstream>
#include <iostream>

namespace ba {

Eigen::Matrix3d toRoll(const double &r) {
    Eigen::Matrix3d roll;
    roll << 1, 0, 0, 0, cos(r), sin(r), 0, -sin(r), cos(r);
    return roll;
}

Eigen::Matrix3d toPitch(const double &p) {
    Eigen::Matrix3d pitch;
    pitch << cos(p), 0, -sin(p), 0, 1, 0, sin(p), 0, cos(p);
    return pitch;
}

Eigen::Matrix3d toYaw(const double &y) {
    Eigen::Matrix3d yaw;
    yaw << cos(y), sin(y), 0, -sin(y), cos(y), 0, 0, 0, 1;
    return yaw;
}

Eigen::Matrix3d rpy2rot(const double &r, const double &p, const double &y) {
    return toRoll(r) * toPitch(p) * toYaw(y);
}

double roundToPi(double value) {
    return std::round(value / M_PI) * M_PI;
}

Eigen::Matrix4d load_T_radar_applanix(const std::filesystem::path &path) {
    std::ifstream ifs1(path / "calib" / "T_applanix_lidar.txt", std::ios::in);
    std::ifstream ifs2(path / "calib" / "T_radar_lidar.txt", std::ios::in);

    Eigen::Matrix4d T_applanix_lidar_mat;
    for (size_t row = 0; row < 4; row++)
        for (size_t col = 0; col < 4; col++) ifs1 >> T_applanix_lidar_mat(row, col);

    Eigen::Matrix4d T_radar_lidar_mat;
    for (size_t row = 0; row < 4; row++)
        for (size_t col = 0; col < 4; col++) ifs2 >> T_radar_lidar_mat(row, col);

    return Eigen::Matrix4d(T_radar_lidar_mat * T_applanix_lidar_mat.inverse());
}

void load_groundtruth_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times) {
    // Load transform
    Eigen::Matrix4d T_radar_applanix = load_T_radar_applanix(path);
    lgmath::se3::Transformation T_radar_applanix_tf = lgmath::se3::Transformation(T_radar_applanix);
    std::ifstream ifs(path / "applanix" / "gps_post_process.csv", std::ios::in);
    // Clear header line
    std::string line;
    std::getline(ifs, line);
    // Loop through all gt data
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> gt;
        for (std::string str; std::getline(ss, str, ',');)
        gt.push_back(std::stod(str));

        // Store gt pose
        Eigen::Matrix4d T_ab_mat = Eigen::Matrix4d::Identity();
        T_ab_mat.block<3, 3>(0, 0) = rpy2rot(roundToPi(gt[7]), roundToPi(gt[8]), gt[9]);
        T_ab_mat.block<3, 1>(0, 3) << gt[1], gt[2], 0.0;
        lgmath::se3::Transformation T_ab = lgmath::se3::Transformation(T_ab_mat);
        all_poses.push_back((T_radar_applanix_tf * T_ab.inverse()).inverse());
        all_times.push_back(gt[0]);
    }
}

void load_pogo_poses_and_times(const std::filesystem::path &path, std::vector<lgmath::se3::Transformation> &all_poses, std::vector<double> &all_times) {
    std::ifstream ifs(path / "pose_graph_traj.txt", std::ios::in);
    // Clear header line
    std::string line;
    std::getline(ifs, line);
    // Loop through all gt data
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> data;
        for (std::string str; std::getline(ss, str, ' ');)
        data.push_back(std::stod(str));

        // Store pogo pose
        Eigen::Matrix4d T_ab_mat = Eigen::Matrix4d::Identity();
        T_ab_mat.block<3, 1>(0, 3) << data[1], data[2], 0.0;
        T_ab_mat.block<2, 2>(0, 0) << cos(data[3]), -sin(data[3]),
                                   sin(data[3]),  cos(data[3]);
        lgmath::se3::Transformation T_ab = lgmath::se3::Transformation(T_ab_mat);
        all_poses.push_back(T_ab);
        all_times.push_back(data[0] / 1e6);  // convert to seconds
    }
}

lgmath::se3::Transformation get_interpolated_pose(
    const std::vector<lgmath::se3::Transformation> &all_poses,
    const std::vector<double> &all_times,
    double query_time) {
    const size_t N = all_times.size();
    // Handle exceptions
    if (N == 0) throw std::runtime_error("Empty pose/time vectors");
    if (query_time < all_times.front()) throw std::runtime_error("Query time before first timestamp");
    if (query_time > all_times.back()) throw std::runtime_error("Query time after last timestamp");

    // Sort for nearest idx
    auto it = std::lower_bound(all_times.begin(), all_times.end(), query_time);
    size_t idx = std::distance(all_times.begin(), it);

    // Handle idx exceptions right on boundary
    if (idx == 0) return all_poses.front();
    if (idx >= N) return all_poses.back();

    // We already handled boundaries, so idx >= 1 here
    size_t i = idx - 1;
    // Compute interpolation alpha
    double t0 = all_times[i];
    double t1 = all_times[i + 1];
    double alpha = (query_time - t0) / (t1 - t0);

    // Extract matrices
    Eigen::Matrix<double, 6, 1> xi1 = all_poses[i].vec();
    Eigen::Matrix<double, 6, 1> xi2 = all_poses[i+1].vec();
    Eigen::Matrix<double, 6, 1> xi_interp = (1.0 - alpha) * xi1 + alpha * xi2;
    return lgmath::se3::Transformation(xi_interp);
}



} // namespace ba