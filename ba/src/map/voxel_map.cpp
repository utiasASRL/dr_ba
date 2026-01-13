#include <ba/map/voxel_map.hpp>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace ba {

VoxelMap::VoxelMap(double res) : res_(res) {}

auto VoxelMap::index(double x, double y) const -> Index {
	const int32_t a = static_cast<int32_t>(std::floor(x / res_));
	const int32_t b = static_cast<int32_t>(std::floor(y / res_));
	return {a, b};
}

std::size_t VoxelMap::size() const { return voxels_.size(); }

void VoxelMap::add_single_voxel(int32_t a, int32_t b, double intensity) {
    if (voxels_.contains({a, b})) return;
	voxels_[{a, b}] = intensity;
}

void VoxelMap::add_single_voxel(double x, double y, double intensity) {
    const Index ab = index(x, y);
    if (voxels_.contains(ab)) return;
    voxels_[ab] = intensity;
}

void VoxelMap::zero_out() {
	for (auto& kv : voxels_) kv.second = 0.0;
}

void VoxelMap::randomize(uint32_t seed) {
	std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (auto& kv : voxels_) kv.second = dist(rng);
}

void VoxelMap::init_map(const lgmath::se3::Transformation& pose, double max_dist) {
    const Eigen::Matrix<double, 4, 4> pose_mat = pose.matrix();
	const double x_center = pose_mat(0, 3);
	const double y_center = pose_mat(1, 3);
	const auto center = index(x_center, y_center);
	const int32_t n = static_cast<int32_t>(std::ceil(max_dist / res_));
	for (int32_t da = -n; da <= n; ++da) {
		for (int32_t db = -n; db <= n; ++db) {
            if (std::sqrt(std::pow(da * res_, 2) + std::pow(db * res_, 2)) > max_dist) {
                continue;
            }
			const Index idx{center.first + da, center.second + db};
			if (!contains(idx)) voxels_.emplace(idx, 0.0);
		}
	}
}

void VoxelMap::init_map(const lgmath::se2::Transformation& pose, double max_dist) {
    init_map(pose.toSE3(), max_dist);
}

std::vector<VoxelMap::Index> VoxelMap::get_sorted_keys_downsampled(double downsample_factor) const {
    if (voxels_.empty()) return {};
    if (downsample_factor > 1.0 || downsample_factor <= 0.0) {
        throw std::invalid_argument("Downsample factor must be in (0, 1]");
    }

    std::vector<Index> keys;
    keys.reserve(voxels_.size());

    // Collect and sort all keys
    for (const auto& kv : voxels_) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    // No downsample
    if (downsample_factor >= 1.0) return keys;

    std::vector<Index> downsampled_keys;
    downsampled_keys.reserve(static_cast<std::size_t>(std::ceil(keys.size() * downsample_factor)));

    const std::size_t step = static_cast<std::size_t>(
        std::max<double>(1.0, std::round(1.0 / downsample_factor)));
    for (std::size_t i = 0; i < keys.size(); i += step) {
        downsampled_keys.push_back(keys[i]);
    }
    return downsampled_keys;
}

void VoxelMap::visualize(double downsample_factor) const {
    if (voxels_.empty()) return;

    // 1. Find bounds of the indices
    int32_t min_x = std::numeric_limits<int32_t>::max();
    int32_t max_x = std::numeric_limits<int32_t>::min();
    int32_t min_y = std::numeric_limits<int32_t>::max();
    int32_t max_y = std::numeric_limits<int32_t>::min();

    for (const auto &[idx, val] : voxels_) {
        min_x = std::min(min_x, idx.first);
        max_x = std::max(max_x, idx.first);
        min_y = std::min(min_y, idx.second);
        max_y = std::max(max_y, idx.second);
    }

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    cv::Mat img(height, width, CV_64F, cv::Scalar(0)); // use double for intensity

    // 2. Fill image
    for (const auto &[idx, val] : voxels_) {
        int x = idx.first - min_x;
        int y = idx.second - min_y;
        img.at<double>(y, x) = val; // row = y, col = x
    }

    // 3. Normalize to 0-255 and convert to 8-bit for display
    cv::Mat img8;
    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    img.convertTo(img8, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    // Show image

    cv::namedWindow("Scan", cv::WINDOW_NORMAL);
    cv::resizeWindow("Scan", 480, 480);
    cv::imshow("Scan", img);
    cv::waitKey(0);      // waits for key press
    cv::destroyAllWindows();
}

void VoxelMap::save_to_file(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    // Write resolution
    ofs.write(reinterpret_cast<const char*>(&res_), sizeof(res_));

    // Write voxel data
    for (const auto& kv : voxels_) {
        int32_t x = kv.first.first;
        int32_t y = kv.first.second;
        double intensity = kv.second;
        // Only save non-zero voxels
        if (intensity < 0.01) continue;
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(x));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(y));
        ofs.write(reinterpret_cast<const char*>(&intensity), sizeof(intensity));
    }

    ofs.close();
}



bool VoxelMap::contains(Index idx) const { return voxels_.find(idx) != voxels_.end(); }

double& VoxelMap::at(Index idx) { return voxels_.at(idx); }

const double& VoxelMap::at(Index idx) const { return voxels_.at(idx); }

}