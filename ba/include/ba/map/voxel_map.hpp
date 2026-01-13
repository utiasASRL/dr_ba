// voxel_map.hpp
#pragma once

#include <ankerl/unordered_dense.h>

#include <cstdint>
#include <utility>
#include <vector>
#include <lgmath/se2/Transformation.hpp>
#include <lgmath/se3/Transformation.hpp>

namespace ba {

class VoxelMap {
public:
	using Index = std::pair<int32_t, int32_t>;

	explicit VoxelMap(double res = 1.0);

	// Convert world coordinates (meters) to voxel indices
	Index index(double x, double y) const;

	// Number of stored voxels
	std::size_t size() const;

	// Create or set a voxel intensity at integer coordinates
	void add_single_voxel(int32_t a, int32_t b, double intensity);

	// Create or set a voxel intensity at cartesian coordinates in meters
	void add_single_voxel(double x, double y, double intensity);

	// Reset all intensities to 0.0
	void zero_out();

	// Fill all existing voxels with random values in [min_val, max_val]
	void randomize(uint32_t seed = 0) /* deterministic if seed!=0 */;

	// Initialize empty voxels in a square window around pose within max_dist
	// If SE3 pose provided, only the SE2 components are used
	void init_map(const lgmath::se2::Transformation& pose, double max_dist);
	void init_map(const lgmath::se3::Transformation& pose, double max_dist);

	// Return sorted voxel keys downsampled by factor in [0,1]; 1.0 means no downsample
	std::vector<Index> get_sorted_keys_downsampled(double downsample_factor = 1.0) const;

	// Direct access helpers
	bool contains(Index idx) const;
	double& at(Index idx);
	const double& at(Index idx) const;

	// Resolution access
	double res() const { return res_; }

	// Visualize as pixel image (for debugging)
	void visualize(double downsample_factor = 1.0) const;

	// Save map as a binary file
	void save_to_file(const std::string& filepath) const;

private:
	double res_;
	ankerl::unordered_dense::map<Index, double> voxels_;
};

} // namespace ba
