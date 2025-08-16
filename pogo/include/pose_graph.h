#pragma once

#include <vector>
#include <map>
#include <array>
#include <stdint.h>



class PoseGraph {
    public:
        PoseGraph(const double loss_scale_loop_pos, const double loss_scale_loop_rot);

        void addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);

        void addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);

    private:
        std::map<int64_t, size_t> node_indices_;
        std::vector<int64_t> node_times_;
        std::vector<std::array<double, 3>> node_poses_;
        // Other private members

        double loss_scale_loop_pos_;
        double loss_scale_loop_rot_;
};
