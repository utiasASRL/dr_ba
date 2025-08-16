#include "pose_graph.h"
#include <iostream>
#include <cmath>




PoseGraph::PoseGraph(const double loss_scale_loop_pos, const double loss_scale_loop_rot)
    : loss_scale_loop_pos_(loss_scale_loop_pos)
    , loss_scale_loop_rot_(loss_scale_loop_rot * M_PI / 180.0)
{
    std::cout << "PoseGraph initialized with loss scales: "
              << "Position: " << loss_scale_loop_pos_ 
              << ", Rotation: " << loss_scale_loop_rot_ << std::endl;
}



void PoseGraph::addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    std::cout << "Adding odometry edge from " << t0 << " to " << t1 
              << " with relative pose: [" << relative_pose[0] << ", "
              << relative_pose[1] << ", " << relative_pose[2] << "]" << std::endl;
}



void PoseGraph::addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    std::cout << "Adding loop closure edge from " << t0 << " to " << t1 
              << " with relative pose: [" << relative_pose[0] << ", "
              << relative_pose[1] << ", " << relative_pose[2] << "]" << std::endl;
}