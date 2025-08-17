#include "pose_graph.h"
#include <iostream>
#include <cmath>
#include <stdexcept>




PoseGraph::PoseGraph(const double loss_scale_loop_pos, const double loss_scale_loop_rot)
{
    std::cout << "PoseGraph initialized with loss scales: "
              << "Position: " << loss_scale_loop_pos 
              << ", Rotation: " << loss_scale_loop_rot << std::endl;

    // Initialize loss functions
    loss_function_loop_pos_ = new ceres::CauchyLoss(loss_scale_loop_pos);
    loss_function_loop_rot_ = new ceres::CauchyLoss(loss_scale_loop_rot * M_PI / 180.0); 
            
}



void PoseGraph::addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    if(node_indices_.size() == 0)
    {
        node_times_.push_back(t0);
        node_poses_.push_back(std::make_shared<std::array<double, 3>>(std::array<double, 3>{0.0, 0.0, 0.0}));
        node_indices_[t0] = 0;

        problem_.AddParameterBlock(node_poses_[0]->data(), 3);
        problem_.SetParameterBlockConstant(node_poses_[0]->data());

    }

    if(t0 != node_times_.back())
    {
        throw std::runtime_error("Odometry edge t0 does not match the last node time.");
    }

    if(t1 <= t0)
    {
        throw std::runtime_error("Odometry edge t1 must be greater than t0.");
    }

    // Add the new node
    node_indices_[t1] = node_poses_.size();
    node_times_.push_back(t1);
    std::array<double, 3> new_pose = combinePoses(*(node_poses_.back()), relative_pose);
    node_poses_.push_back(std::make_shared<std::array<double, 3>>(new_pose));

    // Add the new pose as a parameter block
    problem_.AddParameterBlock(node_poses_.back()->data(), 3);


    // Create residual for the odometry edge
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<OdometryResidualFunctor, 3, 3, 3>(
        new OdometryResidualFunctor(relative_pose)
    );
    problem_.AddResidualBlock(cost_function, nullptr, node_poses_[node_indices_[t0]]->data(), node_poses_[node_indices_[t1]]->data());

}



void PoseGraph::addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    if(node_indices_.find(t0) == node_indices_.end() || node_indices_.find(t1) == node_indices_.end())
    {
        throw std::runtime_error("Loop closure edge contains unknown timestamps.");
    }

    size_t index0 = node_indices_[t0];
    size_t index1 = node_indices_[t1];

    if(index0 >= node_poses_.size() || index1 >= node_poses_.size())
    {
        throw std::runtime_error("Loop closure edge indices out of bounds.");
    }

    // Add the loop closure edge to the problem
    ceres::CostFunction* cost_function_pos = new ceres::AutoDiffCostFunction<LoopClosurePosResidualFunctor, 2, 3, 3>(
        new LoopClosurePosResidualFunctor(relative_pose)
    );
    problem_.AddResidualBlock(cost_function_pos, loss_function_loop_pos_, node_poses_[index0]->data(), node_poses_[index1]->data());
    ceres::CostFunction* cost_function_rot = new ceres::AutoDiffCostFunction<LoopClosureRotResidualFunctor, 1, 3, 3>(
        new LoopClosureRotResidualFunctor(relative_pose)
    );
    problem_.AddResidualBlock(cost_function_rot, loss_function_loop_rot_, node_poses_[index0]->data(), node_poses_[index1]->data());
}