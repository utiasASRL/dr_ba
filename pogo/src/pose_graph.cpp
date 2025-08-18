#include "pose_graph.h"
#include <iostream>
#include <cmath>
#include <stdexcept>




PoseGraph::PoseGraph(const PoseGraphOpts& opts)
    : opts_(opts)
{
    std::cout << "PoseGraph initialized with "
              << "\n\tloss_scale_loop_pos: " << opts.loss_scale_loop_pos << " m, "
              << "\n\tloss_scale_loop_rot: " << opts.loss_scale_loop_rot << " rad, "
              << "\n\tstd_odom_pos: " << opts.odom_pos_std << " m, "
              << "\n\tstd_odom_rot: " << opts.odom_rot_std << " rad, "
              << "\n\tstd_loop_pos: " << opts.loop_pos_std << " m, "
              << "\n\tstd_loop_rot: " << opts.loop_rot_std << " rad" << std::endl;

    // Initialize loss functions
    loss_function_loop_pos_ = new ceres::CauchyLoss(opts.loss_scale_loop_pos / opts.loop_pos_std);
    loss_function_loop_rot_ = new ceres::CauchyLoss(opts.loss_scale_loop_rot / opts.loop_rot_std);
    loss_function_odom_pos_ = new ceres::CauchyLoss(10.0);
    loss_function_odom_rot_ = new ceres::CauchyLoss(100.0);
            
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

    // Add noise
    node_poses_.back()->at(0) += ((double)rand() / RAND_MAX - 0.5) * 0.1;
    node_poses_.back()->at(1) += ((double)rand() / RAND_MAX - 0.5) * 0.1;
    node_poses_.back()->at(2) += ((double)rand() / RAND_MAX - 0.5) * 0.1*M_PI/180.0;

    // Add the new pose as a parameter block
    problem_.AddParameterBlock(node_poses_.back()->data(), 3);


    ceres::CostFunction* cost_function_pos = new RelativePosCostFunction(relative_pose, 1.0 / opts_.odom_pos_std);
    problem_.AddResidualBlock(cost_function_pos, loss_function_odom_pos_, node_poses_[node_indices_[t0]]->data(), node_poses_[node_indices_[t1]]->data());

    ceres::CostFunction* cost_function_rot = new RelativeRotCostFunction(relative_pose, 1.0 / opts_.odom_rot_std);
    problem_.AddResidualBlock(cost_function_rot, nullptr, node_poses_[node_indices_[t0]]->data(), node_poses_[node_indices_[t1]]->data());
}





void PoseGraph::addLoopClosureRotEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
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

    ceres::CostFunction* cost_function_rot = new RelativeRotCostFunction(relative_pose, 1.0 / opts_.loop_rot_std);
    problem_.AddResidualBlock(cost_function_rot, loss_function_loop_rot_, node_poses_[index0]->data(), node_poses_[index1]->data());

}

void PoseGraph::addLoopClosurePosEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
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

    ceres::CostFunction* cost_function_pos = new RelativePosCostFunction(relative_pose, 1.0 / opts_.loop_pos_std);
    problem_.AddResidualBlock(cost_function_pos, loss_function_loop_pos_, node_poses_[index0]->data(), node_poses_[index1]->data());
}

void PoseGraph::addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose)
{
    addLoopClosurePosEdge(t0, t1, relative_pose);
    addLoopClosureRotEdge(t0, t1, relative_pose);
}


void PoseGraph::optimize()
{
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1000;
    options.num_threads = 16;
    options.function_tolerance = 1e-8;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem_, &summary);
    std::cout << summary.FullReport() << std::endl;
}


void PoseGraph::printPoses() const
{
    for(size_t i = 0; i < node_times_.size(); ++i)
    {
        const auto& pose = *(node_poses_[i]);
        std::cout << node_times_[i] << " -> ("
                    << pose[0] << ", " << pose[1] << ", " << pose[2] << " rad)" << std::endl;
    }
}

void PoseGraph::printLastPose() const
{
    if(node_poses_.empty())
    {
        std::cout << "No poses available." << std::endl;
        return;
    }
    const auto& pose = *(node_poses_.back());
    std::cout << "Last pose: "
                << pose[0] << ", " << pose[1] << ", " << pose[2] << " rad" << std::endl;
}


void PoseGraph::writeToFile(const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing.");
    }

    for (size_t i = 0; i < node_times_.size(); ++i)
    {
        const auto& pose = *(node_poses_[i]);
        file << node_times_[i] << " "
             << pose[0] << " " << pose[1] << " " << pose[2] << "\n";
    }

    file.close();
}














RelativeRotCostFunction::RelativeRotCostFunction(const std::array<double, 3>& relative_pose, double weight)
{
    inv_meas_ = xyThetaToMat(relative_pose).inverse();
    weight_ = weight;
}

bool RelativeRotCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
    const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

    Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

    Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
    Eigen::Matrix3d delta = inv_meas_ * relative_pose;
    residuals[0] = std::atan2(delta(1, 0), delta(0, 0)) * weight_;

    if(jacobians)
    {
        // Compute the Jacobian
        if (jacobians[0] != nullptr) {
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

            jacobian1.setZero();
            jacobian1(0,2) = -1;
            jacobian1 *= weight_;
        }
        if (jacobians[1] != nullptr) {
            // Jacobian w.r.t. pose2
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

            jacobian2.setZero();
            jacobian2(0,2) = 1;
            jacobian2 *= weight_;
        }
    }


    return true;
}



RelativePosCostFunction::RelativePosCostFunction(const std::array<double, 3>& relative_pose, double weight)
    : weight_(weight)
{
    meas_[0] = relative_pose[0];
    meas_[1] = relative_pose[1];

}

bool RelativePosCostFunction::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
    const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

    Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
    Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

    Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
    Eigen::Vector2d delta = relative_pose.block<2,1>(0,2) - meas_;
    residuals[0] = weight_ * delta(0);
    residuals[1] = weight_ * delta(1);

    if(jacobians)
    {
        // Compute the Jacobian
        if (jacobians[0] != nullptr) {
            double s1 = std::sin(pose1[2]);
            double c1 = std::cos(pose1[2]);
            double dx = pose2[0] - pose1[0];
            double dy = pose2[1] - pose1[1];

            Eigen::Vector2d temp;
            temp[0] = -s1 * dx + c1 * dy;
            temp[1] = -c1 * dx - s1 * dy;

            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

            jacobian1.setZero();
            jacobian1.block<2, 2>(0, 0) = -inv_mat1.block<2, 2>(0, 0);
            jacobian1.block<2, 1>(0, 2) = temp;
            jacobian1 *= weight_;
        }
        if (jacobians[1] != nullptr) {
            // Jacobian w.r.t. pose2
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

            jacobian2.setZero();
            jacobian2.block<2, 2>(0, 0) = inv_mat1.block<2, 2>(0, 0);
            jacobian2 *= weight_;
        }
    }


    return true;
}