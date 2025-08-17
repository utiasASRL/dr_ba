#pragma once

#include <vector>
#include <map>
#include <array>
#include <stdint.h>
#include <ceres/ceres.h>
#include <utils.h>



class PoseGraph {
    public:
        PoseGraph(const double loss_scale_loop_pos, const double loss_scale_loop_rot);

        void addOdometryEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);

        void addLoopClosurePosEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);
        void addLoopClosureRotEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);
        void addLoopClosureEdge(const int64_t t0, const int64_t t1, std::array<double, 3> relative_pose);

        void optimize();


        void printPoses() const;

        void printLastPose() const;

        void writeToFile(const std::string& filename) const;

    private:
        // Storing state variables and timestamps
        std::map<int64_t, size_t> node_indices_;
        std::vector<int64_t> node_times_;
        std::vector<std::shared_ptr<std::array<double, 3>>> node_poses_;


        // Loss functions
        ceres::LossFunction* loss_function_loop_pos_ = nullptr;
        ceres::LossFunction* loss_function_loop_rot_ = nullptr;


        ceres::Problem problem_;

};



class OdometryCostFunction : public ceres::SizedCostFunction<3, 3, 3>
{
    public:
        OdometryCostFunction(const std::array<double, 3>& relative_pose);

        virtual ~OdometryCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    private:
        Eigen::Matrix3d inv_meas_;
};


class LoopClosureRotCostFunction: public ceres::SizedCostFunction<1, 3, 3>
{
    public:
        LoopClosureRotCostFunction(const std::array<double, 3>& relative_pose);

        virtual ~LoopClosureRotCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    private:
        Eigen::Matrix3d inv_meas_;
};



class LoopClosurePosCostFunction : public ceres::SizedCostFunction<2, 3, 3>
{
    public:
        LoopClosurePosCostFunction(const std::array<double, 3>& relative_pose);

        virtual ~LoopClosurePosCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    private:
        Eigen::Matrix3d inv_meas_;
};
