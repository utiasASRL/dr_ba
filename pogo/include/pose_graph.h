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

        void optimize()
        {
            ceres::Solver::Options options;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 1000;
            options.num_threads = 16;
            options.function_tolerance = 1e-16;
            options.gradient_tolerance = 1e-16;
            options.parameter_tolerance = 1e-16;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem_, &summary);
            std::cout << summary.FullReport() << std::endl;
        }


        void printPoses() const
        {
            for(size_t i = 0; i < node_times_.size(); ++i)
            {
                const auto& pose = *(node_poses_[i]);
                std::cout << node_times_[i] << " -> ("
                          << pose[0] << ", " << pose[1] << ", " << pose[2] << " rad)" << std::endl;
            }
        }

        void printLastPose() const
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



template<typename T>
inline Eigen::Matrix<T, 3, 3> xyThetaToMatT(const T* const pose)
{
    Eigen::Matrix<T, 3, 3> mat = Eigen::Matrix<T, 3, 3>::Identity();
    mat(0, 0) = ceres::cos(pose[2]);
    mat(0, 1) = -ceres::sin(pose[2]);
    mat(1, 0) = ceres::sin(pose[2]);
    mat(1, 1) = ceres::cos(pose[2]);
    mat(0, 2) = pose[0];
    mat(1, 2) = pose[1];
    return mat;
}


template<typename T>
inline std::array<T, 3> matToXYThetaT(const Eigen::Matrix<T, 3, 3>& mat)
{
    std::array<T, 3> pose;
    pose[0] = mat(0, 2);
    pose[1] = mat(1, 2);
    pose[2] = ceres::atan2(mat(1, 0), mat(0, 0));
    return pose;
}


class OdometryResidualFunctor
{
    public:
        OdometryResidualFunctor(const std::array<double, 3>& relative_pose)
        {
            inv_relative_pose_ = matToXYTheta(xyThetaToMat(relative_pose).inverse());
        }


        template <typename T>
        bool operator()(const T* const pose1, const T* const pose2, T* residuals) const
        {
            Eigen::Matrix<T, 3, 3> mat1 = xyThetaToMatT<T>(pose1);
            Eigen::Matrix<T, 3, 3> mat2 = xyThetaToMatT<T>(pose2);
            Eigen::Matrix<T, 3, 3> relative_mat = mat1.inverse() * mat2;

            std::array<T, 3> inv_meas;
            inv_meas[0] = T(inv_relative_pose_[0]);
            inv_meas[1] = T(inv_relative_pose_[1]);
            inv_meas[2] = T(inv_relative_pose_[2]);

            Eigen::Matrix<T, 3, 3> inv_meas_mat = xyThetaToMatT<T>(inv_meas.data());

            Eigen::Matrix<T, 3, 3> delta_mat = relative_mat * inv_meas_mat;
            std::array<T, 3> res = matToXYThetaT(delta_mat);

            // Compute the residuals
            residuals[0] =  res[0];
            residuals[1] =  res[1];
            residuals[2] =  res[2];// * T(10.0); // Scale the rotation residual
            return true;
        }

    private:
        std::array<double, 3> inv_relative_pose_;
};


class LoopClosureRotResidualFunctor
{
    public:
        LoopClosureRotResidualFunctor(const std::array<double, 3>& relative_pose)
            : odometry_functor_(relative_pose)
        {
        }

        template <typename T>
        bool operator()(const T* const pose1, const T* const pose2, T* residuals) const
        {
            std::array<T, 3> temp_res;
            if (!odometry_functor_(pose1, pose2, temp_res.data())) {
                return false;
            }
            residuals[0] = temp_res[2]; // Only the rotation component
            return true;
        }

    private:
        OdometryResidualFunctor odometry_functor_;

};


class LoopClosurePosResidualFunctor
{
    public:
        LoopClosurePosResidualFunctor(const std::array<double, 3>& relative_pose)
            : odometry_functor_(relative_pose)
        {
        }

        template <typename T>
        bool operator()(const T* const pose1, const T* const pose2, T* residuals) const
        {
            std::array<T, 3> temp_res;
            if (!odometry_functor_(pose1, pose2, temp_res.data())) {
                return false;
            }
            residuals[0] = temp_res[0]; // X position component
            residuals[1] = temp_res[1]; // Y position component
            return true;
        }

    private:
        OdometryResidualFunctor odometry_functor_;
};



class OdometryCostFunction : public ceres::SizedCostFunction<3, 3, 3>
{
    public:
        OdometryCostFunction(const std::array<double, 3>& relative_pose)
        {
            inv_meas_ = xyThetaToMat(relative_pose).inverse();
        }

        virtual ~OdometryCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
            const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

            Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
            Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

            Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
            Eigen::Matrix3d delta = inv_meas_ * relative_pose;
            residuals[0] = delta(0, 2);
            residuals[1] = delta(1, 2);
            residuals[2] = std::atan2(delta(1, 0), delta(0, 0));

            if(jacobians)
            {
                Eigen::Matrix2d temp_rot = inv_meas_.block<2, 2>(0, 0) * inv_mat1.block<2, 2>(0, 0);

                // Compute the Jacobian
                if (jacobians[0] != nullptr) {
                    double s1 = std::sin(pose1[2]);
                    double c1 = std::cos(pose1[2]);
                    double dx = pose2[0] - pose1[0];
                    double dy = pose2[1] - pose1[1];

                    Eigen::Vector2d temp;
                    temp[0] = -s1 * dx + c1 * dy;
                    temp[1] = -c1 * dx - s1 * dy;

                    temp = inv_meas_.block<2, 2>(0, 0) * temp;

                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

                    jacobian1.setZero();
                    jacobian1.block<2, 2>(0, 0) = -temp_rot;
                    jacobian1.block<2, 1>(0, 2) = temp;
                    jacobian1(2,2) = -1;
                }
                if (jacobians[1] != nullptr) {
                    // Jacobian w.r.t. pose2
                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

                    jacobian2.setZero();
                    jacobian2.block<2, 2>(0, 0) = temp_rot;
                    jacobian2(2,2) = 1;
                }
            }


            return true;
        }

    private:
        Eigen::Matrix3d inv_meas_;
};


class LoopClosureRotCostFunction: public ceres::SizedCostFunction<1, 3, 3>
{
    public:
        LoopClosureRotCostFunction(const std::array<double, 3>& relative_pose)
        {
            inv_meas_ = xyThetaToMat(relative_pose).inverse();
        }

        virtual ~LoopClosureRotCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
            const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

            Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
            Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

            Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
            Eigen::Matrix3d delta = inv_meas_ * relative_pose;
            residuals[0] = std::atan2(delta(1, 0), delta(0, 0));

            if(jacobians)
            {
                // Compute the Jacobian
                if (jacobians[0] != nullptr) {
                    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

                    jacobian1.setZero();
                    jacobian1(0,2) = -1;
                }
                if (jacobians[1] != nullptr) {
                    // Jacobian w.r.t. pose2
                    Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

                    jacobian2.setZero();
                    jacobian2(0,2) = 1;
                }
            }


            return true;
        }
    private:
        Eigen::Matrix3d inv_meas_;
};



class LoopClosurePosCostFunction : public ceres::SizedCostFunction<2, 3, 3>
{
    public:
        LoopClosurePosCostFunction(const std::array<double, 3>& relative_pose)
        {
            inv_meas_ = xyThetaToMat(relative_pose).inverse();
        }

        virtual ~LoopClosurePosCostFunction() {}

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            const std::array<double, 3> pose1 = {parameters[0][0], parameters[0][1], parameters[0][2]};
            const std::array<double, 3> pose2 = {parameters[1][0], parameters[1][1], parameters[1][2]};

            Eigen::Matrix3d inv_mat1 = xyThetaToMat(pose1).inverse();
            Eigen::Matrix3d mat2 = xyThetaToMat(pose2);

            Eigen::Matrix3d relative_pose = inv_mat1 * mat2;
            Eigen::Matrix3d delta = inv_meas_ * relative_pose;
            residuals[0] = delta(0, 2);
            residuals[1] = delta(1, 2);

            if(jacobians)
            {
                Eigen::Matrix2d temp_rot = inv_meas_.block<2, 2>(0, 0) * inv_mat1.block<2, 2>(0, 0);

                // Compute the Jacobian
                if (jacobians[0] != nullptr) {
                    double s1 = std::sin(pose1[2]);
                    double c1 = std::cos(pose1[2]);
                    double dx = pose2[0] - pose1[0];
                    double dy = pose2[1] - pose1[1];

                    Eigen::Vector2d temp;
                    temp[0] = -s1 * dx + c1 * dy;
                    temp[1] = -c1 * dx - s1 * dy;

                    temp = inv_meas_.block<2, 2>(0, 0) * temp;

                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian1(jacobians[0]);

                    jacobian1.setZero();
                    jacobian1.block<2, 2>(0, 0) = -temp_rot;
                    jacobian1.block<2, 1>(0, 2) = temp;
                }
                if (jacobians[1] != nullptr) {
                    // Jacobian w.r.t. pose2
                    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian2(jacobians[1]);

                    jacobian2.setZero();
                    jacobian2.block<2, 2>(0, 0) = temp_rot;
                }
            }


            return true;
        }

    private:
        Eigen::Matrix3d inv_meas_;
};
