// ba_problem.hpp
#pragma once

#include <ba/problem/problem.hpp>

namespace ba {

class LocProblem : public Problem {
public:
    LocProblem(Options& opts)
        : Problem(opts) {}

    void init_scans_and_map() override;
    void finalize() override;

    void load_map_from_estimate();
    void load_scans();

    std::vector<lgmath::se3::Transformation> gt_map_poses() const { return gt_map_poses_; }
    std::vector<lgmath::se3::Transformation> gt_poses() const { return gt_poses_; }
    std::vector<lgmath::se3::Transformation> dro_poses() const { return dro_poses_; }

private:
    std::vector<lgmath::se3::Transformation> gt_map_poses_;
    std::vector<lgmath::se3::Transformation> gt_poses_;
    std::vector<lgmath::se3::Transformation> dro_poses_;
};

}   // namespace ba