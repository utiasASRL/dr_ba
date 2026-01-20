// ba_problem.hpp
#pragma once

#include <ba/problem/problem.hpp>


namespace ba {

class BAProblem : public Problem {
public:
    BAProblem(Options& opts)
        : Problem(opts) {}

    void init_scans();
    void init_map();
    void init_scans_and_map() override {
        init_scans();
        init_map();
    }
    void finalize() override;

private:


};

}   // namespace ba