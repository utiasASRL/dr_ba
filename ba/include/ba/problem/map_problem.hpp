// ba_problem.hpp
#pragma once

#include <ba/problem/problem.hpp>

namespace ba {

class MapProblem : public Problem {
public:
    MapProblem(Options& opts)
        : Problem(opts, opts.map_seq) {
            // Overwrite options for mapping
            opts_.max_dist = opts.map_max_dist;
            opts_.dist_field_preproc = opts.map_dist_field_preproc;
            opts_.gauss_blur_sigma = opts.map_gauss_blur_sigma;
            opts_.adaptive_blur = opts.map_adaptive_blur;
            opts_.min_int_val_tol = opts.map_min_int_val_tol;
            opts_.min_percent_nonzero = opts.map_min_percent_nonzero;
        }

    void get_scan_indeces() override;
    void init_scans_and_map() override;
    void init_scans_and_map_from_estimates();
    void init_scans_and_map_from_data();
    void finalize() override;

private:


};

}   // namespace ba