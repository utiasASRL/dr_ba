#include <ba/scans/local_map_scan.hpp>
#include <lgmath/se2/Operations.hpp>

namespace ba {

std::optional<double> LocalMapScan::interpolate(double x, double y, Eigen::Matrix<double, 1, 3> *jacobian) const {
    // Check coverage first
    if (!check_coverage_at_point(x, y)) {
        return std::nullopt;
    }
    
    // Get pixel coordinates and Jacobian
    Eigen::Matrix<double, 2, 3> d_g_d_T;
    PixelCoords p = coord_to_pixel(x, y, jacobian ? &d_g_d_T : nullptr);
    double u = p.first;
    double v = p.second;

    // Get root pixel coords
    Index root_px = get_root_pixel_coords(x, y);
    int a = root_px.first;
    int b = root_px.second;

    // Get intensities at four corners
    double int_ab = local_map_(b, a);
    double int_a1b = local_map_(b, a + 1);
    double int_ab1 = local_map_(b + 1, a);
    double int_a1b1 = local_map_(b + 1, a + 1);

    // Get weights
    double u_tilde = u - static_cast<double>(a);
    double v_tilde = v - static_cast<double>(b);
    double w0 = (1.0 - u_tilde) * (1.0 - v_tilde);
    double w1 = (1.0 - u_tilde) * v_tilde;
    double w2 = u_tilde * (1.0 - v_tilde);
    double w3 = u_tilde * v_tilde;

    // Bilinear interpolation
    double int_xy = w0 * int_ab + w1 * int_ab1 + w2 * int_a1b + w3 * int_a1b1;

    // Compute Jacobian if requested
    if (jacobian) {
        double d_B_d_u = (1.0 - v_tilde) * (int_a1b - int_ab) + v_tilde * (int_a1b1 - int_ab1);
        double d_B_d_v = (1.0 - u_tilde) * (int_ab1 - int_ab) + u_tilde * (int_a1b1 - int_a1b);
        Eigen::Matrix<double, 1, 2> d_B_d_g;
        d_B_d_g << d_B_d_u, d_B_d_v;
        *jacobian = d_B_d_g * d_g_d_T;
    }

    return int_xy;
}

bool LocalMapScan::check_coverage_at_point(double x, double y) const {
    // Check if the four pixels surrounding (x, y) are within image bounds
    Index root_px = get_root_pixel_coords(x, y);
    int a = root_px.first;
    int b = root_px.second;
    return (a >= 0 && b >= 0 && a < (img_width_ - 1) && b < (img_height_ - 1));
}

LocalMapScan::PixelCoords LocalMapScan::coord_to_pixel(double x, double y, Eigen::Matrix<double, 2, 3> *jacobian) const {
    Eigen::Matrix<double, 2, 3> D;
    D << 0, 1/res_, 0,
         -1/res_, 0, 0;
    Eigen::Matrix<double, 3, 3> pose2d_inv_mat = pose_.toSE2().inverse().matrix();
    Eigen::Vector2d p = D * pose2d_inv_mat * Eigen::Vector3d(x, y, 1.0) + 0.5 * Eigen::Vector2d(img_width_ - 1, img_height_ - 1);

    if (jacobian) {
        *jacobian = D * pose2d_inv_mat * lgmath::se2::point2fs(Eigen::Vector2d(x, y), 1.0);
    }

    return {p(0), p(1)};
}

LocalMapScan::Index LocalMapScan::get_root_pixel_coords(double x, double y) const {
    PixelCoords p = coord_to_pixel(x, y);
    int px = static_cast<int>(std::floor(p.first));
    int py = static_cast<int>(std::floor(p.second));
    return {px, py};
}

} // namespace ba