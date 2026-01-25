#include <ba/problem/problem.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace ba {

void Problem::preload_images() {
    if (scan_indices_.empty()) {
        throw std::runtime_error("Scan indices are empty. Cannot preload images.");
    }
    std::string seq_id = opts_.seq_id;
    std::cout << "Preloading images for sequence: " << seq_id << std::endl;

    // Set up temporary folder for Gaussian-blurred images to be stored
    fs::path temp_dir = opts_.meas_path / "dr_ba_temp" / seq_id;
    fs::create_directories(temp_dir);

    // TODO: Add support for more than just local_maps
    if (opts_.input_type != "scans" && opts_.input_type != "local_maps") {
        throw std::invalid_argument("Input type " + opts_.input_type + " not supported yet.");
    }

    // Load in images
    fs::path all_img_dir = opts_.meas_path / seq_id / opts_.input_type;
    // Sort files in directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(all_img_dir)) {
        if (entry.path().extension() != ".png") {
            continue;
        }
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    // Load in cumulative return images
    std::vector<fs::path> cumul_files;
    if (opts_.use_cumul_thresh) {
        fs::path cumul_img_dir = opts_.meas_path / seq_id / "cumulated_returns";
        for (const auto& entry : fs::directory_iterator(cumul_img_dir)) {
            if (entry.is_regular_file()) {
                cumul_files.push_back(entry.path());
            }
        }
        std::sort(cumul_files.begin(), cumul_files.end());
    }

    // Set reference timestamp based on first scan time
    int64_t ref_timestamp = std::stoll(files[0].stem().string());
    scan_manager_.set_ref_timestamp(ref_timestamp);

    // Loop through all images
    int num_scans = img_paths_.size();
    opts_.end_frame = (opts_.end_frame == -1) ? (num_scans - 1) : opts_.end_frame;
    for (int idx : scan_indices_) {
        const auto& img_path = files[idx];
        std::string img_stem = img_path.stem().string(); // stem is just timestamp

        // Add timestamp to list
        int64_t timestamp = std::stoll(img_stem); // in microseconds
        timestamps_.push_back(timestamp);

        // Create temporary scan path
        std::string int_gauss_blur = std::to_string(static_cast<int>(opts_.gauss_blur_sigma));
        fs::path temp_img_path;
        if (opts_.dist_field_preproc)
            temp_img_path = temp_dir / (img_stem + "_distfield_" + int_gauss_blur + ".png");
        else
            temp_img_path = temp_dir / (img_stem + "_" + int_gauss_blur + ".png");

        // Load in image as Eigen matrix
        if (!fs::exists(temp_img_path)) {
            // Only process if the temp image does not already exist
            cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);

            if (img.empty()) {
                throw std::runtime_error("Failed to load image: " + img_path.string());
            }

            if (opts_.dist_field_preproc) {
                // Convert to float
                img.convertTo(img, CV_32F, 1.0 / 255.0);

                // Min–max normalization
                double min_val, max_val;
                cv::minMaxLoc(img, &min_val, &max_val);
                img = (img - min_val) / (max_val - min_val);

                // Invert: 1 - normalized
                img = 1.0 - img;

                // Exponential transform
                cv::exp(-2.0 * img, img);
            }

            // Apply Gaussian blur
            if (opts_.gauss_blur_sigma > 0.0) {
                int ksize = static_cast<int>(std::ceil(opts_.gauss_blur_sigma * 6)) | 1; // kernel size should be odd
                cv::GaussianBlur(img, img, cv::Size(ksize, ksize), opts_.gauss_blur_sigma);
            }

            if (opts_.dist_field_preproc) {
                // Re-normalize after blur
                double min_val, max_val;
                cv::minMaxLoc(img, &min_val, &max_val);
                img = (img - min_val) / (max_val - min_val);

                // Clip to [0, 1] (should already be in this range, but just to be safe)
                cv::threshold(img, img, 0.0, 0.0, cv::THRESH_TOZERO);
                cv::threshold(img, img, 1.0, 1.0, cv::THRESH_TRUNC);

                // Convert back to 8-bit for saving
                img.convertTo(img, CV_8U, 255.0);
            }

            // Save blurred image to temp directory for easy loading
            cv::imwrite(temp_img_path, img);
        }

        // Store path
        img_paths_.push_back(temp_img_path);
        if (opts_.use_cumul_thresh) {
            cumul_paths_.push_back(cumul_files[idx]);
        }
    }
    std::cout << "Preloaded " << img_paths_.size() << " images out of " << files.size() << " total images." << std::endl;
}





}   // namespace ba