// loader.hpp
#pragma once

#include <string>
#include <memory>
#include <ba/scans/scan.hpp>

namespace ba {

class ScanLoader {
public:
    ScanLoader() = default;
    void add_scan(std::shared_ptr<Scan> scan) {
        scans_.emplace(scan->id(), scan);
    }

    std::shared_ptr<Scan> get_scan(int scan_id) {
        return scans_.at(scan_id);
    }

    int num_scans() const {
        return static_cast<int>(scans_.size());
    }

private:
    ankerl::unordered_dense::map<int, std::shared_ptr<Scan>> scans_;
};


} // namespace ba