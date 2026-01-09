// loader.hpp
#pragma once

#include <string>
#include <ba/scans/scan.hpp>

namespace ba {

class ScanLoader {
public:
    ScanLoader() = default;
    void add_scan(const &Scan scan) {
        scans_.emplace(scan.id(), scan);
    }

    Scan get_scan(int scan_id) {
        return scans_.at(scan_id);
    }

private:
    ankerl::unordered_dense::map<int, Scan> scans_;
};


} // namespace ba